#!/usr/bin/env python3
"""
STM32 INA226 Serial Logger — Continuous high-integrity power measurement capture.

Logs the STM32 INA226 serial stream at full hardware speed, providing:
  - Live PuTTY-style terminal pass-through
  - Clean CSV with validated readings
  - Automatic summary statistics on exit

Usage:
  python stm32_auto_logger.py --port COM3 --label config_c_deferred_burst
  python stm32_auto_logger.py --label config_b_50ms --duration 10
  python stm32_auto_logger.py  # auto-detect port, default label
"""

import argparse
import math
import os
import re
import signal
import sys
import time
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Physical plausibility bounds for INA226 readings on Pico power rail
# ---------------------------------------------------------------------------
V_MIN, V_MAX = 2.5, 5.0       # Bus voltage (V)
I_MIN, I_MAX = 0.0, 200.0     # Current (mA)
P_MIN, P_MAX = 0.0, 1000.0    # Power (mW)

# Regex patterns for common INA226 serial output formats
# Pattern 1: "V: 3.30  I: 12.45  P: 41.09"
PATTERN_VIP = re.compile(
    r'V\s*[:=]\s*(?P<v>[\d.]+)\s+'
    r'I\s*[:=]\s*(?P<i>[\d.]+)\s+'
    r'P\s*[:=]\s*(?P<p>[\d.]+)',
    re.IGNORECASE
)
# Pattern 2: CSV-like "3.30,12.45,41.09" or "3.30, 12.45, 41.09"
PATTERN_CSV = re.compile(
    r'^\s*(?P<v>[\d.]+)\s*[,\t]\s*(?P<i>[\d.]+)\s*[,\t]\s*(?P<p>[\d.]+)\s*$'
)
# Pattern 3: "ts_ms, V, I, P" — with optional MCU timestamp prefix
PATTERN_TS_VIP = re.compile(
    r'^\s*(?P<ts>\d+)\s*[,\t]\s*(?P<v>[\d.]+)\s*[,\t]\s*(?P<i>[\d.]+)\s*[,\t]\s*(?P<p>[\d.]+)\s*$'
)


def auto_detect_port():
    """Find the first available serial port (prefer STM32 VCP)."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    # Prefer ports with STM32 or ST-Link in description
    for p in ports:
        desc = (p.description or "").lower()
        if "stm" in desc or "st-link" in desc or "virtual com" in desc:
            return p.device
    # Fallback: first port
    return ports[0].device


def parse_line(raw_line):
    """
    Parse a single serial line for INA226 V/I/P readings.
    Returns (mcu_ts_ms, vbus_V, current_mA, power_mW) or None if invalid.
    mcu_ts_ms is None if the line has no MCU timestamp field.
    """
    line = raw_line.strip()
    if not line:
        return None

    mcu_ts = None
    v = i = p = None

    # Try timestamp + V/I/P pattern first
    m = PATTERN_TS_VIP.match(line)
    if m:
        mcu_ts = int(m.group("ts"))
        v, i, p = float(m.group("v")), float(m.group("i")), float(m.group("p"))
    else:
        # Try V: I: P: pattern
        m = PATTERN_VIP.search(line)
        if m:
            v, i, p = float(m.group("v")), float(m.group("i")), float(m.group("p"))
        else:
            # Try plain CSV
            m = PATTERN_CSV.match(line)
            if m:
                v, i, p = float(m.group("v")), float(m.group("i")), float(m.group("p"))

    if v is None:
        return None

    # Physical plausibility check
    if not (V_MIN <= v <= V_MAX):
        return None
    if not (I_MIN <= i <= I_MAX):
        return None
    if not (P_MIN <= p <= P_MAX):
        return None

    return (mcu_ts, v, i, p)


def run_logger(port, baud, label, duration):
    """Main logging loop."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    raw_path = os.path.join(log_dir, f"{label}_{timestamp_str}_raw.log")
    csv_path = os.path.join(log_dir, f"{label}_{timestamp_str}_clean.csv")
    summary_path = os.path.join(log_dir, f"{label}_summary.txt")

    print(f"┌─────────────────────────────────────────────────────")
    print(f"│ STM32 INA226 Serial Logger")
    print(f"│ Port: {port}  Baud: {baud}")
    print(f"│ Label: {label}")
    print(f"│ Duration: {'continuous (Ctrl+C to stop)' if duration is None else f'{duration}s'}")
    print(f"│ Raw log:  {raw_path}")
    print(f"│ Clean CSV: {csv_path}")
    print(f"└─────────────────────────────────────────────────────")
    print()

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: Cannot open {port}: {e}", file=sys.stderr)
        sys.exit(1)

    # Statistics accumulators
    total_lines = 0
    valid_samples = 0
    dropped_lines = 0
    currents = []
    powers = []

    stop_flag = [False]

    def signal_handler(sig, frame):
        stop_flag[0] = True

    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.monotonic()

    try:
        with open(raw_path, "w", encoding="utf-8") as f_raw, \
             open(csv_path, "w", encoding="utf-8") as f_csv:

            f_csv.write("sample_idx,host_elapsed_ms,mcu_timestamp_ms,vbus_V,current_mA,power_mW\n")

            while not stop_flag[0]:
                # Duration check
                elapsed = time.monotonic() - start_time
                if duration is not None and elapsed >= duration:
                    break

                # Read line from serial
                try:
                    raw_bytes = ser.readline()
                except serial.SerialException:
                    break

                if not raw_bytes:
                    continue

                # Decode with error handling
                try:
                    line = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    dropped_lines += 1
                    continue

                total_lines += 1

                # Live pass-through (PuTTY-style)
                print(line, flush=True)

                # Parse for valid INA226 reading
                result = parse_line(line)
                if result is None:
                    dropped_lines += 1
                    continue

                mcu_ts, v, i_ma, p_mw = result
                valid_samples += 1

                # Write raw log (only valid, clean lines)
                f_raw.write(line + "\n")

                # Write CSV
                host_elapsed_ms = (time.monotonic() - start_time) * 1000.0
                mcu_ts_str = str(mcu_ts) if mcu_ts is not None else ""
                f_csv.write(f"{valid_samples},{host_elapsed_ms:.1f},{mcu_ts_str},{v:.4f},{i_ma:.4f},{p_mw:.4f}\n")

                currents.append(i_ma)
                powers.append(p_mw)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
    finally:
        ser.close()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed_total = time.monotonic() - start_time
    n = len(currents)

    if n > 0:
        mean_i = sum(currents) / n
        var_i = sum((x - mean_i) ** 2 for x in currents) / n
        std_i = math.sqrt(var_i)
        min_i = min(currents)
        max_i = max(currents)
        mean_p = sum(powers) / n
        rate_hz = n / elapsed_total if elapsed_total > 0 else 0
    else:
        mean_i = std_i = min_i = max_i = mean_p = rate_hz = 0

    summary_lines = [
        f"═══════════════════════════════════════════════════",
        f"  STM32 INA226 Logger — Summary ({label})",
        f"═══════════════════════════════════════════════════",
        f"  Duration:           {elapsed_total:.1f} s",
        f"  Total lines read:   {total_lines}",
        f"  Valid samples:      {valid_samples}",
        f"  Dropped/corrupted:  {dropped_lines}",
        f"  Effective rate:     {rate_hz:.1f} Hz",
        f"───────────────────────────────────────────────────",
        f"  Mean current:       {mean_i:.3f} mA",
        f"  Std dev current:    {std_i:.3f} mA",
        f"  Min current:        {min_i:.3f} mA",
        f"  Max current:        {max_i:.3f} mA",
        f"  Mean power:         {mean_p:.3f} mW",
        f"═══════════════════════════════════════════════════",
    ]

    print()
    for line in summary_lines:
        print(line)

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        for line in summary_lines:
            f_sum.write(line + "\n")

    print(f"\n  Files saved:")
    print(f"    {raw_path}")
    print(f"    {csv_path}")
    print(f"    {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="STM32 INA226 Serial Logger — continuous high-integrity power measurement capture"
    )
    parser.add_argument("--port", default=None,
                        help="Serial port (e.g. COM3). Auto-detect if omitted.")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--label", default="ina226_log",
                        help="Label prefix for output files (e.g. config_c_deferred_burst)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Capture duration in seconds (continuous if omitted)")

    args = parser.parse_args()

    port = args.port
    if port is None:
        port = auto_detect_port()
        if port is None:
            print("ERROR: No serial ports found. Connect STM32 and retry, or specify --port.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    run_logger(port, args.baud, args.label, args.duration)


if __name__ == "__main__":
    main()

