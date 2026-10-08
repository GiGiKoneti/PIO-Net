"""
Analytical Energy & Latency Model for PIO-Net (RP2040).
Based on datasheet typical values.
"""
# RP2040 Datasheet typical currents @ 3.3V, 125 MHz
I_DORMANT = 280e-6      # A (XOSC off)
I_ACTIVE_CORES = 18e-3  # A (both Cortex-M0+ active)
I_PIO_DMA = 2.1e-3      # A estimate for PIO+DMA active
V = 3.3                 # V

# Timing
F_CLK = 125e6           # Hz
T_CLK = 1/F_CLK         # 8 ns
FRAME_BITS = 32         # kernel size
DMA_WORDS = 1024        # ping-pong buffer length
T_WARM_XOSC = 1.2e-6    # 1.2 µs wake from dormant (datasheet)
T_IRQ_OVERHEAD = 150 * T_CLK  # ~1.2 µs

def energy_per_inference(keyword_rate_hz=1.0, snr_db=20):
    """
    Returns dict with energy breakdown (Joules) for one inference cycle.
    Assumes idle most of time, keyword triggers wake.
    """
    # Idle energy per second
    P_idle = V * I_DORMANT
    # Active window per keyword
    # PIO first layer runs continuously in hardware while dormant? Actually PIO runs while CPU dormant.
    # Power during detection: PIO+DMA active, CPU dormant
    P_detect = V * (I_DORMANT + I_PIO_DMA)  # ~2.38 mA
    # Detection runs continuously; we account per second
    # But for per-inference, consider the active classification window
    T_classify = 500e-6  # assume 0.5 ms for deep layers (CMSIS-NN)
    P_classify = V * I_ACTIVE_CORES
    E_classify = P_classify * T_classify
    E_wake = P_classify * (T_WARM_XOSC + T_IRQ_OVERHEAD)
    # Energy during detection for one frame (32 bits)
    T_frame = FRAME_BITS * T_CLK  # 256 ns
    E_frame = P_detect * T_frame
    # Total per inference (one keyword event)
    E_total = E_wake + E_classify + E_frame
    # Average power at given keyword rate
    P_avg = P_idle + keyword_rate_hz * (E_wake + E_classify + E_frame)
    return {
        'P_idle_mW': P_idle*1e3,
        'P_detect_mW': P_detect*1e3,
        'E_wake_uJ': E_wake*1e6,
        'E_classify_uJ': E_classify*1e6,
        'E_frame_nJ': E_frame*1e9,
        'E_total_uJ': E_total*1e6,
        'P_avg_mW': P_avg*1e3,
    }

def baseline_cmsisnn_energy(keyword_rate_hz=1.0):
    # CPU active continuously polling I2S + inference
    # Assume 12% active duty (from blueprint)
    duty = 0.12
    P_active = V * I_ACTIVE_CORES
    P_avg = duty * P_active + (1-duty) * V * I_DORMANT
    # Energy per inference: active window 450 µJ (blueprint)
    E_inf = 450e-6
    return {
        'P_avg_mW': P_avg*1e3,
        'E_inf_uJ': E_inf*1e6,
    }

if __name__ == "__main__":
    print("=== PIO-Net Energy Model ===")
    for rate in [0.1, 1, 10]:
        e = energy_per_inference(keyword_rate_hz=rate)
        print(f"Keyword rate {rate}/s: avg power {e['P_avg_mW']:.3f} mW, "
              f"E/inf {e['E_total_uJ']:.2f} µJ")
    print("\n=== CMSIS-NN Baseline ===")
    b = baseline_cmsisnn_energy()
    print(f"Avg power {b['P_avg_mW']:.2f} mW, E/inf {b['E_inf_uJ']:.0f} µJ")