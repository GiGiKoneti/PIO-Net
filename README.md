# 🎙️ PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on RP2040 Microcontrollers

[![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen)](verification/)
[![Target Conference](https://img.shields.io/badge/Target-ACM%20SenSys%202027-blue)](https://sensys27.hotcrp.com/)
[![Target Symposium](https://img.shields.io/badge/Target-IIT%20Bombay%20ResCon%202026-orange)](https://unstop.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Gigi Koneti\*, Adarsha Udupa Baikady\*, Anukul Anand\*  
*\*All authors contributed equally to this work (Co-First Authorship).*

---

## 📌 Overview

**PIO-Net** is an ultra-low-power, two-stage cascaded keyword wake-up system designed for the Raspberry Pi RP2040 microcontroller.

Continuous acoustic keyword spotting (KWS) typically demands tens of milliwatts, draining portable batteries within hours. PIO-Net solves this by decoupling acoustic tripwire monitoring and audio ingestion from the main CPU using a **Two-Gear Dynamic Voltage and Frequency Scaling (DVFS)** architecture with a **Deferred Inference Burst** co-designed around the RP2040's **Programmable I/O (PIO)** hardware subsystem:

1. **Quiescent Listening Gear (24 MHz):**
   - System clock is downscaled to **24 MHz** with all unused peripherals gated (`pll_usb` off, `clk_adc` stopped, UART/SPI/I2C/PWM held in reset).
   - **Stage 1 (Hardware 1-Bit Energy Gate):** PIO1 evaluates a dual-tier energy gate in pure silicon ($\ge 19$ ones within a 32-clock window, requiring 12 consecutive qualifying blocks for ~0.2–0.4 ms persistence) directly on the incoming 1.024 MHz PDM bitstream.
   - Core 0 remains in deep `__wfi()` sleep with **0 Hz CPU wakeups** during silence and ambient noise, drawing only **7.00 mA (28.56 mW @ 4.08V)**.

2. **Low-Power Audio Ingestion & Decimation (24 MHz):**
   - On an acoustic match, PIO1 asserts an interrupt waking Core 0.
   - **Crucially, clock upscaling is deferred:** Core 0 remains at **24 MHz** during the entire **850 ms** audio capture window (with 150 ms pre-roll preserved in the DMA circular ring buffer to prevent onset clipping).
   - Core 0 wakes every 8 ms via hardware timer alarms to decimate chunks (executing at an ultra-low **~2.3% CPU duty cycle** and sleeping in `__wfi()` for >97.5% of the time). Current draw is **~7.50 mA (30.60 mW)**, expending only **$E_{\text{cap}} \approx 26.00\text{ mJ}$** (a **63.9% cut** compared to 125 MHz continuous ingestion).

3. **Active Inference Burst Gear (125 MHz):**
   - Once the audio frame is captured, the system triggers a glitchless frequency transition (safely updating PIO dividers before upscaling) to **125 MHz**.
   - **Stage 2 (On-Demand M0+ Classifier):** Core 0 extracts a [39 frames $\times$ 16 Mel bins] spectrogram in **Feat ~87 ms**, and executes a 3,636-parameter Q12 fixed-point depthwise separable convolutional neural network (`DSCNNMicro`) in **Infer ~24 ms at 125 MHz** (~111.4 ms total compute burst consuming **10.43 mJ**).
   - Total end-to-end wake event energy is only **$E_{\text{event}} = 26.00\text{ mJ} + 10.43\text{ mJ} = 36.43\text{ mJ}$** (incremental energy $\Delta E_{\text{wake}} = 8.97\text{ mJ}$).
   - Immediately upon decision completion, the clock downscales back to 24 MHz and Core 0 returns to quiescent sleep.

```
[Digital PDM Microphone (SPH0641LU4H-1)]
           │ (1.024 MHz 1-bit PDM stream)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ RP2040 PIO Subsystem (PIO0 + PIO1 @ 24 MHz)                 │
│ • PIO0 SM0: 1.024 MHz PDM bitstream reader -> DMA circular  │
│ • PIO1 SM0: Autonomous 1-density energy tripwire in silicon  │
│   (>= 19 ones / 32 clocks, 12 consecutive blocks ~0.375 ms) │
└──────────────────────────┬──────────────────────────────────┘
                           │ (Acoustic Tripwire Qualified?)
                           ├────────────── No  ──> Stay in __wfi() (0 Hz CPU wakeups, 7.00 mA)
                           ▼ Yes (PIO1 IRQ Asserted)
┌─────────────────────────────────────────────────────────────┐
│ Ingestion Phase (Core 0 remains at 24 MHz - DEFERRED BURST) │
│ • 850 ms capture window + 150 ms pre-roll in 32 KB DMA ring │
│ • Core 0 wakes every 8 ms via timer alarm (~2.3% duty cycle)│
│ • Draws ~7.50 mA (26.00 mJ) vs. 71.96 mJ at 125 MHz         │
└──────────────────────────┬──────────────────────────────────┘
                           │ Capture Complete (850 ms elapsed)
                           ▼ Glitchless PLL Frequency Switch (24 -> 125 MHz)
┌─────────────────────────────────────────────────────────────┐
│ Stage-2 Compute Phase (Core 0 @ 125 MHz Burst)              │
│ • Mel Spectrogram: 16 Mel bins, 32-bit BFP FFT (87.0 ms)    │
│ • Classifier: 3,636-param Q12 fixed-point DSCNN (24.0 ms)   │
│ • Total Compute Duration: 111.4 ms (10.43 mJ energy)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ├─> Keyword Detected: Application callback
                           ▼
        [Glitchless Downscale to 24 MHz & Return to Quiescent WFI Sleep]
```

---

## 🚀 Key Results & Performance Benchmark

### Empirical Hardware Telemetry (INA226 External Power Monitor @ VSYS = 4.08V)

Physical supply voltage and current were captured in real-time using a calibrated high-side INA226 current sensor sampled by an external DAQ over dedicated hardware runs across comparison configurations:

| Metric | Config A: Continuous Software VAD | Config B: Duty-Cycled Wake (125 MHz) | Config B-Low: Duty-Cycled (24 MHz) | Config C: PIO-Net (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **Quiescent System Clock** | 125 MHz | 125 MHz | 24 MHz | **24 MHz** |
| **Burst System Clock** | 125 MHz | 125 MHz | 125 MHz | **125 MHz (Deferred)** |
| **CPU Wake Rate in Silence** | Continuous Polling | 100 Hz Timer Wakeups | 100 Hz Timer Wakeups | **0 Hz (Zero CPU Wakeups)** |
| **Quiescent Current ($I_{\text{listen}}$)** | 23.90 mA | 20.55 mA | 11.20 mA | **7.00 mA** |
| **Quiescent Power ($P_{\text{listen}}$)** | 97.51 mW | 83.84 mW | 45.70 mW | **28.56 mW** |
| **Idle Savings vs. Config A** | *Baseline (0%)* | 14.0% saved | 53.1% saved | **70.7% saved ($3.41\times$ lower)** |
| **Idle Savings vs. Config B-Low**| — | — | *Baseline (0%)* | **37.5% saved ($1.60\times$ lower)** |
| **Active Burst Power ($P_{\text{burst}}$)** | 99.6 mW | 99.1 mW | 100.4 mW | **100.36 ± 1.57 mW** |
| **Capture Current / Energy (850 ms)** | 24.42 mA (84.8 mJ) | 24.30 mA (84.4 mJ) | 24.65 mA (84.4 mJ) | **~7.50 mA (~26.00 mJ)** |
| **Compute Burst Energy ($E_{\text{compute}}$)** | Continuous | 10.82 mJ | 10.82 mJ | **10.43 ± 0.65 mJ** |
| **Total Event Energy ($E_{\text{event}}$)** | Continuous | 82.78 mJ | 82.78 mJ | **36.43 mJ ($\Delta E_{\text{wake}} = 8.97\text{ mJ}$)** |
| **Model Format & Parameters** | INT8 (3,892 params) | INT8 (3,892 params) | INT8 (3,892 params) | **Q12 Fixed-Point (3,636 params)** |
| **Flash Footprint** | 4.10 KB | 4.10 KB | 4.10 KB | **7.30 KB** |
| **Peak SRAM Allocation** | 6.25 KB | 6.25 KB | 6.25 KB | **15.36 KB (< 6% of RP2040 SRAM)** |

---

### Operational Battery Lifetime Comparison

Modeled across standard electrochemical cells (500 mAh 3.7V LiPo @ $\eta = 88\%$, 1.85 Wh; 850 mAh 2×AAA @ $\eta = 85\%$, 2.55 Wh) across acoustic operational environments:

| Operating Scenario | Wake Rate | Effective Power | 500 mAh LiPo | 2× AAA Alkaline | Gain vs. Continuous |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Quiet Room / Office** | 0.01 /s (36 wakes/h) | **28.65 mW** | **56.8 hours** | **75.7 hours** | **$3.40\times$** |
| **Active Domestic Home** | 0.05 /s (180 wakes/h) | **28.86 mW** | **56.4 hours** | **75.1 hours** | **$3.38\times$** |
| **Street / Loud TV Babble** | 0.10 /s (360 wakes/h) | **29.46 mW** | **55.3 hours** | **73.6 hours** | **$3.31\times$** |
| **Saturated Continuous Noise** | 1.04 /s (3,745 wakes/h max) | **37.89 mW** | **43.0 hours** | **57.2 hours** | **$2.57\times$ (61.1% cut vs. 97.5 mW)** |
| **Config A (Continuous Polling)** | Continuous | 97.50 mW | 16.7 hours | 22.2 hours | *Baseline ($1.00\times$)* |

> **Key Architectural Takeaways:**
> 1. **Zero-CPU Tripwire:** PIO-Net slashes quiescent listening power by **70.7%** ($3.41\times$) vs continuous polling, and **37.5%** vs 24 MHz software polling (Config B-Low), purely by offloading the bitstream energy check to silicon.
> 2. **Deferred DVFS Burst:** Deferring the 125 MHz clock upscale to the 111.4 ms neural inference phase drops capture energy from 71.96 mJ to 26.00 mJ, shrinking total event energy from 83.16 mJ to **36.43 mJ** (>56% reduction).
> 3. **Mathematical Bounding:** Even in worst-case saturated noise ($f_{\text{max}} = 1.040\text{ s}^{-1}$), system power saturates at **37.89 mW**—maintaining a **61.1% power advantage** over continuous polling. The break-even rate is $f_{\text{break}} = 7.685\text{ s}^{-1}$ (27,668 wakes/h), which is physically impossible to exceed.

---

### Pipeline Stage & Latency Audit (RP2040 Cortex-M0+)

| Pipeline Stage / Metric | Basis | Measured Value | Implementation / Hardware Setup |
|---|---|---|---|
| **Stage-1 Evaluation Window** | **HARDWARE** | $\ge 19$ ones within 32 clocks | Inner sub-block threshold evaluated in PIO1 (`pdm_stage1.pio`) |
| **Stage-1 Persistence Gate** | **HARDWARE** | 12 consecutive blocks | Persistence window: ~0.375 ms at 1.024 MHz PDM clock before IRQ assert |
| **Quiescent CPU Duty Cycle** | **MEASURED** | **0.0%** | Core 0 in `__wfi()`, circular DMA keeps microphone clocked (0 CPU IRQs) |
| **Capture CPU Duty Cycle** | **MEASURED** | **~2.3%** | Core 0 wakes every 8 ms for boxcar decimation, remaining in `__wfi()` >97.5% |
| **Audio Capture Window** | **MEASURED** | 850 ms | 13,600 PCM samples decimated at 64x popcount from 1.024 MHz stream |
| **Stage-2 Feature Extraction** | **MEASURED** | 87.0 ms | [39 $\times$ 16] Mel-spectrogram with 32-bit BFP FFT on Cortex-M0+ @ 125 MHz |
| **Stage-2 Classifier Inference** | **MEASURED** | 24.0 ms | 3,636-param Q12 DSCNN inference @ 125 MHz on Cortex-M0+ hardware multiplier |
| **Total Inference Compute** | **MEASURED** | **111.4 ms** | Stage-2 Feature extraction + neural classifier at 125 MHz burst |
| **Total Wake Event Duration** | **MEASURED** | 961.4 ms | 850 ms capture (24 MHz) + 111.4 ms compute (125 MHz) |
| **Balanced Keyword Recall** | **MEASURED** | **90.8%** | Macro-averaged recall across all 10 keywords (including 2.4% tripwire miss) |
| **False Rejection / Wakes** | **MEASURED** | **Zero in 72 h** | Zero false keyword classifications across 72 hours of calibrated noise suite |

---

## 📂 Repository Structure

```
PIO-Net/
├── ahb_sim/                 # AHB 4-port crossbar cycle-accurate bus simulator
├── baseline/                # 3rd-order CIC decimator, Energy VAD, and WebRTC VAD
│   └── vad_reference.py     # System configurations power model (Config A, B, PIO-Net)
├── dataset/                 # Dataset curation and PDM conversion
│   ├── download_curate.py   # Benchmark curation & noise suite generator
│   └── pcm_to_pdm.py        # 2nd-order Sigma-Delta 1.024 MHz PDM modulator
├── dma_sim/                 # DMA ping-pong controller cycle-accurate simulator
├── docs/                    # Research reports, surveys, and publication figures
│   ├── figures/             # Camera-ready vector PDFs and high-res PNGs
│   │   ├── pionet_confusion_matrix.pdf
│   │   ├── pionet_det_curves.pdf
│   │   └── pionet_energy_waterfall.pdf
│   ├── Gigi_Research_and_Implementation_Report.md  # Comprehensive research chronicle
│   ├── Detection_Metrics_Report.md
│   ├── Noise_Robustness_Report.md
│   └── Go_NoGo_Checkpoint_Report.md
├── energy_model/            # Physical RP2040 dynamic & static energy profiler
│   ├── clap_detector.py     # Real-time streaming INA226 current/voltage telemetry parser
│   └── yes_test_log.txt     # Empirical hardware INA226 capture logs
├── evaluation/              # Sim-to-real noise robustness and visual generation
│   ├── evaluate_noise_robustness.py  # ITU-T P.56 active speech & RIR SNR sweeps
│   └── generate_detection_metrics.py # ACM publication figures generator
├── nas/                     # Stage-1 matched filter optimization
│   ├── train_nas.py         # Differentiable PyTorch STE with Dirichlet regularization
│   ├── pionet_pattern.h     # Exported C header for RP2040 PIO
│   └── pionet_pattern.json  # Metadata and threshold calibration
├── paper/                   # Camera-ready ACM LaTeX snippets
│   └── tables_and_figures.tex
├── pio_sim/                 # RP2040 PIO state machine cycle-accurate simulator
├── pionet_comparisons/      # Standalone SenSys 2027 benchmark targets (Config A, B, C)
│   ├── CMakeLists.txt       # Unified Ninja/CMake build file
│   ├── config_a_cpu_continuous.c # 125 MHz 100% active polling baseline
│   ├── config_b_dutycycle.c # 125 MHz / 24 MHz periodic timer buffer-wake baseline
│   ├── config_c_pionet.c    # 24 MHz -> 125 MHz Two-Gear DVFS with Deferred Burst
│   └── README.md            # Empirical measurement protocol and verification data
├── pionet_core/             # Production RP2040 C firmware implementation
│   ├── baseline_wfi.c       # Pure baseline WFI power benchmark target (1.3 mA floor)
│   ├── CMakeLists.txt       # Firmware build with clock wrap & peripheral gating
│   ├── main.c               # Two-gear DVFS executive, PDM DMA ring, and Stage-2 driver
│   ├── main.h               # Configuration macros, pin definitions, and DMA parameters
│   ├── model_data.h         # Q12 fixed-point DSCNNMicro weights and biases (3,636 params)
│   ├── pdm_reader.pio       # PIO0 state machine: 1.024 MHz PDM clock & bit capture
│   ├── pdm_stage1.pio       # PIO1 state machine: Autonomous 1-bit temporal energy gate
│   ├── stage2_features.c    # 16-channel Mel-filterbank with Radix-2 BFP FFT
│   ├── stage2_features.h    # Feature extraction interfaces
│   ├── stage2_infer.c       # Q12 CMSIS-NN depthwise separable CNN inference engine
│   └── stage2_infer.h       # Classifier interfaces
├── stage2/                  # Stage-2 DSCNNMicro neural network
│   ├── model.py             # PyTorch architecture
│   ├── train_stage2.py      # Training, BatchNorm folding, and CMSIS-NN exporter
│   ├── model_data.h         # Exported Q12 CMSIS-NN C header
│   └── model_config.json    # Hardware memory & latency budget audit
├── tools/                   # Automated data loggers and PDF revision utilities
│   ├── apply_path_b_perfect.py # Automated camera-ready PDF compiler
│   └── stm32_auto_logger.py    # Automated multi-board INA226 DAQ power logger
├── verification/            # Automated test suite (34 unit and system tests)
│   ├── gonogo_checkpoint.py # Official Go/No-Go feasibility evaluator
│   └── test_*.py            # PyTest automated unit test modules
├── Makefile                 # One-command build and verification entrypoint
├── .gitignore               # Strict exclude list for dataset bloat and caches
└── README.md
```

---

## 🛠️ Quickstart & Reproduction

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/adarshaudupa/PIO-Net.git
cd PIO-Net
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch numpy scipy soundfile matplotlib pytest
```

### 2. Run Complete Verification Suite (34/34 Passing)
```bash
make verify
```

### 3. Build & Flash Firmware Targets
To replicate the SenSys 2027 power benchmark on an RP2040 board with an INA226 current monitor:

#### Option A: Build Comparison Suite
```bash
cd pionet_comparisons
mkdir build && cd build
cmake -G Ninja -DPICO_BOARD=pico ..
ninja
```
Generated targets:
- `config_a_continuous.uf2`: 125 MHz Continuous Software Polling (measures ~23.90 mA)
- `config_b_dutycycle.uf2`: 125 MHz Duty-Cycled 10ms Wake (measures ~20.55 mA)
- `config_c_pionet.uf2`: Two-Gear DVFS with Deferred Burst (measures ~7.00 mA quiescent, ~7.50 mA capture, 24.65 mA burst)

#### Option B: Build Core Production Firmware
```bash
cd pionet_core
mkdir build && cd build
cmake -G Ninja -DPICO_BOARD=pico ..
ninja
```
Generated targets:
- `pionet_core.uf2`: Production wake-on-sound engine with deferred 125 MHz burst and Q12 DSCNN classifier
- `pico_wfi_baseline.uf2`: Pure `__wfi()` sleep baseline with Core 1 and all clocks off (measures ~1.3–1.5 mA)

---

## 📄 Citation & Attribution

This project is submitted to ACM SenSys 2027.

```bibtex
@inproceedings{koneti2027pionet,
  title     = {PIO-Net: Acoustic Wake-on-Sound for Commodity Microcontrollers Using Programmable I/O and Two-Gear Clock Scaling},
  author    = {Koneti, Gigi and Baikady, Adarsha Udupa and Anand, Anukul},
  booktitle = {Proceedings of the International Conference on Embedded Artificial Intelligence and Sensing Systems (ACM SenSys)},
  year      = {2027}
}
```
