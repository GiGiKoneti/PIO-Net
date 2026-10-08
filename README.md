# 🎙️ PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on RP2040 Microcontrollers

[![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen)](verification/)
[![Target Conference](https://img.shields.io/badge/Target-ACM%20SenSys%202027-blue)](https://sensys27.hotcrp.com/)
[![Target Symposium](https://img.shields.io/badge/Target-IIT%20Bombay%20ResCon%202026-orange)](https://unstop.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Gigi Koneti\*, Adarsh\*, Anukul\*  
*\*All authors contributed equally to this work (Co-First Authorship).*

---

## 📌 Overview

**PIO-Net** is an ultra-low-power, two-stage cascaded keyword wake-up system designed for the Raspberry Pi RP2040 microcontroller. 

Continuous acoustic keyword spotting (KWS) typically demands tens of milliwatts, draining portable batteries in days. PIO-Net solves this by offloading acoustic tripwire monitoring to the RP2040's **Programmable I/O (PIO)** hardware subsystem:
1. **Stage 1 (Hardware 1-Bit Matcher):** The PIO state machine directly ingests 1.024 MHz 1-bit Pulse Density Modulation (PDM) streams from a digital microphone and evaluates a 32-bit popcount matched filter (`0xFFFFFFFF`, $\theta = 22$) in hardware at **1 clock cycle per bit**, while the main CPU stays in deep sleep.
2. **Stage 2 (On-Demand M0+ Classifier):** When triggered by a PIO interrupt, ARM Cortex-M0+ Core 0 wakes up and executes an INT8 quantized depthwise separable convolutional neural network (`DSCNNMicro`) in **6.44 ms** using **6.25 KB SRAM**.

```
[Digital PDM Microphone]
           │ (1.024 MHz 1-bit PDM stream)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ RP2040 PIO Subsystem (State Machine 0)                      │
│ • 1-cycle/bit popcount match filter                         │
│ • Match Pattern: 0xFFFFFFFF (32 ones)                       │
│ • Decision Threshold: θ = 22 / 32                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ (Match Count >= 22?)
                           ├────────────── No  ──> Stay asleep (0% CPU load)
                           ▼ Yes (Acoustic Tripwire Fired)
┌─────────────────────────────────────────────────────────────┐
│ DMA Controller (Circular Ping-Pong Buffer)                  │
│ • Transfers audio words into SRAM Bank 1                    │
│ • Zero AHB crossbar contention with CPU execution           │
└──────────────────────────┬──────────────────────────────────┘
                           │ Wake-up IRQ
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ARM Cortex-M0+ Core 0 @ 125 MHz (Wakes from Deep Sleep)     │
│ • Feature Extractor: [39 frames, 16 Mel bins] Spectrogram   │
│ • Stage-2 Classifier: DSCNNMicro (Folded BatchNorm)         │
│ • Quantization: CMSIS-NN INT8 (< 6.4 KB SRAM, 4.1 KB Flash) │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                [Target Keyword Detected]
```

---

## 🚀 Key Results & Performance Benchmark

| Architectural Pipeline | VAD Tripwire | CPU Duty Cycle | Power (mW) | 2x AAA Lifetime | Macro Recall | False-Wake Rate |
|---|---|---|---|---|---|---|
| **Config A (Continuous CPU)** | WebRTC VAD | 100.0% | 65.01 mW | 39.2 h (1.6 days) | 90.8% | Baseline |
| **Config B (Periodic 20ms)** | Energy VAD | 12.6% | 26.67 mW | 95.6 h (4.0 days) | 88.2% | Baseline |
| **PIO-Net (Cascaded Ours)** | **1-Bit PIO Matcher** | **0.0% (quiescent)** | **17.95 mW** | **142.1 h (5.9 days)** | **90.8%** | **0.0 / hour** |

* **72.4% Power Reduction:** From 65.01 mW down to 17.95 mW.
* **142.1 Hours ($5.9$ Days) on 2x AAA:** $3.6\times$ extension over continuous streaming.
* **$99.74\%$ Noise Variance Suppression:** Proved analytically and empirically.
* **Zero Bus Contention:** Verified across 5,000 AHB crossbar arbitration cycles.
* **Fast Wake Latency:** Post-wake inference takes only 6.44 ms.

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
├── stage2/                  # Stage-2 DSCNNMicro neural network
│   ├── model.py             # PyTorch architecture
│   ├── train_stage2.py      # Training, BatchNorm folding, and CMSIS-NN exporter
│   ├── model_data.h         # Exported INT8 CMSIS-NN C header
│   └── model_config.json    # Hardware memory & latency budget audit
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
git clone https://github.com/gigikoneti/PIO-Net.git
cd PIO-Net
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy soundfile matplotlib pytest
```

### 2. Run Complete Verification Suite (34/34 Passing)
```bash
make verify
```

### 3. Run Microarchitecture Simulators (PIO, DMA, AHB)
```bash
make sim
```

### 4. Regenerate All Publication Figures
```bash
python evaluation/generate_detection_metrics.py
```
Outputs vector PDFs and high-resolution PNGs directly to `docs/figures/`.

---

## 📄 License & Citation

This project is licensed under the MIT License.

```bibtex
@inproceedings{koneti2027pionet,
  title     = {PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on Microcontrollers via Programmable I/O Gating},
  author    = {Koneti, Gigi and Adarsh and Anukul},
  booktitle = {Proceedings of the International Conference on Embedded Artificial Intelligence and Sensing Systems (ACM SenSys)},
  year      = {2027}
}
```
