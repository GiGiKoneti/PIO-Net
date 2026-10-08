# 📊 PIO-Net Sim-to-Real Acoustic Noise Robustness Report

**Audited Standard:** ITU-T P.56 Active Speech Level Normalization & RIR ($T_{60} = 0.20\text{s}$)
**Target Microcontroller:** ARM Cortex-M0+ (RP2040 @ 125 MHz)

## 1. Parametric SNR Degradation Tables

### Environment: `office_babble`
| SNR (dB) | PIO-Net Stage-1 Wake (%) | PIO-Net End-to-End (%) | Config A (WebRTC VAD) (%) | Config B (Energy VAD) (%) |
|---|---|---|---|---|
| ** -5 dB** | 92.5% | 92.5% | 95.0% | 100.0% |
| ** +0 dB** | 85.0% | 85.0% | 100.0% | 100.0% |
| ** +5 dB** | 85.0% | 85.0% | 100.0% | 100.0% |
| **+10 dB** | 77.5% | 77.5% | 100.0% | 100.0% |
| **+15 dB** | 75.0% | 75.0% | 100.0% | 100.0% |
| **+20 dB** | 75.0% | 75.0% | 100.0% | 100.0% |

### Environment: `factory_machinery`
| SNR (dB) | PIO-Net Stage-1 Wake (%) | PIO-Net End-to-End (%) | Config A (WebRTC VAD) (%) | Config B (Energy VAD) (%) |
|---|---|---|---|---|
| ** -5 dB** | 92.5% | 92.5% | 100.0% | 100.0% |
| ** +0 dB** | 87.5% | 87.5% | 100.0% | 100.0% |
| ** +5 dB** | 85.0% | 85.0% | 100.0% | 100.0% |
| **+10 dB** | 77.5% | 77.5% | 100.0% | 100.0% |
| **+15 dB** | 77.5% | 77.5% | 100.0% | 100.0% |
| **+20 dB** | 75.0% | 75.0% | 100.0% | 100.0% |

### Environment: `stationary_ambient`
| SNR (dB) | PIO-Net Stage-1 Wake (%) | PIO-Net End-to-End (%) | Config A (WebRTC VAD) (%) | Config B (Energy VAD) (%) |
|---|---|---|---|---|
| ** -5 dB** | 97.5% | 97.5% | 100.0% | 100.0% |
| ** +0 dB** | 90.0% | 90.0% | 100.0% | 100.0% |
| ** +5 dB** | 85.0% | 85.0% | 100.0% | 100.0% |
| **+10 dB** | 82.5% | 82.5% | 100.0% | 100.0% |
| **+15 dB** | 75.0% | 75.0% | 100.0% | 100.0% |
| **+20 dB** | 75.0% | 75.0% | 100.0% | 100.0% |

## 2. Quiescent False-Wake Rate (FWR) & Energy Waterfall Model
| Acoustic Environment | Stage-1 Wakes/h | End-to-End False Keywords/h | Cascaded System Power | Energy Savings vs Config A |
|---|---|---|---|---|
| `office_babble` | 7200.0 /h | **0.0 /h** | **17.95 mW** | 🟢 **72.4%** |
| `factory_machinery` | 7200.0 /h | **0.0 /h** | **17.95 mW** | 🟢 **72.4%** |
| `stationary_ambient` | 7200.0 /h | **0.0 /h** | **17.95 mW** | 🟢 **72.4%** |

---
*Generated automatically for ACM SenSys 2027 & IIT Bombay ResCon 2026.*