# 📊 PIO-Net Task 4.2: Detection Metrics & Publication Visuals Report

**Target Publication:** ACM SenSys 2027 / IIT Bombay ResCon 2026  
**Figure Dimensions:** Strictly ACM Single-Column (3.33 in / 8.45 cm width, 300 DPI vector PDF)  
**Color Palette:** Okabe-Ito (Color Universal Design) + Perceptually Uniform Blues  
**Classification Evaluation:** Macro Recall / Balanced Accuracy: 91.0%  

---

## 1. Generated Publication Figures

| Figure Name | File Path | ACM Dimensional Width | Core Scientific Purpose |
|---|---|---|---|
| **Confusion Matrix** | [`docs/figures/pionet_confusion_matrix.pdf`](figures/pionet_confusion_matrix.pdf) | 3.33 in (8.45 cm) | 12-Class normalized recall (90.8% macro recall) with integer rounding note & keyword partition |
| **DET / ROC Curves** | [`docs/figures/pionet_det_curves.pdf`](figures/pionet_det_curves.pdf) | 3.33 in (8.45 cm) | Log-log Miss Rate vs False Wakes/hr with secondary $P_{\mathrm{FA}}$ axis (per 1.0-s inference) |
| **Energy Waterfall** | [`docs/figures/pionet_energy_waterfall.pdf`](figures/pionet_energy_waterfall.pdf) | 3.33 in (8.45 cm) | 72.4% power drop & usable electrochemical lifetime on CR2032, LiPo, and 2x AAA |

---

## 2. End-to-End System Performance Benchmark

| Architectural Configuration | VAD Tripwire | CPU Active Duty | Average Current | Average Power | 2x AAA Lifetime | Energy Savings |
|---|---|---|---|---|---|---|
| **Config A (Continuous CPU)** | WebRTC VAD | 100.0% | 19.70 mA | 65.01 mW | 39.2 hours (1.6 days) | Baseline |
| **Config B (Periodic 20 ms)** | Energy VAD | 12.6% | 8.08 mA | 26.67 mW | 95.6 hours (4.0 days) | 58.9% |
| **PIO-Net (Proposed)** | **1-Bit PIO Matcher** | **0.0% (quiescent)** | **5.44 mA** | **17.95 mW** | **142.1 hours (5.9 days)** | 🟢 **72.4%** |

---

## 3. LaTeX Snippets Exported
Ready-to-use LaTeX code exported to [`paper/tables_and_figures.tex`](file:///home/gigikoneti/SenSys2027/paper/tables_and_figures.tex) containing:
- `\begin{figure}` blocks for all three figures formatted for `acmart.cls` with refined scientific captions.
- Formatted LaTeX `\begin{table}` summary comparing battery life and power.

🎉 **Task 4.2 Completed Successfully!**