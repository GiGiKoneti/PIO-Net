# PIO-Net: Peripheral-Fabric Acoustic Wake-Up on the RP2040

Target: ACM SenSys 2027
Status as of 7 Oct 2026: design agreed in principle, nothing implemented or measured yet.

---

## 1. The idea in one paragraph

Always-on acoustic sensing wastes energy because the CPU must repeatedly wake to screen audio that is mostly silence or noise. PIO-Net moves first-stage screening into the RP2040's programmable I/O (PIO) and DMA fabric. A PIO state machine clocks a 1-bit PDM MEMS microphone, and DMA streams the bits into an SRAM ring buffer. A second state machine compares the stream against a stored 32-bit binary pattern and raises an interrupt when the match count crosses a threshold. PIO has no ALU, so XNOR and popcount are built from conditional jumps and decrement-and-branch instructions. While this runs, the Cortex-M0+ sleeps with the system clock still running (the PIO is clocked from it). On wake, the CPU runs a small quantized classifier on the buffered audio.

The contribution is a **measured characterization**: how much a general-purpose programmable peripheral (PIO + DMA) can offload from the CPU on a chip with no audio hardware, what it costs in detection quality, and whether it actually saves power.

## 2. Pipeline

1. **Stage 0 (PIO SM A):** generates the mic clock, samples the data pin, autopushes 32-bit words.
2. **DMA:** PIO RX FIFO to SRAM ring buffer (also keeps pre-trigger audio for stage 2).
3. **Stage 1 (PIO SM B):** matches the bitstream against a 32-bit pattern, raises a PIO IRQ at threshold.
4. **CPU:** sleeps (WFI) with `clk_sys` running, wakes on the IRQ.
5. **Stage 2 (CPU):** small quantized classifier on the buffered audio, then back to sleep.

## 3. What we are NOT claiming

- Not a keyword spotter in PIO. 32 instruction words per PIO block only allows a small correlator. It is a gate.
- Not "CPU dormant, PIO alive". `xosc_dormant` stops `clk_sys`, which stops the PIO. We use sleep with the clock running.
- Not "zero-CPU". The CPU is asleep, not absent, and still runs stage 2.
- No benchmark numbers from the original Gemini draft (37x, 375x, 0.92 mW, F1 0.89). They were never measured. Only bench values go in the paper.
- Not the lowest-power way to do wake-on-sound. Dedicated parts will beat the RP2040.

## 4. Criticism received and corrections made

### From Ashutosh Bhaiya
| Critique | Our response |
|---|---|
| "Why not use ADC/DSP? Isn't this reinventing the wheel?" | Dedicated options exist. Our angle: the RP2040 has no audio hardware, so we measure how much PIO + DMA can offload and what it costs in quality. Add a PDM-to-PCM + energy-VAD CPU baseline and a published-numbers table against dedicated parts. |
| "Compare power, not just offloading. Microcontrollers have a flat power curve." | Correct. CPU load is a poor metric. We compare average current and energy directly, with a clock-frequency sweep per configuration. |
| "Measure power externally, don't rely on software." | Agreed. All power numbers come from external instruments. |
| "Try an extremely small MCU like the ATtiny85." | Follow-up only, after RP2040 results exist. It has no PDM hardware. Its SRAM is, I believe, 512 bytes, not 85, but verify in the datasheet before saying so. |

### From our own review (flaws in the original draft)
| Flaw | Fix |
|---|---|
| Dormant mode stops the PIO's clock | CPU sleeps with `clk_sys` running |
| Full keyword spotter claimed in PIO | Stage 1 is a gate; CPU runs the real classifier |
| 24-bit I2S mic made "1-bit input" meaningless | Use a PDM mic (native 1-bit) |
| Invented benchmark numbers | Removed; placeholders until measured |
| Weak baseline (M0+ only) | Add STM32F446RE (Cortex-M4) baseline |
| Hidden accuracy loss | Report false-wake and miss rates |
| "Zero bus contention" proof | Dropped; not supported |
| `.pio` kernel in the draft was incorrect | Rewrite from scratch; do not debug the draft |
| DMA chain in the draft was incorrect | Redesign with proper pacing |

## 5. Experimental design

Three configurations, same audio test set, same mic:

| Config | CPU behaviour | Wake trigger |
|---|---|---|
| A. CPU-only | Active, handles every sample (PDM-to-PCM + energy VAD) | none |
| B. PIO+DMA capture, CPU wakes per buffer | Sleeps between blocks, runs VAD per block | Timer / DMA complete |
| C. PIO-Net | Sleeps until matcher fires | PIO IRQ |

- **B vs C is the real test.** B is the fair competitor. If B wins, that is a valid negative result and we report it.
- Sweep `clk_sys` for each config. PIO-Net may need a faster clock for its per-bit cycle budget, which could cancel its savings.
- Baselines run on the RP2040 M0+ and on the STM32F446RE (M4).
- Metrics: average current in quiet state, current during wake, wake latency, CPU sleep residency, false-wake rate, miss rate, total energy per hour of mixed audio (including stage-2 energy).
- Power is measured externally at the 3V3 pin, not VSYS and not via software. Check that the sensor resolves idle current (INA219 may be too coarse; consider a lab SMU or PPK2).
- Wake latency measured with a logic analyzer (trace GPIO toggled at trigger and at CPU resume).

## 6. Division of work (proposed, confirm with Gigi)

### Your job (Adarsha): firmware, hardware, measurement
- Toolchain and SWD debug setup (second Pico as debug probe).
- PIO programs: PDM reader and the matcher. Hand-written, no vibecoding.
- DMA ring buffer and pacing; memory placement.
- Clock and sleep configuration (WFI with `clk_sys` running); wake path.
- Baseline firmware: Config A and B on RP2040, and the STM32F446RE baseline.
- The whole measurement rig: external power measurement, logic analyzer timing, clock sweep.
- Sections: implementation, hardware setup, power and latency results.

### Gigi's job: training, model, detection-quality evaluation
- Dataset and test audio set, including noise conditions.
- Train the 32-bit binary matcher pattern(s). **Open requirement:** it must be trained on raw PDM data (or PDM generated from PCM), since stage 1 sees the raw bitstream.
- Stage-2 quantized classifier, sized to fit the M0+.
- The reference PDM-to-PCM + energy-VAD algorithm (agree on the exact algorithm with you so both baselines match).
- Detection-quality evaluation: false-wake and miss rates across SNR levels.
- Sections: model and training, detection-quality results.

### Shared
- Related work, literature positioning against ASICs, wake-on-sound mics and STM32 DFSDM-style peripherals.
- Abstract, introduction, discussion, limitations; final paper writing and review.
- Agree the interface early: pattern format (32-bit words, bit order), threshold semantics, test audio format.

## 7. Plan and timeline

Deadlines as you gave me: **abstract 29 Oct 2026, paper 4 or 5 Nov 2026.** I have not verified these against the SenSys 2027 call for papers. Confirm the official dates and format (page limit, anonymization) first.

Plan against **4 Nov** so the 5th is a buffer.

| Dates | Milestone |
|---|---|
| 7-8 Oct | Confirm deadlines and roles with Gigi. Order PDM mic, 2 RP2040 boards, logic analyzer, current-sense hardware. |
| 5-18 Oct (14-day learning plan, started 5 Oct) | Days 1-4: PIO fundamentals, toolchain, PDM capture, memory mapping. Days 5-9: matcher, popcount tripwire, DMA ring buffer. Days 10-14: sleep/wake path, instrumentation, failure tests. The mic arrives late for Day 3, so keep Days 1-2 (PIO study, no-mic exercises) first. |
| By 12 Oct (Gigi) | Dataset chosen, PDM conversion pipeline, first matcher pattern trained. |
| By ~14 Oct | **Go/no-go checkpoint:** does a sliding-window match fit the PIO per-bit cycle budget and 32 words? If not, fall back to non-overlapping 32-bit windows. Can raw-PDM-trained patterns discriminate at all? |
| 19-24 Oct | Config A, B, C working end to end. Power rig validated. First real measurements. |
| 25-27 Oct | Clock sweep and detection-quality runs. Draft abstract with whatever is measured. |
| **29 Oct** | **Abstract submission.** Only include measured results. |
| 30 Oct-2 Nov | Full runs, figures, tables, writing. |
| 3 Nov | Full draft read-through, fix and proofread. |
| **4 Nov** | **Paper submission** (5 Nov as buffer only). |

If the 14-day learning plan runs long, protect the measurement window (19-27 Oct). Cut scope (drop the STM32 baseline or the clock sweep granularity) before cutting measurement time.

## 8. Open risks (settle early)

1. **PIO timing budget.** At ~1 MHz PDM clock and 125 MHz `clk_sys`, there are ~120 cycles per input bit. Whether a sliding-window match fits that and 32 words is unproven.
2. **Training on raw PDM.** A fixed bit pattern matched against noise-shaped PDM is a weak detector unless trained for it. Gigi must confirm feasibility early.
3. **Idle power resolution.** Cheap current sensors may not resolve idle current. Fix the rig before relying on any number.
4. **Novelty.** Reviewers may say it is an engineering exercise. Positioning as a measured characterization, with honest negative results if B beats C, is the safest framing.
5. **Pico W radio.** If using a Pico W (RP2040), never initialize the CYW43 driver during measurements, and note in the paper that the radio module is present but idle. A Pico 2 W is an RP2350 and is not equivalent; do not use it for the paper board.
6. **Time.** About four weeks to the paper deadline, and the first two weeks are partly learning.

## 9. Bill of materials (rough, prices not verified)

| Item | Qty | Purpose |
|---|---|---|
| Raspberry Pi Pico (RP2040) or Pico W | 2 | Device under test, plus one as SWD debug probe |
| PDM MEMS mic breakout (MP34DT01-type) | 2 | Primary 1-bit input |
| INMP441 I2S mic | 1 | Optional control input |
| 24 MHz 8-channel logic analyzer | 1 | Jitter and wake latency |
| Current-sense module (INA219/INA228) or lab SMU/PPK2 | 1 | External power measurement |
| Precision shunt resistors (1 and 10 ohm, 0.1%) | few | Low-current measurement |
| Breadboard, jumpers, 33 ohm resistors, 100 nF and 10 uF caps | 1 set | Wiring and decoupling |
| Micro-USB data cables | 2 | Power and flash |
| Speaker (repeatable test stimulus) | 1 | Audio source |
| STM32F446RE Nucleo | owned | Second baseline |

## 10. Things to verify before relying on them

- SenSys 2027 deadlines and submission format.
- ATtiny85 SRAM size (believed 512 bytes).
- RP2350 GPIO-leakage erratum (only relevant if a Pico 2 / 2 W is used for measurements).
- Published power figures for wake-on-sound mics, audio ASICs and STM32 DFSDM-style options (needed for the comparison table).
- Gigi's actual scope (this split is proposed, not confirmed).
