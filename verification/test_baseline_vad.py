"""
Unit and System Verification Suite for Task 3.2: Reference VAD Baselines.

Verifies:
1. Fixed-Point 3rd-order CIC filter decimation accuracy and passband bounds.
2. Energy + ZCR VAD detection on speech vs background noise.
3. WebRTC-style 6-subband statistical VAD likelihood calculation.
4. System execution profiler for Config A, Config B, and PIO-Net (Config C).
5. Cortex-M0+ cycle formulas and energy savings bounds.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from baseline.vad_reference import CICDecimator, EnergyZCRVAD, WebRTCSubbandVAD, VADSystemSimulator
from dataset.pcm_to_pdm import pcm_to_pdm

def test_cic_decimator_functionality():
    """Verify CIC decimation converts 1.024 MHz PDM into 16 kHz PCM within [-1.0, 1.0]."""
    decimator = CICDecimator(decimation_factor=64, order=3)
    
    # Generate 1 kHz test sine wave at 16 kHz (100 ms = 1600 samples)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    orig_pcm = 0.8 * np.sin(2 * np.pi * 1000 * t)
    
    # Modulate to 1.024 MHz PDM
    pdm_bits, _ = pcm_to_pdm(orig_pcm, orig_sr=16000, target_sr=1024000, order=2)
    
    # Decimate back to PCM
    decimated_pcm = decimator.decimate(pdm_bits, apply_fir_comp=True)
    
    assert len(decimated_pcm) == len(orig_pcm), f"Length mismatch: {len(decimated_pcm)} vs {len(orig_pcm)}"
    assert np.all(decimated_pcm >= -1.0) and np.all(decimated_pcm <= 1.0), "Decimated PCM out of bounds"
    
    # Correlation between original and decimated (ignoring filter transient boundary)
    corr = np.corrcoef(orig_pcm[100:-100], decimated_pcm[100:-100])[0, 1]
    assert corr > 0.90, f"Reconstruction correlation {corr:.4f} is too low (expected > 0.90)"

def test_energy_zcr_vad_detection():
    """Verify Energy + ZCR VAD detects active speech bursts and ignores low silence."""
    vad = EnergyZCRVAD(frame_len=320, hop_len=160)
    
    # 1. Test Silence (zero or low noise floor)
    silence = np.random.normal(0, 0.001, 1600).astype(np.float32)
    decisions_silence, _, _ = vad.process_audio(silence)
    assert np.mean(decisions_silence) == 0.0, "VAD falsely triggered on quiet background"
    
    # 2. Test Speech-like Burst (voiced sinusoid + noise)
    speech = np.sin(2 * np.pi * 300 * np.linspace(0, 0.2, 3200)).astype(np.float32) * 0.5
    decisions_speech, _, _ = vad.process_audio(speech)
    assert np.sum(decisions_speech) > 0, "VAD failed to detect active speech signal"

def test_webrtc_subband_vad():
    """Verify WebRTC-style 6-subband VAD likelihood ratio calculation."""
    vad = WebRTCSubbandVAD(sample_rate=16000, frame_len=320, aggressiveness=2)
    
    quiet_frame = np.random.normal(0, 0.001, 320).astype(np.float32)
    is_speech_quiet, llr_quiet = vad.process_frame(quiet_frame)
    assert is_speech_quiet is False, "WebRTC VAD triggered on quiet frame"
    
    active_frame = (np.sin(2 * np.pi * 800 * np.linspace(0, 0.02, 320)) * 0.4).astype(np.float32)
    is_speech_active, llr_active = vad.process_frame(active_frame)
    assert llr_active > llr_quiet, "Active frame LLR should exceed quiet frame LLR"

def test_system_configurations_simulator():
    """Verify power and duty-cycle metrics for Config A, Config B, and PIO-Net."""
    sim = VADSystemSimulator(block_duration_ms=20.0)
    
    # Config A: Continuous CPU
    res_a = sim.evaluate_configuration("CONFIG_A", "WEBRTC_VAD")
    assert res_a["active_duty_cycle_pct"] == 100.0
    assert 60.0 <= res_a["average_power_mw"] <= 70.0
    
    # Config B: Periodic Wake-Up
    res_b = sim.evaluate_configuration("CONFIG_B", "STE_ZCR")
    assert 10.0 <= res_b["active_duty_cycle_pct"] <= 18.0
    assert 24.0 <= res_b["average_power_mw"] <= 30.0
    
    # Config C: PIO-Net
    res_c = sim.evaluate_configuration("CONFIG_C_PIONET", "1BIT_PIO_MATCHER")
    assert res_c["active_duty_cycle_pct"] == 0.0
    assert res_c["active_latency_ms"] == 0.0
    assert 18.0 <= res_c["average_power_mw"] <= 24.0
    
    # PIO-Net should achieve > 60% energy savings vs Config A
    savings = res_c["energy_savings_vs_config_a_pct"]
    assert savings > 60.0, f"PIO-Net energy savings {savings:.1f}% below 60%"

def test_cortex_m0_cycle_formulas():
    """Verify Cortex-M0+ cycle count formulas."""
    pdm_samples = 20480 # 20 ms
    cic_cycles = CICDecimator.get_cortex_m0_cycles(pdm_samples)
    assert cic_cycles == (20480 * 15) + (320 * 18)
    
    pcm_samples = 320
    vad_cycles = EnergyZCRVAD.get_cortex_m0_cycles(pcm_samples)
    assert vad_cycles == (320 * 5) + (320 * 4) + 120
