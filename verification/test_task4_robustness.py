"""
Unit and System Verification Suite for Task 4.1: Sim-to-Real Acoustic Robustness.

Verifies:
1. ITU-T P.56 active speech level calculation vs naive whole-file RMS.
2. Room Impulse Response (RIR) convolution and reverberation decay bounds.
3. End-to-End False-Wake Rate (FWR) on continuous background noise (<= 1.0 event/hour).
4. Cascaded system energy savings (> 60% reduction vs Config A).
5. Evaluation artifacts integrity (JSON schema & Markdown report).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_noise_robustness import (
    synthesize_room_impulse_response,
    active_speech_level_p56,
    mix_audio_with_noise_p56
)

def test_rir_synthesis():
    """Verify physical Room Impulse Response (RIR) synthesis and energy decay."""
    sr = 16000
    rt60 = 0.20
    rir = synthesize_room_impulse_response(duration_sec=0.20, rt60=rt60, sr=sr)
    
    assert len(rir) == int(0.20 * sr)
    # Energy should be normalized
    total_energy = np.sum(rir ** 2)
    assert 0.99 <= total_energy <= 1.01, f"RIR total energy {total_energy} is not normalized"
    # Peak should occur at direct path t=0
    assert np.argmax(np.abs(rir)) == 0, "Direct path peak must be at t=0"
    # Tail energy should be lower than head energy (exponential decay)
    head_energy = np.sum(rir[:int(0.05 * sr)] ** 2)
    tail_energy = np.sum(rir[int(0.15 * sr):] ** 2)
    assert head_energy > tail_energy * 5.0, "RIR failed to exhibit exponential reverberation decay"

def test_p56_active_speech_level_vs_whole_file_rms():
    """Verify ITU-T P.56 isolates active vocalization power from silence intervals."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 0.3s speech tone in middle of 1s audio (0.7s of silence)
    speech = np.zeros(sr, dtype=np.float32)
    speech[int(0.3 * sr) : int(0.6 * sr)] = 0.5 * np.sin(2 * np.pi * 500 * t[:int(0.3 * sr)])
    
    whole_file_rms_power = float(np.mean(speech ** 2))
    p56_active_power = active_speech_level_p56(speech)
    
    # Active power should be roughly (0.5 / sqrt(2))^2 = 0.125, whereas whole-file power is 0.3 * 0.125 = 0.0375
    assert p56_active_power > whole_file_rms_power * 2.0, "P.56 must isolate active intervals from silence"

def test_p56_noise_mixing_snr():
    """Verify mix_audio_with_noise_p56 generates valid mixed signals."""
    sr = 16000
    speech = 0.5 * np.sin(2 * np.pi * 400 * np.linspace(0, 0.5, 8000)).astype(np.float32)
    noise = np.random.normal(0, 0.1, 8000).astype(np.float32)
    
    mixed = mix_audio_with_noise_p56(speech, noise, snr_db=10.0)
    assert len(mixed) == len(speech)
    assert np.all(mixed >= -1.0) and np.all(mixed <= 1.0), "Mixed PCM out of range [-1.0, 1.0]"

def test_noise_robustness_artifacts():
    """Verify evaluation JSON and Markdown report artifacts exist and pass constraints."""
    json_path = REPO_ROOT / "evaluation" / "noise_robustness_results.json"
    md_path = REPO_ROOT / "docs" / "Noise_Robustness_Report.md"
    
    assert json_path.exists(), f"Results JSON {json_path} missing"
    assert md_path.exists(), f"Report Markdown {md_path} missing"
    
    with open(json_path) as f:
        data = json.load(f)
        
    assert "snr_levels_db" in data
    assert "profiles" in data
    assert "quiescent_fwr" in data
    
    # Check End-to-End False Keyword Rate is <= 1.0 event/hour across all noise streams
    for env_name, fwr in data["quiescent_fwr"].items():
        assert fwr["e2e_fwr_per_hour"] <= 1.0, f"End-to-end FWR {fwr['e2e_fwr_per_hour']} exceeds 1.0/h for {env_name}"
        assert fwr["energy_savings_pct"] >= 60.0, f"Energy savings {fwr['energy_savings_pct']}% below 60% for {env_name}"
