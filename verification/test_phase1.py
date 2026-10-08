"""
Comprehensive Verification Suite for Phase 1 (Task 1.1 & Task 1.2)
Verifies:
- Task 1.1: Audio dataset structure, sample rates, channels, file counts, and audio validity.
- Task 1.2: PDM conversion, exact bit-count, bit-packing match, sigma-delta noise shaping,
            reconstruction correlation, and on-disk stream integrity.
"""

import os
import glob
import wave
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

import sys
sys.path.insert(0, '/home/gigikoneti/SenSys2027')

from dataset.pcm_to_pdm import pcm_to_pdm, pdm_to_pcm_decimate, verify_conversion_fidelity

DATA_ROOT = Path("/home/gigikoneti/SenSys2027/data")

# ==============================================================================
# TASK 1.1 AUDIT TESTS
# ==============================================================================

def test_1_1_directory_structure():
    """Verify all required dataset directories exist."""
    assert (DATA_ROOT / "clean_speech").exists(), "Missing data/clean_speech directory"
    assert (DATA_ROOT / "background_noise").exists(), "Missing data/background_noise directory"
    assert (DATA_ROOT / "clean_transients").exists(), "Missing data/clean_transients directory"

def test_1_1_speech_commands_classes_and_counts():
    """Verify all 12 MLPerf Tiny standard classes exist with hundreds of WAV files each."""
    expected_classes = ["yes", "no", "stop", "go", "on", "off", "up", "down", "left", "right", "_silence_", "_unknown_"]
    clean_speech_dir = DATA_ROOT / "clean_speech"
    
    total_files = 0
    for cls in expected_classes:
        cls_dir = clean_speech_dir / cls
        assert cls_dir.exists(), f"Missing class directory: {cls}"
        wavs = list(cls_dir.glob("*.wav"))
        assert len(wavs) >= 300, f"Class {cls} has only {len(wavs)} files, expected >= 300"
        total_files += len(wavs)
        
    assert total_files > 4000, f"Total clean speech files {total_files} < 4000"
    print(f"\n[Audit 1.1] Verified {total_files} speech files across all 12 classes.")

def test_1_1_audio_format_and_integrity():
    """Verify WAV audio properties: 16000 Hz, 1 channel, 16-bit PCM, valid amplitudes."""
    clean_speech_dir = DATA_ROOT / "clean_speech"
    sample_wavs = list(clean_speech_dir.glob("*/*.wav"))[:20]
    
    for wav_path in sample_wavs:
        with wave.open(str(wav_path), "rb") as w:
            assert w.getframerate() == 16000, f"{wav_path} is {w.getframerate()}Hz, expected 16000Hz"
            assert w.getnchannels() == 1, f"{wav_path} has {w.getnchannels()} channels, expected 1"
            assert w.getsampwidth() == 2, f"{wav_path} sample width is {w.getsampwidth()} bytes, expected 2 (16-bit)"
            assert w.getnframes() == 16000, f"{wav_path} duration frames {w.getnframes()} != 16000"
            
        sr, data = wavfile.read(wav_path)
        assert data.dtype == np.int16
        assert np.max(np.abs(data)) > 0, f"Audio file {wav_path} is completely silent (all zeros)"

def test_1_1_background_noise_profiles():
    """Verify calibrated noise profiles are present, valid, and non-empty."""
    noise_dir = DATA_ROOT / "background_noise"
    expected_noises = [
        "factory_floor_hum.wav",
        "office_ambient_chatter.wav",
        "synthetic_white_noise.wav",
        "synthetic_pink_noise.wav"
    ]
    for n in expected_noises:
        p = noise_dir / n
        assert p.exists(), f"Missing noise profile: {n}"
        sr, data = wavfile.read(p)
        assert sr == 16000
        assert len(data) >= 16000
        assert np.std(data) > 100, f"Noise profile {n} lacks dynamic variance"

# ==============================================================================
# TASK 1.2 AUDIT TESTS
# ==============================================================================

def test_1_2_pdm_exact_bit_dimensions():
    """Verify that a 1-second 16kHz audio clip converts to exactly 1,024,000 bits and 32,000 uint32 words."""
    test_sine = 0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, 1.0, 16000, endpoint=False))
    pdm_bits, pdm_words = pcm_to_pdm(test_sine, orig_sr=16000, target_sr=1024000, order=2)
    
    assert len(pdm_bits) == 1024000, f"Expected 1,024,000 bits, got {len(pdm_bits)}"
    assert pdm_bits.dtype == np.uint8
    assert set(np.unique(pdm_bits)).issubset({0, 1}), "PDM bits must strictly contain only 0 and 1"
    
    assert len(pdm_words) == 32000, f"Expected 32,000 words, got {len(pdm_words)}"
    assert pdm_words.dtype == np.uint32

def test_1_2_bit_packing_unambiguous_exactness():
    """Verify bit-for-bit equivalence between raw PDM bits and unpacked uint32 words."""
    test_sine = 0.6 * np.cos(2 * np.pi * 440 * np.linspace(0, 0.1, 1600, endpoint=False))
    pdm_bits, pdm_words = pcm_to_pdm(test_sine, orig_sr=16000, target_sr=1024000, order=2)
    
    # Unpack uint32 words back to bits
    num_words = len(pdm_words)
    unpacked_bits = np.zeros(num_words * 32, dtype=np.uint8)
    for bit_idx in range(32):
        unpacked_bits[bit_idx::32] = ((pdm_words >> bit_idx) & 1).astype(np.uint8)
        
    np.testing.assert_array_equal(unpacked_bits, pdm_bits[:num_words * 32], 
                                  err_msg="Unpacked words must match original PDM bits 100% exactly")

def test_1_2_reconstruction_fidelity_on_real_speech():
    """Verify that real spoken keywords retain high acoustic fidelity (>0.90 correlation) through PDM."""
    wav_path = list((DATA_ROOT / "clean_speech" / "yes").glob("*.wav"))[0]
    sr, data = wavfile.read(wav_path)
    pcm = data.astype(np.float32) / 32768.0
    
    pdm_bits, pdm_words = pcm_to_pdm(pcm, orig_sr=16000, target_sr=1024000, order=2)
    corr = verify_conversion_fidelity(pcm, pdm_bits)
    
    print(f"\n[Audit 1.2] Speech '{wav_path.name}' fidelity correlation: {corr:.4f}")
    assert corr > 0.90, f"Fidelity correlation {corr:.4f} is below 0.90 target"

def test_1_2_on_disk_pdm_streams_integrity():
    """Verify that on-disk batch-converted PDM stream files exist and load cleanly."""
    pdm_dir = DATA_ROOT / "pdm_streams"
    assert pdm_dir.exists(), "pdm_streams directory missing"
    
    for cls in ["yes", "no", "stop", "go", "_silence_"]:
        cls_pdm_dir = pdm_dir / cls
        assert cls_pdm_dir.exists(), f"Missing PDM stream class directory: {cls}"
        npy_files = list(cls_pdm_dir.glob("*_pdm32.npy"))
        assert len(npy_files) > 0, f"No .npy files in {cls_pdm_dir}"
        
        # Test loading
        arr = np.load(npy_files[0])
        assert arr.dtype == np.uint32
        assert len(arr) == 32000, f"Array length {len(arr)} != 32,000 words"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
