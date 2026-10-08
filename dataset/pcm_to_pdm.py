"""
PIO-Net Task 1.2: PCM-to-1-Bit PDM Audio Converter

Implements:
1. High-fidelity upsampling from 16 kHz to 1.024 MHz (OSR = 64).
2. 1st-order and 2nd-order Sigma-Delta (ΣΔ) Modulation.
3. Bit-packing into uint32 words matching RP2040 PIO ISR/FIFO words.
4. Validation & reconstruction (PDM -> PCM CIC decimation) for fidelity check.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

def pcm_to_pdm(audio_pcm: np.ndarray, orig_sr: int = 16000, target_sr: int = 1024000, order: int = 2):
    """
    Converts 16 kHz float PCM audio into a 1-bit PDM bitstream at target_sr (1.024 MHz).
    
    Args:
        audio_pcm: 1D numpy array of audio samples, normalized to [-1.0, 1.0].
        orig_sr: Input sample rate (default 16000 Hz).
        target_sr: PDM clock rate (default 1024000 Hz, OSR = 64).
        order: Sigma-Delta modulator order (1 or 2).
        
    Returns:
        pdm_bits: 1D numpy array of uint8 (0 or 1) at target_sr.
        pdm_words: 1D numpy array of uint32 words (each containing 32 consecutive PDM bits).
    """
    osr = target_sr // orig_sr
    assert target_sr % orig_sr == 0, f"Target rate {target_sr} must be integer multiple of {orig_sr}"
    
    # Scale slightly below 1.0 to avoid sigma-delta integrator saturation
    pcm_norm = np.clip(audio_pcm, -1.0, 1.0) * 0.90
    
    # 1. High-fidelity interpolation to target clock rate
    # resample_poly upsamples by osr using a polyphase FIR filter
    upsampled = signal.resample_poly(pcm_norm, up=osr, down=1)
    
    # 2. Sigma-Delta Modulation loop
    n_samples = len(upsampled)
    pdm_bits = np.empty(n_samples, dtype=np.uint8)
    
    if order == 1:
        integrator = 0.0
        y = 1.0
        for i in range(n_samples):
            # Error = input - DAC feedback
            integrator += upsampled[i] - y
            if integrator >= 0.0:
                y = 1.0
                pdm_bits[i] = 1
            else:
                y = -1.0
                pdm_bits[i] = 0
    elif order == 2:
        # Standard 2nd-order Sigma-Delta topology
        int1 = 0.0
        int2 = 0.0
        y = 1.0
        for i in range(n_samples):
            int1 += upsampled[i] - y
            int2 += int1 - 2.0 * y
            if int2 >= 0.0:
                y = 1.0
                pdm_bits[i] = 1
            else:
                y = -1.0
                pdm_bits[i] = 0
    else:
        raise ValueError(f"Unsupported sigma-delta order: {order}. Choose 1 or 2.")
        
    # 3. Pack bits into 32-bit words (matching RP2040 PIO ISR autopush)
    # Truncate to multiple of 32
    num_words = n_samples // 32
    bits_aligned = pdm_bits[:num_words * 32].reshape(num_words, 32)
    
    # Shift bits into uint32 (bit 0 = first received bit, matching PIO shift direction)
    powers_of_two = (1 << np.arange(32, dtype=np.uint32))
    pdm_words = np.dot(bits_aligned, powers_of_two).astype(np.uint32)
    
    return pdm_bits, pdm_words

def pdm_to_pcm_decimate(pdm_bits: np.ndarray, decimation_factor: int = 64):
    """
    Decimates 1-bit PDM stream back to PCM using a multi-stage low-pass boxcar filter.
    Used for verifying that the PDM conversion preserved audio content.
    """
    # Map bits: 0 -> -1.0, 1 -> +1.0
    bipolar = pdm_bits.astype(np.float32) * 2.0 - 1.0
    
    # Cascaded Integrator-Comb (CIC) equivalent: boxcar moving sum
    window = np.ones(decimation_factor, dtype=np.float32) / decimation_factor
    filtered = np.convolve(bipolar, window, mode='valid')
    
    # Downsample by decimation factor
    downsampled = filtered[::decimation_factor]
    
    # Normalize
    max_val = np.max(np.abs(downsampled)) + 1e-8
    return downsampled / max_val

def convert_wav_file(wav_path: Path, output_dir: Path, order: int = 2):
    """
    Converts a single WAV file to .pdm (packed binary) and .npy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sr, data = wavfile.read(wav_path)
    
    if data.dtype == np.int16:
        pcm = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        pcm = data.astype(np.float32) / 2147483648.0
    else:
        pcm = data.astype(np.float32)
        
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1) # Mono
        
    pdm_bits, pdm_words = pcm_to_pdm(pcm, orig_sr=sr, target_sr=1024000, order=order)
    
    out_stem = wav_path.stem
    words_path = output_dir / f"{out_stem}_pdm32.npy"
    np.save(words_path, pdm_words)
    
    # Also write raw binary format
    bin_path = output_dir / f"{out_stem}.pdm"
    pdm_words.tofile(bin_path)
    
    return words_path, bin_path, pdm_bits, pcm

def verify_conversion_fidelity(orig_pcm: np.ndarray, pdm_bits: np.ndarray):
    """
    Verifies that the PDM stream faithfully preserves the audio by computing correlation.
    """
    reconstructed = pdm_to_pcm_decimate(pdm_bits, decimation_factor=64)
    min_len = min(len(orig_pcm), len(reconstructed))
    sig1 = orig_pcm[:min_len]
    sig2 = reconstructed[:min_len]
    
    # Cross-correlation at zero lag
    norm1 = np.linalg.norm(sig1) + 1e-8
    norm2 = np.linalg.norm(sig2) + 1e-8
    correlation = np.dot(sig1, sig2) / (norm1 * norm2)
    return correlation

def main():
    parser = argparse.ArgumentParser(description="PIO-Net PCM to 1-Bit PDM Converter")
    parser.add_argument("--test-sample", action="store_true", default=False,
                        help="Run validation on a sample audio file.")
    parser.add_argument("--batch-samples", type=int, default=10,
                        help="Convert N representative samples from each class.")
    args = parser.parse_args()
    
    data_root = Path("/home/gigikoneti/SenSys2027/data")
    speech_dir = data_root / "clean_speech"
    output_pdm_dir = data_root / "pdm_streams"
    
    sample_wav = None
    for target in ["yes", "stop", "go"]:
        target_dir = speech_dir / target
        if target_dir.exists():
            wavs = list(target_dir.glob("*.wav"))
            if wavs:
                sample_wav = wavs[0]
                break
                
    if sample_wav is None:
        print("❌ No WAV files found in data/clean_speech!")
        sys.exit(1)
        
    print(f"🎵 Testing conversion on sample: {sample_wav.name} (class: {sample_wav.parent.name})")
    words_path, bin_path, pdm_bits, orig_pcm = convert_wav_file(sample_wav, output_pdm_dir / "samples")
    
    corr = verify_conversion_fidelity(orig_pcm, pdm_bits)
    print(f"✅ Conversion complete:")
    print(f"   - PDM Bits Count: {len(pdm_bits):,} bits @ 1.024 MHz")
    print(f"   - 32-bit Words:   {len(np.load(words_path)):,} words (packed for RP2040 PIO)")
    print(f"   - Audio Fidelity (Correlation): {corr:.4f} (Expected > 0.90)")
    
    if args.batch_samples > 0:
        print(f"\n📦 Batch converting {args.batch_samples} samples per class to PDM...")
        classes = ["yes", "no", "stop", "go", "_silence_"]
        for cls in classes:
            c_dir = speech_dir / cls
            if not c_dir.exists():
                continue
            wav_list = list(c_dir.glob("*.wav"))[:args.batch_samples]
            out_cls = output_pdm_dir / cls
            for w in wav_list:
                convert_wav_file(w, out_cls)
            print(f"   - Converted {len(wav_list)} clips for '{cls}' -> {out_cls}")
            
    print(f"\n🎉 Task 1.2 successfully executed! PDM streams saved in: {output_pdm_dir}")

if __name__ == "__main__":
    main()
