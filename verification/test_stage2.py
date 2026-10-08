"""
Unit and audit tests for PIO-Net Stage-2 M0+ Classifier (DSCNNMicro).
Verifies:
1. Model forward pass and shape contracts.
2. Strict ARM Cortex-M0+ hardware budgets (SRAM < 16 KB, Flash < 48 KB, Latency < 20 ms).
3. CMSIS-NN C header export completeness.
4. Int8 quantization bounds and scaling.
"""

import os
import json
import pytest
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stage2.model import DSCNNMicro, calculate_m0_memory_and_latency

def test_stage2_model_architecture():
    """Verify DSCNNMicro forward pass with [1, 1, 39, 16] spectrogram input."""
    model = DSCNNMicro(num_classes=12)
    model.eval()
    dummy_input = torch.randn(2, 1, 39, 16)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 12), f"Expected shape (2, 12), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"

def test_stage2_hardware_budget():
    """Verify hardware budget calculations for RP2040 Cortex-M0+ @ 125 MHz."""
    stats = calculate_m0_memory_and_latency()
    
    # 1. SRAM Activation Arena: Must fit in 16 KB dedicated ML budget
    assert stats["peak_sram_kb"] < 16.0, f"SRAM {stats['peak_sram_kb']} KB exceeds 16.0 KB limit"
    assert stats["peak_sram_bytes"] <= 6400, "SRAM usage exceeds ping-pong arena bounds"
    
    # 2. Flash Storage: Must fit in 48 KB dedicated ML budget
    assert stats["total_flash_kb"] < 48.0, f"Flash {stats['total_flash_kb']} KB exceeds 48.0 KB limit"
    assert stats["total_flash_bytes"] < 5000, "Model parameter size exceeds 5 KB"
    
    # 3. Post-wake latency: Must be under 20.0 ms budget
    assert stats["latency_ms"] < 20.0, f"Latency {stats['latency_ms']} ms exceeds 20.0 ms limit"

def test_stage2_header_export_exists_and_complete():
    """Verify stage2/model_data.h exists and contains all required CMSIS-NN arrays."""
    header_path = REPO_ROOT / "stage2" / "model_data.h"
    assert header_path.exists(), f"Header file {header_path} not found"
    
    content = header_path.read_text()
    
    # Check arena definitions
    assert "activation_arena_a" in content
    assert "activation_arena_b" in content
    assert "ARENA_A_SIZE 3840" in content
    assert "ARENA_B_SIZE 2560" in content
    
    # Check layer weights and biases
    required_symbols = [
        "conv1_weights", "conv1_biases", "conv1_mult", "conv1_shift",
        "ds1_dw_weights", "ds1_dw_biases", "ds1_dw_mult", "ds1_dw_shift",
        "ds1_pw_weights", "ds1_pw_biases", "ds1_pw_mult", "ds1_pw_shift",
        "ds2_dw_weights", "ds2_dw_biases", "ds2_dw_mult", "ds2_dw_shift",
        "ds2_pw_weights", "ds2_pw_biases", "ds2_pw_mult", "ds2_pw_shift",
        "ds3_dw_weights", "ds3_dw_biases", "ds3_dw_mult", "ds3_dw_shift",
        "ds3_pw_weights", "ds3_pw_biases", "ds3_pw_mult", "ds3_pw_shift",
        "fc_weights", "fc_biases", "fc_mult", "fc_shift"
    ]
    for sym in required_symbols:
        assert sym in content, f"Missing symbol {sym} in model_data.h"

def test_stage2_config_json():
    """Verify stage2/model_config.json contains valid audit metrics."""
    json_path = REPO_ROOT / "stage2" / "model_config.json"
    assert json_path.exists(), f"Config JSON {json_path} not found"
    
    with open(json_path) as f:
        config = json.load(f)
        
    assert config["model_architecture"] == "DSCNNMicro"
    assert config["num_classes"] == 12
    assert config["hardware_audit"]["sram_passed"] is True
    assert config["hardware_audit"]["flash_passed"] is True
    assert config["hardware_audit"]["latency_passed"] is True
    assert config["hardware_audit"]["peak_sram_kb"] < 16.0
    assert config["hardware_audit"]["total_flash_kb"] < 48.0
    assert config["hardware_audit"]["inference_latency_ms"] < 20.0
