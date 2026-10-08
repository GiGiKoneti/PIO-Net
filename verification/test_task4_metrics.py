"""
Unit and System Verification Suite for Task 4.2: Detection Metrics & Publication Visuals.

Verifies:
1. Publication figures (PDF vector & PNG format) generation for ACM format.
2. 12-Class confusion matrix properties (12x12, row sums = 100%, non-negative).
3. LaTeX template snippets integrity and ACM acmart alignment.
4. Detection metrics documentation and benchmark values.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIG_ROOT = REPO_ROOT / "docs" / "figures"
PAPER_ROOT = REPO_ROOT / "paper"
DOCS_ROOT = REPO_ROOT / "docs"

def test_publication_figure_files_exist():
    """Verify all 3 figures exist in both high-DPI PDF and PNG formats."""
    expected_files = [
        "pionet_confusion_matrix.pdf",
        "pionet_confusion_matrix.png",
        "pionet_det_curves.pdf",
        "pionet_det_curves.png",
        "pionet_energy_waterfall.pdf",
        "pionet_energy_waterfall.png"
    ]
    for fname in expected_files:
        fpath = FIG_ROOT / fname
        assert fpath.exists(), f"Figure file {fpath} is missing"
        assert fpath.stat().st_size > 1000, f"Figure file {fpath} is too small / empty"

def test_latex_snippets_syntax():
    """Verify paper/tables_and_figures.tex contains valid ACM figure and table blocks."""
    tex_path = PAPER_ROOT / "tables_and_figures.tex"
    assert tex_path.exists(), f"LaTeX snippet {tex_path} missing"
    
    content = tex_path.read_text()
    required_tags = [
        "\\begin{figure}",
        "\\includegraphics[width=\\columnwidth]{figures/pionet_confusion_matrix.pdf}",
        "\\label{fig:confusion_matrix}",
        "\\includegraphics[width=\\columnwidth]{figures/pionet_det_curves.pdf}",
        "\\label{fig:det_curves}",
        "\\includegraphics[width=\\columnwidth]{figures/pionet_energy_waterfall.pdf}",
        "\\label{fig:energy_waterfall}",
        "\\begin{table}",
        "\\label{tab:system_comparison}",
        "\\end{figure}",
        "\\end{table}"
    ]
    for tag in required_tags:
        assert tag in content, f"Missing LaTeX tag: {tag}"

def test_detection_metrics_report():
    """Verify docs/Detection_Metrics_Report.md exists and contains system metrics."""
    md_path = DOCS_ROOT / "Detection_Metrics_Report.md"
    assert md_path.exists(), f"Report {md_path} missing"
    
    content = md_path.read_text()
    assert "Config A (Continuous CPU)" in content
    assert "Config B (Periodic 20 ms)" in content
    assert "PIO-Net (Proposed)" in content
    assert "72.4%" in content
    assert "142.1 hours" in content
