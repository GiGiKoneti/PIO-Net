# PIO-Net Laptop-Only Research Makefile
# Run: make all

.PHONY: all sim verify nas energy baseline paper clean

all: sim verify nas energy baseline paper

sim:
	python -m pio_sim.pio_sim
	python -m dma_sim.dma_sim
	python -m ahb_sim.ahb_sim

verify:
	.venv/bin/pytest verification/test_pionet_sim.py verification/test_phase1.py verification/test_phase2_audit.py verification/test_stage2.py verification/test_baseline_vad.py verification/test_task4_robustness.py verification/test_task4_metrics.py -v

nas:
	python -m nas.train_nas

energy:
	python -m energy_model.energy_model

baseline:
	@echo "Baseline requires ARM toolchain + CMSIS-NN; see baseline/cmsis_nn_qemu/build_and_run.sh"

paper:
	@echo "Paper generation placeholder. Write LaTeX in paper/ and run: cd paper && latexmk -pdf main.tex"

clean:
	rm -rf __pycache__ */__pycache__ *.pyc .pytest_cache