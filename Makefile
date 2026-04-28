.PHONY: help install clean clean-checkpoints clean-data clean-logs clean-research clean-all collect train-vae train-rnn train-ctrl train eval watch debug quick full quick-collect quick-vae quick-rnn quick-ctrl viz-recon viz-replay viz-latent viz-walk viz-dream viz-curves research

PYTHON = .venv/bin/python
VENV   = .venv

# ── Default ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  World Models — available commands"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install dependencies into venv"
	@echo "    make clean            Remove all generated files (checkpoints, logs, data, research)"
	@echo "    make clean-checkpoints  Remove only model checkpoints (.pt files)"
	@echo "    make clean-data       Remove only collected rollouts"
	@echo "    make clean-logs       Remove only training logs"
	@echo "    make clean-research   Remove only research/ output folder"
	@echo "    make clean-all        Remove everything including venv"
	@echo ""
	@echo "  Quick runs"
	@echo "    make quick            Collect + train VAE (~2 min), open viz"
	@echo "    make full             Full pipeline with minimal settings (~10 min), watch agent play"
	@echo "    make quick-collect    Collect 10 rollouts"
	@echo "    make quick-vae        Train VAE for 2 epochs"
	@echo "    make quick-rnn        Train MDN-RNN for 3 epochs"
	@echo "    make quick-ctrl       Train controller: 5 gens x pop 4"
	@echo "    make debug            Verify gym works: fixed steer/gas/brake, ignores controller"
	@echo ""
	@echo "  Pipeline (step by step)"
	@echo "    make collect          Collect 200 random rollouts from the environment"
	@echo "    make train-vae        Train the VAE (Vision model)"
	@echo "    make train-rnn        Train the MDN-RNN (Memory model)"
	@echo "    make train-ctrl       Train the Controller with CMA-ES"
	@echo "    make train            Run all three training steps in sequence"
	@echo ""
	@echo "  Research (paper-scale)"
	@echo "    make research         Full pipeline at Ha & Schmidhuber scale:"
	@echo "                          10k rollouts | VAE 1ep | RNN 20ep | CMA-ES 1800gen×pop64×16eval"
	@echo "                          All outputs saved to research/ (override: RESEARCH_DIR=path)"
	@echo ""
	@echo "  Evaluate"
	@echo "    make eval             Benchmark: 100 episodes headless (paper protocol)"
	@echo "    make watch            Watch agent play live in game window (3 episodes)"
	@echo ""
	@echo "  Visualize"
	@echo "    make viz-recon        VAE: real vs reconstructed frames + latent histogram"
	@echo "    make viz-replay       Scrub through a recorded rollout with slider"
	@echo "    make viz-latent       PCA of latent z vectors colored by reward"
	@echo "    make viz-walk         Slerp interpolation between two frames in latent space"
	@echo "    make viz-dream        MDN-RNN hallucinating future frames (no real env)"
	@echo "    make viz-curves       VAE + RNN training loss curves"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source .venv/bin/activate"

clean: clean-checkpoints clean-data clean-logs clean-research
	@echo "All generated files removed."

clean-checkpoints:
	rm -rf checkpoints/*
	@echo "Checkpoints removed."

clean-data:
	rm -rf data/rollouts/
	@echo "Rollout data removed."

clean-logs:
	rm -rf logs/*
	@echo "Logs removed."

clean-research:
	rm -rf $(RESEARCH_DIR)
	@echo "Research outputs removed."

clean-all: clean
	rm -rf $(VENV)
	@echo "Removed venv."

# ── Quick ─────────────────────────────────────────────────────────────────────

quick:
	$(PYTHON) main.py quick --panel vae_reconstruction

full:
	$(PYTHON) main.py quick --full

STEER ?= 0.1
GAS   ?= 0.05
BRAKE ?= 0.0

debug:
	@echo "Running with fixed action [steer=$(STEER), gas=$(GAS), brake=$(BRAKE)] to verify gym physics..."
	$(PYTHON) main.py eval --render --episodes 1 --debug-action $(STEER) $(GAS) $(BRAKE)

# ── Pipeline ──────────────────────────────────────────────────────────────────

collect:
	$(PYTHON) main.py collect

train-vae:
	$(PYTHON) main.py train-vae

train-rnn:
	$(PYTHON) main.py train-rnn

train-ctrl:
	$(PYTHON) main.py train-ctrl

quick-collect:
	$(PYTHON) main.py collect --n-rollouts 10

quick-vae:
	$(PYTHON) main.py train-vae --epochs 2

quick-rnn:
	$(PYTHON) main.py train-rnn --epochs 3

quick-ctrl:
	$(PYTHON) main.py train-ctrl --generations 5 --pop-size 4 --n-workers 1

train: train-vae train-rnn train-ctrl

# ── Research (paper-scale) ────────────────────────────────────────────────────

RESEARCH_DIR ?= research

research:
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) collect --n-rollouts 10000
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) train-vae --epochs 1
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) train-rnn --epochs 20
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) train-ctrl --generations 1800 --pop-size 64 --n-eval-episodes 16
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) eval --episodes 100
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) viz --panel vae_reconstruction --save $(RESEARCH_DIR)/viz_reconstruction.png
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) viz --panel latent_space       --save $(RESEARCH_DIR)/viz_latent.png
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) viz --panel training_curves     --save $(RESEARCH_DIR)/viz_curves.png
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) viz --panel latent_walk         --save $(RESEARCH_DIR)/viz_walk.gif
	$(PYTHON) main.py --base-dir $(RESEARCH_DIR) viz --panel rnn_dream           --save $(RESEARCH_DIR)/viz_dream.gif

# ── Evaluate ──────────────────────────────────────────────────────────────────

eval:
	$(PYTHON) main.py eval --episodes 100

watch:
	$(PYTHON) main.py eval --render --episodes 3

# ── Visualize ─────────────────────────────────────────────────────────────────

viz-recon:
	$(PYTHON) main.py viz --panel vae_reconstruction

viz-replay:
	$(PYTHON) main.py viz --panel rollout_replay

viz-latent:
	$(PYTHON) main.py viz --panel latent_space

viz-walk:
	$(PYTHON) main.py viz --panel latent_walk

viz-dream:
	$(PYTHON) main.py viz --panel rnn_dream

viz-curves:
	$(PYTHON) main.py viz --panel training_curves
