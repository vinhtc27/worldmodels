.PHONY: help install clean clean-all collect train-vae train-rnn train-ctrl train eval watch debug quick full viz-recon viz-replay viz-latent viz-walk viz-dream viz-curves

PYTHON = venv/bin/python
VENV   = venv

# ── Default ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  World Models — available commands"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install dependencies into venv"
	@echo "    make clean            Remove checkpoints, logs, and collected data"
	@echo "    make clean-all        Remove everything including venv"
	@echo ""
	@echo "  Quick runs"
	@echo "    make quick            Collect + train VAE (~2 min), open viz"
	@echo "    make full             Full pipeline with minimal settings (~10 min), watch agent play"
	@echo "    make debug            Verify gym works: fixed steer/gas/brake, ignores controller"
	@echo ""
	@echo "  Pipeline (step by step)"
	@echo "    make collect          Collect 200 random rollouts from the environment"
	@echo "    make train-vae        Train the VAE (Vision model)"
	@echo "    make train-rnn        Train the MDN-RNN (Memory model)"
	@echo "    make train-ctrl       Train the Controller with CMA-ES"
	@echo "    make train            Run all three training steps in sequence"
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
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source venv/bin/activate"

clean:
	rm -rf checkpoints/* logs/* data/rollouts/
	@echo "Cleared checkpoints, logs, and rollouts."

clean-all: clean
	rm -rf $(VENV)
	@echo "Removed venv."

# ── Quick ─────────────────────────────────────────────────────────────────────

quick:
	$(PYTHON) main.py quick --panel vae_reconstruction

full:
	$(PYTHON) main.py quick --full

debug:
	@echo "Running with fixed action [steer=+0.5, gas=0.1, brake=0.1] to verify gym physics..."
	$(PYTHON) main.py eval --render --episodes 1 --debug 0.5 0.5 0.1

# ── Pipeline ──────────────────────────────────────────────────────────────────

collect:
	$(PYTHON) main.py collect

train-vae:
	$(PYTHON) main.py train-vae

train-rnn:
	$(PYTHON) main.py train-rnn

train-ctrl:
	$(PYTHON) main.py train-ctrl

train: train-vae train-rnn train-ctrl

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
