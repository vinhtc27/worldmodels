"""
World Models — Main CLI
========================

Pipeline:
  1. collect   — gather random rollouts from the environment
  2. train-vae — train the Vision (VAE) model
  3. train-rnn — train the Memory (MDN-RNN) model
  4. train-ctrl— train the Controller with CMA-ES
  5. eval      — evaluate the full pipeline
  6. viz       — interactive visualizations

  all          — run the full pipeline end-to-end

Usage examples:
  python main.py collect
  python main.py train-vae --epochs 20
  python main.py train-rnn --epochs 30
  python main.py train-ctrl --generations 50
  python main.py eval --episodes 20 --render
  python main.py viz --panel vae_reconstruction
  python main.py viz --panel rnn_dream --save dream.gif
  python main.py all
"""
import argparse
import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
[bold cyan]██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗[/]
[bold cyan]██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝[/]
[bold cyan]██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗[/]
[bold cyan]██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║[/]
[bold cyan]╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║[/]
[bold cyan] ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝     ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝[/]
[dim]V (VAE) + M (MDN-RNN) + C (CMA-ES Controller)[/]
"""


def print_banner():
    console.print(BANNER)


# ── Arg parsing ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="world_models",
        description="World Models — modular PyTorch implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── collect ───────────────────────────────────────────────────────────────
    p_collect = sub.add_parser("collect", help="Collect random rollouts from the environment")
    p_collect.add_argument("--n-rollouts", type=int, default=None,
                           help="Override cfg.env.n_rollouts")
    p_collect.add_argument("--tag", default="train", help="Dataset split tag")

    # ── train-vae ─────────────────────────────────────────────────────────────
    p_vae = sub.add_parser("train-vae", help="Train the VAE (Vision Model)")
    p_vae.add_argument("--epochs",    type=int,   default=None)
    p_vae.add_argument("--lr",        type=float, default=None)
    p_vae.add_argument("--latent-dim",type=int,   default=None)
    p_vae.add_argument("--kl-weight", type=float, default=None)
    p_vae.add_argument("--resume",    action="store_true")

    # ── train-rnn ─────────────────────────────────────────────────────────────
    p_rnn = sub.add_parser("train-rnn", help="Train the MDN-RNN (Memory Model)")
    p_rnn.add_argument("--epochs",      type=int,   default=None)
    p_rnn.add_argument("--lr",          type=float, default=None)
    p_rnn.add_argument("--hidden-size", type=int,   default=None)
    p_rnn.add_argument("--n-gaussians", type=int,   default=None)
    p_rnn.add_argument("--resume",      action="store_true")

    # ── train-ctrl ────────────────────────────────────────────────────────────
    p_ctrl = sub.add_parser("train-ctrl", help="Train the Controller (CMA-ES)")
    p_ctrl.add_argument("--generations", type=int,   default=None)
    p_ctrl.add_argument("--pop-size",    type=int,   default=None)
    p_ctrl.add_argument("--n-workers",   type=int,   default=None)
    p_ctrl.add_argument("--resume",      action="store_true")

    # ── eval ──────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate the full pipeline")
    p_eval.add_argument("--episodes",     type=int, default=10)
    p_eval.add_argument("--render",       action="store_true")
    p_eval.add_argument("--seed",         type=int, default=42)
    p_eval.add_argument("--window-size",  type=int, nargs=2, metavar=("W", "H"),
                        default=None, help="Render window size e.g. --window-size 600 600")
    p_eval.add_argument("--debug-action", type=float, nargs=3, metavar=("STEER", "GAS", "BRAKE"),
                        default=None, help="Override controller with a fixed action e.g. --debug-action 1.0 0.8 0.0")

    # ── viz ───────────────────────────────────────────────────────────────────
    p_viz = sub.add_parser("viz", help="Interactive visualizations")
    p_viz.add_argument(
        "--panel",
        choices=[
            "vae_reconstruction", "latent_space", "rnn_dream",
            "training_curves",    "rollout_replay", "latent_walk",
        ],
        default="vae_reconstruction",
    )
    p_viz.add_argument("--save", default=None, help="Save output to file (PNG or GIF)")
    p_viz.add_argument("--n-samples",   type=int,   default=8)
    p_viz.add_argument("--rollout-idx", type=int,   default=0)
    p_viz.add_argument("--n-steps",     type=int,   default=200)
    p_viz.add_argument("--temperature", type=float, default=1.0)

    # ── all ───────────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="Run the full pipeline end-to-end")
    p_all.add_argument("--n-rollouts",    type=int, default=None)
    p_all.add_argument("--vae-epochs",    type=int, default=None)
    p_all.add_argument("--rnn-epochs",    type=int, default=None)
    p_all.add_argument("--ctrl-gens",     type=int, default=None)
    p_all.add_argument("--skip-collect",  action="store_true", help="Skip data collection (reuse existing rollouts)")
    p_all.add_argument("--skip-vae",      action="store_true", help="Skip VAE training (load checkpoint)")
    p_all.add_argument("--skip-rnn",      action="store_true", help="Skip RNN training (load checkpoint)")
    p_all.add_argument("--skip-ctrl",     action="store_true", help="Skip controller training (load checkpoint)")

    # ── quick ─────────────────────────────────────────────────────────────────
    p_quick = sub.add_parser(
        "quick",
        help="~2 min smoke test: tiny collect → VAE → viz",
    )
    p_quick.add_argument(
        "--panel",
        choices=[
            "vae_reconstruction", "latent_space",
            "training_curves",    "rollout_replay", "latent_walk",
        ],
        default="vae_reconstruction",
        help="Which viz panel to open after training",
    )
    p_quick.add_argument("--skip-collect", action="store_true", help="Skip data collection (reuse existing rollouts)")
    p_quick.add_argument("--skip-vae",     action="store_true", help="Skip VAE training (load checkpoint)")
    p_quick.add_argument("--skip-rnn",     action="store_true", help="Skip RNN training (load checkpoint)")
    p_quick.add_argument("--skip-ctrl",    action="store_true", help="Skip controller training (load checkpoint)")
    p_quick.add_argument("--max-steps",    type=int, default=10000, help="Steps per rollout (default 10000)")
    p_quick.add_argument("--full",         action="store_true",
                         help="Run ALL steps (collect→VAE→RNN→Controller) then watch the agent play live (~5-10 min)")

    return parser


# ── Apply CLI overrides to config ─────────────────────────────────────────────

def apply_overrides(cfg, args):
    cmd = args.command
    if cmd == "train-vae":
        if args.epochs:      cfg.vae.epochs    = args.epochs
        if args.lr:          cfg.vae.lr        = args.lr
        if args.latent_dim:  cfg.vae.latent_dim = args.latent_dim
        if args.kl_weight:   cfg.vae.kl_weight = args.kl_weight
    elif cmd == "train-rnn":
        if args.epochs:       cfg.rnn.epochs      = args.epochs
        if args.lr:           cfg.rnn.lr          = args.lr
        if args.hidden_size:  cfg.rnn.hidden_size = args.hidden_size
        if args.n_gaussians:  cfg.rnn.n_gaussians = args.n_gaussians
    elif cmd == "train-ctrl":
        if args.generations: cfg.controller.n_generations = args.generations
        if args.pop_size:    cfg.controller.pop_size       = args.pop_size
        if args.n_workers:   cfg.controller.n_workers      = args.n_workers
    elif cmd == "all":
        if args.n_rollouts:  cfg.env.n_rollouts            = args.n_rollouts
        if args.vae_epochs:  cfg.vae.epochs                = args.vae_epochs
        if args.rnn_epochs:  cfg.rnn.epochs                = args.rnn_epochs
        if args.ctrl_gens:   cfg.controller.n_generations  = args.ctrl_gens


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_collect(args, cfg):
    from data import collect_rollouts
    n = args.n_rollouts or cfg.env.n_rollouts
    console.print(Panel(f"Collecting [cyan]{n}[/] rollouts  (env: [bold]{cfg.env.name}[/])"))
    collect_rollouts(cfg, n_rollouts=n, tag=args.tag)


def cmd_train_vae(args, cfg):
    from training import train_vae
    console.print(Panel(
        f"VAE — latent_dim=[cyan]{cfg.vae.latent_dim}[/]  "
        f"epochs=[cyan]{cfg.vae.epochs}[/]  lr=[cyan]{cfg.vae.lr}[/]"
    ))
    train_vae(cfg, resume=getattr(args, "resume", False))


def cmd_train_rnn(args, cfg):
    from training import train_rnn
    console.print(Panel(
        f"MDN-RNN — hidden=[cyan]{cfg.rnn.hidden_size}[/]  "
        f"gaussians=[cyan]{cfg.rnn.n_gaussians}[/]  "
        f"epochs=[cyan]{cfg.rnn.epochs}[/]"
    ))
    train_rnn(cfg, resume=getattr(args, "resume", False))


def cmd_train_ctrl(args, cfg):
    from training import train_controller
    console.print(Panel(
        f"Controller (CMA-ES) — pop=[cyan]{cfg.controller.pop_size}[/]  "
        f"generations=[cyan]{cfg.controller.n_generations}[/]  "
        f"workers=[cyan]{cfg.controller.n_workers}[/]"
    ))
    train_controller(cfg, resume=getattr(args, "resume", False))


def cmd_eval(args, cfg):
    from evaluation import evaluate
    if args.window_size:
        cfg.env.window_width, cfg.env.window_height = args.window_size
    console.print(Panel(f"Evaluating — [cyan]{args.episodes}[/] episodes"))
    evaluate(cfg, n_episodes=args.episodes, render=args.render, seed=args.seed,
             debug_action=args.debug_action)


def cmd_viz(args, cfg):
    from visualization import (
        vae_reconstruction, latent_space_pca, rnn_dream,
        training_curves, rollout_replay, latent_walk,
    )
    panel = args.panel
    console.print(Panel(f"Visualization — [cyan]{panel}[/]"))

    if panel == "vae_reconstruction":
        vae_reconstruction(cfg, n_samples=args.n_samples, save_path=args.save)
    elif panel == "latent_space":
        latent_space_pca(cfg, save_path=args.save)
    elif panel == "rnn_dream":
        rnn_dream(cfg, n_steps=args.n_steps, temperature=args.temperature, save_gif=args.save)
    elif panel == "training_curves":
        training_curves(cfg, save_path=args.save)
    elif panel == "rollout_replay":
        rollout_replay(cfg, rollout_idx=args.rollout_idx, save_gif=args.save)
    elif panel == "latent_walk":
        latent_walk(cfg, n_steps=args.n_steps, save_gif=args.save)


def _skip_msg(step: str):
    console.print(f"  [yellow]⏭  Skipping {step} (using existing checkpoint/data)[/]")


def cmd_quick(args, cfg):
    from data import collect_rollouts
    from training import train_vae, train_rnn, train_controller
    from evaluation import evaluate
    from visualization import (
        vae_reconstruction, latent_space_pca,
        training_curves, rollout_replay, latent_walk,
    )

    if args.full:
        # ── Full quick pipeline: all steps with tiny settings, ends with live play ──
        cfg.env.n_rollouts          = 10
        cfg.env.max_steps           = args.max_steps
        cfg.vae.epochs              = 2
        cfg.vae.batch_size          = 32
        cfg.rnn.epochs              = 3
        cfg.rnn.batch_size          = 16
        cfg.controller.pop_size     = 4
        cfg.controller.n_generations = 5
        cfg.controller.n_eval_episodes = 1
        cfg.controller.n_workers    = 1   # avoid multiprocessing overhead for tiny pop

        console.print(Panel(
            "[bold green]Quick FULL pipeline[/]\n"
            "10 rollouts  |  VAE 2 epochs  |  RNN 3 epochs  |  CMA-ES 5 gens × pop 4\n"
            "[dim]The agent won't drive well — but you'll see it play live at the end[/]",
            title="~5-10 min end-to-end",
        ))

        console.rule("[bold cyan]1/5  Collect")
        if args.skip_collect:
            _skip_msg("data collection")
        else:
            collect_rollouts(cfg, tag="train")

        console.rule("[bold cyan]2/5  Train VAE")
        if args.skip_vae:
            _skip_msg("VAE training")
            from models import VAE
            from utils import load_checkpoint
            vae = VAE(cfg.vae)
            vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, cfg.get_device())["model"])
        else:
            vae = train_vae(cfg)

        console.rule("[bold cyan]3/5  Train MDN-RNN")
        if args.skip_rnn:
            _skip_msg("RNN training")
            from models import MDNRNN
            from utils import load_checkpoint
            rnn = MDNRNN(cfg.rnn)
            rnn.load_state_dict(load_checkpoint(cfg.paths.rnn_checkpoint, cfg.get_device())["model"])
        else:
            rnn = train_rnn(cfg)

        console.rule("[bold cyan]4/5  Train Controller")
        if args.skip_ctrl:
            _skip_msg("controller training")
        else:
            train_controller(cfg, vae=vae, rnn=rnn)

        console.rule("[bold cyan]5/5  Watch agent play live")
        evaluate(cfg, n_episodes=2, render=True)

    else:
        # ── VAE-only quick mode: collect → VAE → viz panel ────────────────────
        cfg.env.n_rollouts  = 15
        cfg.env.max_steps   = args.max_steps
        cfg.vae.epochs      = 2
        cfg.vae.batch_size  = 32

        console.print(Panel(
            "[bold green]Quick VAE demo[/]\n"
            f"15 rollouts × {args.max_steps} steps  |  VAE 2 epochs\n"
            "[dim]Use --full to run all steps and watch the agent play[/]",
            title="~2 min smoke test",
        ))

        console.rule("[bold cyan]1/3  Collect")
        if args.skip_collect:
            _skip_msg("data collection")
        else:
            collect_rollouts(cfg, tag="train")

        console.rule("[bold cyan]2/3  Train VAE")
        if args.skip_vae:
            _skip_msg("VAE training")
        else:
            train_vae(cfg)

        console.rule("[bold cyan]3/3  Visualize")
        panel = args.panel
        console.print(f"Opening [cyan]{panel}[/] …")
        if panel == "vae_reconstruction":
            vae_reconstruction(cfg, n_samples=6)
        elif panel == "latent_space":
            latent_space_pca(cfg)
        elif panel == "training_curves":
            training_curves(cfg)
        elif panel == "rollout_replay":
            rollout_replay(cfg, rollout_idx=0)
        elif panel == "latent_walk":
            latent_walk(cfg, n_steps=40)


def cmd_all(args, cfg):
    from data import collect_rollouts
    from training import train_vae, train_rnn, train_controller
    from evaluation import evaluate

    console.print(Panel("[bold green]Full pipeline: collect → VAE → RNN → Controller → Eval"))

    console.rule("[bold cyan]Step 1/5: Collect Rollouts")
    if args.skip_collect:
        _skip_msg("data collection")
    else:
        collect_rollouts(cfg, tag="train")

    console.rule("[bold cyan]Step 2/5: Train VAE")
    if args.skip_vae:
        _skip_msg("VAE training")
        from models import VAE
        from utils import load_checkpoint
        vae = VAE(cfg.vae)
        vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, cfg.get_device())["model"])
    else:
        vae = train_vae(cfg)

    console.rule("[bold cyan]Step 3/5: Train MDN-RNN")
    if args.skip_rnn:
        _skip_msg("RNN training")
        from models import MDNRNN
        from utils import load_checkpoint
        rnn = MDNRNN(cfg.rnn)
        rnn.load_state_dict(load_checkpoint(cfg.paths.rnn_checkpoint, cfg.get_device())["model"])
    else:
        rnn = train_rnn(cfg)

    console.rule("[bold cyan]Step 4/5: Train Controller")
    if args.skip_ctrl:
        _skip_msg("controller training")
    else:
        train_controller(cfg, vae=vae, rnn=rnn)

    console.rule("[bold cyan]Step 5/5: Evaluate")
    evaluate(cfg, n_episodes=5)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print_banner()
    parser = build_parser()
    args = parser.parse_args()

    from config import cfg
    apply_overrides(cfg, args)

    console.print(f"  Device: [cyan]{cfg.get_device()}[/]   Env: [cyan]{cfg.env.name}[/]   Seed: [cyan]{cfg.seed}[/]\n")

    dispatch = {
        "collect":    cmd_collect,
        "train-vae":  cmd_train_vae,
        "train-rnn":  cmd_train_rnn,
        "train-ctrl": cmd_train_ctrl,
        "eval":       cmd_eval,
        "viz":        cmd_viz,
        "all":        cmd_all,
        "quick":      cmd_quick,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
