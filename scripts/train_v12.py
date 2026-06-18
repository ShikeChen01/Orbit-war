"""Entry point for the packaged v12 training stack (orbit_wars_v12).

Mirrors the notebook run/plot/save tail: optional BC warm-start -> Trainer.train -> plot ->
save the league agents. Honors OW_CKPT_DIR / OW_LEAGUE_DIR like the notebook.

    python scripts/train_v12.py                 # full run (SMOKE=False)
    python scripts/train_v12.py --smoke          # fast end-to-end sanity run
    python scripts/train_v12.py --iters 50       # override TOTAL_ITERS
    python scripts/train_v12.py --resume runs/.../setup1_target_train_state.pt
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from orbit_wars_v12 import Config, Trainer, bc_pretrain, plot_training_health, save_league_agents


def main():
    ap = argparse.ArgumentParser(description="Train the Orbit Wars v12 policy.")
    ap.add_argument("--smoke", action="store_true", help="fast shrunk end-to-end sanity run")
    ap.add_argument("--iters", type=int, default=None, help="override TOTAL_ITERS")
    ap.add_argument("--resume", type=str, default=None, help="resume from a train-state .pt")
    ap.add_argument("--no-plot", action="store_true", help="skip the training-health figure")
    ap.add_argument("--no-save-league", action="store_true", help="skip dumping league agents")
    args = ap.parse_args()

    if args.no_plot:
        os.environ.setdefault("MPLBACKEND", "Agg")

    cfg = Config.create(smoke=args.smoke)
    resume_from = args.resume if args.resume is not None else cfg.RESUME_FROM

    # BC warm-start (the from-scratch bootstrap is the proven passivity trap): clone the medium bot
    # then warm-start PPO from it, exactly like the notebook BC cell.
    if cfg.BC_ENABLED and not resume_from:
        bc_pretrain(cfg)
        resume_from = cfg.BC_CKPT_PATH

    net, hist, league = Trainer(cfg).train(total_iters=args.iters, log_every=1, resume_from=resume_from)

    if not args.no_save_league:
        save_league_agents(cfg, league)
    if not args.no_plot:
        fig_path = os.path.join(cfg.CKPT_DIR, "training_health.png")
        try:
            plot_training_health(hist, show=False, save_path=fig_path)
            print("saved training-health figure ->", fig_path)
        except Exception as e:
            print("  !! plot failed -> %r" % e)
    return net, hist, league


if __name__ == "__main__":
    main()
