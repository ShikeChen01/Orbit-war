"""Batch-render a TARGET-actor checkpoint playing N games against each scripted agent, into one
folder named after the checkpoint. Reuses scripts/viz_target_ckpt.py (load defs -> build actor from
the ckpt's OWN config -> greedy self-play -> GIF + keyframe PNG), but pointed at a chosen defs
notebook so OLD-generation checkpoints (e.g. F_DIM=32, discrete dest+phi) render under the matching
code rather than the current working-tree notebook.

    python scripts/render_ckpt_batch.py \
        --ckpt notebooks/1400iter-20hidden-MLPs/weights/256-32-MLPs-E610-Iter950.pt \
        --defs-nb notebooks/render/v5_legacy_defs.ipynb \
        --agents random,starter,noop --seeds 0,1,2

Outputs (default): runs/viz/<ckpt-stem>/<stem>_vs_<agent>_s<seed>.{json,gif,_keyframes.png}
plus results.md (per-agent win/loss/draw + final-ship margins).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import viz_target_ckpt as V   # noqa: E402  (tested load/build/play helpers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--defs-nb", default="notebooks/render/v5_legacy_defs.ipynb",
                    help="notebook whose defs match the checkpoint generation")
    ap.add_argument("--agents", default="random,starter,noop")
    ap.add_argument("--seeds", default="0,1,2", help="comma-sep seeds = plays per agent")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    os.chdir(REPO)

    V.NB = args.defs_nb                       # point the loader at the matching-generation defs
    g = V.load_notebook_defs()
    net0, cfg, blob = V.build_actor(g, args.ckpt)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    outdir = args.outdir or os.path.join("runs", "viz", name)
    os.makedirs(outdir, exist_ok=True)
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    print("ckpt %s | iter=%s win_rate=%s | HIDDEN=%s N_RES=%s ATTN=%s GLU=%s F_DIM=%s | defs=%s"
          % (name, blob.get("iter"), blob.get("win_rate"), cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"),
             cfg.get("USE_ATTENTION"), cfg.get("USE_GLU"), g["F_DIM"], args.defs_nb))
    print("rendering %d agents x %d seeds = %d games -> %s\n" % (len(agents), len(seeds), len(agents) * len(seeds), outdir))

    rows = []
    for ag in agents:
        for sd in seeds:
            ticks, (s0, s1) = V.play(g, net0, ag, args.steps, sd)
            res = "win" if s0 > s1 else ("loss" if s1 > s0 else "draw")
            base = os.path.join(outdir, "%s_vs_%s_s%d" % (name, ag, sd))
            js = base + ".json"
            json.dump({"board_size": g["BOARD_SIZE"], "sun_radius": g["SUN_RADIUS"],
                       "ego": "agent", "opp": ag, "outcome": s0 - s1, "ticks": ticks}, open(js, "w"))
            env = dict(os.environ); env.pop("SSLKEYLOGFILE", None)
            subprocess.run([sys.executable, "scripts/render_recording.py", js, "--out", base,
                            "--stride", str(args.stride)], check=True, env=env)
            rows.append((ag, sd, s0, s1, res, len(ticks)))
            print("  %-8s s%d : p0=%4.0f  p1=%4.0f  %-4s  (%d ticks)" % (ag, sd, s0, s1, res, len(ticks)))

    # ---- summary table ----
    lines = ["# Render report -- `%s`\n" % name,
             "Checkpoint: `%s`  (iter %s, train win_rate %s)  " % (args.ckpt, blob.get("iter"), blob.get("win_rate")),
             "Config: HIDDEN=%s, N_RES_BLOCKS=%s, USE_ATTENTION=%s, USE_GLU=%s, F_DIM=%s  "
             % (cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"), cfg.get("USE_ATTENTION"), cfg.get("USE_GLU"), g["F_DIM"]),
             "Defs notebook: `%s`  |  %d games (greedy, %d steps)\n" % (args.defs_nb, len(rows), args.steps),
             "| agent | seed | p0 (agent) | p1 (opp) | outcome | result | ticks |",
             "|-------|------|-----------|----------|---------|--------|-------|"]
    for ag, sd, s0, s1, res, nt in rows:
        lines.append("| %s | %d | %.0f | %.0f | %+.0f | **%s** | %d |" % (ag, sd, s0, s1, s0 - s1, res, nt))
    lines.append("\n## Per-agent win rate (over %d seeds)\n" % len(seeds))
    lines.append("| agent | W-L-D | win rate | avg margin |")
    lines.append("|-------|-------|----------|-----------|")
    for ag in agents:
        sub = [r for r in rows if r[0] == ag]
        w = sum(1 for r in sub if r[4] == "win"); l = sum(1 for r in sub if r[4] == "loss"); d = sum(1 for r in sub if r[4] == "draw")
        wr = (w + 0.5 * d) / len(sub) if sub else 0.0
        margin = sum(r[2] - r[3] for r in sub) / len(sub) if sub else 0.0
        lines.append("| %s | %d-%d-%d | %.2f | %+.0f |" % (ag, w, l, d, wr, margin))
    md = os.path.join(outdir, "results.md")
    open(md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote summary ->", md)
    print("done:", outdir)


if __name__ == "__main__":
    main()
