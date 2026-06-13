"""Cross-architecture head-to-head in the STRICTLY-OFFICIAL Kaggle engine.

Side A: a v5 TARGET-actor checkpoint (F_DIM 32, e.g. 8x512-MLPs-Att-1300iter.pt) adapted with
the proven eval_kaggle_engine.make_agent (v5_legacy_defs encoding).
Side B: v9 MLP-trunk agents (F_DIM 104) as their PACKAGED submission main.py (exec'd exactly
like Kaggle does -- the seat-preserving adapter with bit-exact parity).

2p: every (A vs B) pairing, seats alternated per game. 4p FFA: A + the three v9 agents in one
game, cyclic seat rotation; per-seat final scores recomputed from the last observation
(ships on owned planets + ships in flight), so we get full rankings, not just the engine's
winner-takes +1/-1 reward.

    python scripts/h2h_v5att_vs_v9.py --ckpt notebooks/weight/MLPs/8x512-MLPs-Att-1300iter.pt \
        --v9-dirs runs/h2h_v5v9/agents/it201 runs/h2h_v5v9/agents/it351 runs/h2h_v5v9/agents/it401 \
        --games2 60 --games4 40
"""
from __future__ import annotations
import argparse, math, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only: the kaggle engine is pure Python and the
                                            # notebook defs would otherwise open a CUDA context per worker

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

_W = {}


def _init_worker(ckpt, defs_nb, v9_dirs, steps):
    import torch as _t
    _t.set_num_threads(1)                       # 1 thread/process; the pool fills the cores
    os.chdir(REPO)
    import viz_target_ckpt as V
    from eval_kaggle_engine import make_agent
    V.NB = defs_nb
    g = V.load_notebook_defs()
    g["DEVICE"] = _t.device("cpu")
    if "PHI_BINS_T" in g:
        g["PHI_BINS_T"] = g["PHI_BINS_T"].to(g["DEVICE"])
    net, _cfg, _blob = V.build_actor(g, ckpt)
    _W["A"] = make_agent(g, net.to(g["DEVICE"]).eval())
    _W["B"] = {}
    for d in v9_dirs:                           # exec each packaged main.py exactly like Kaggle
        name = os.path.basename(os.path.normpath(d))
        src = open(os.path.join(REPO, d, "main.py"), encoding="utf-8").read()
        cwd = os.getcwd()
        os.chdir(os.path.join(REPO, d))         # _find_weights() probes cwd for weights.pt
        try:
            ns = {}
            exec(compile(src, "main.py", "exec"), ns)
        finally:
            os.chdir(cwd)
        _W["B"][name] = ns["agent"]
    _W["steps"] = steps


def _final_scores(out, n):
    """Per-player final score from the last observation: ships on owned planets + in-flight."""
    obs0 = out[-1][0]["observation"]
    scores = [0.0] * n
    for p in obs0["planets"]:
        if p[1] is not None and p[1] >= 0:
            scores[int(p[1])] += float(p[5])
    for f in obs0["fleets"]:
        if f[1] is not None and f[1] >= 0:
            scores[int(f[1])] += float(f[6])
    return scores


def _play2(task):
    kind, opp, idx = task
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"episodeSteps": _W["steps"]}, debug=False)
    seat = idx % 2                              # A's seat alternates for fairness
    roster = [_W["A"], _W["B"][opp]] if seat == 0 else [_W["B"][opp], _W["A"]]
    t0 = time.perf_counter()
    out = env.run(roster)
    dt = time.perf_counter() - t0
    statuses = [out[-1][k]["status"] for k in range(2)]
    s = _final_scores(out, 2)
    a, b = s[seat], s[1 - seat]
    res = "w" if a > b else ("l" if a < b else "d")
    return ("2p", opp, res, dt, statuses)


def _play4(task):
    kind, _, idx = task
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"episodeSteps": _W["steps"]}, debug=False)
    names = ["A"] + sorted(_W["B"])             # fixed agent order; rotate seats cyclically
    rot = idx % 4
    seat_names = [names[(k - rot) % 4] for k in range(4)]
    roster = [_W["A"] if n == "A" else _W["B"][n] for n in seat_names]
    t0 = time.perf_counter()
    out = env.run(roster)
    dt = time.perf_counter() - t0
    statuses = [out[-1][k]["status"] for k in range(4)]
    s = _final_scores(out, 4)
    by_agent = {seat_names[k]: s[k] for k in range(4)}
    return ("4p", "-", by_agent, dt, statuses)


def _ci(wr, n):
    return 1.96 * math.sqrt(max(wr * (1.0 - wr), 1e-9) / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="notebooks/weight/MLPs/8x512-MLPs-Att-1300iter.pt")
    ap.add_argument("--defs-nb", default="notebooks/render/v5_legacy_defs.ipynb")
    ap.add_argument("--v9-dirs", nargs="+", default=[
        "runs/h2h_v5v9/agents/it201", "runs/h2h_v5v9/agents/it351", "runs/h2h_v5v9/agents/it401"])
    ap.add_argument("--games2", type=int, default=60, help="2p games per (A vs B) pairing")
    ap.add_argument("--games4", type=int, default=40, help="4p FFA games (all four agents)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    args = ap.parse_args()
    os.chdir(REPO)
    from concurrent.futures import ProcessPoolExecutor

    opps = [os.path.basename(os.path.normpath(d)) for d in args.v9_dirs]
    tasks = [("2p", op, i) for op in opps for i in range(args.games2)]
    tasks += [("4p", "-", i) for i in range(args.games4)]
    print("A = %s (v5 target-actor) | B = %s (v9 packaged)" % (os.path.basename(args.ckpt), opps))
    print("%d x 2p/pairing + %d x 4p = %d games | %d workers | stock Kaggle engine (comets on)\n"
          % (args.games2, args.games4, len(tasks), args.workers), flush=True)

    tally = {op: {"w": 0, "l": 0, "d": 0} for op in opps}
    pw4 = {}        # agent -> [sum pairwise score, n games]
    place4 = {}     # agent -> [count rank0..rank3]
    durs, bad = [], 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.ckpt, args.defs_nb, args.v9_dirs, args.steps)) as ex:
        for kind, op, res, dt, statuses in ex.map(_play2_or_4, tasks, chunksize=1):
            durs.append(dt); done += 1
            if any(s == "ERROR" for s in statuses):
                bad += 1
            if kind == "2p":
                tally[op][res] += 1
            else:
                sc = res                                  # dict agent -> final score
                names = sorted(sc)
                for a in names:
                    others = [b for b in names if b != a]
                    pw = sum(1.0 if sc[a] > sc[b] else (0.5 if sc[a] == sc[b] else 0.0)
                             for b in others) / len(others)
                    rank = sum(1 for b in others if sc[b] > sc[a])
                    pw4.setdefault(a, [0.0, 0])
                    pw4[a][0] += pw; pw4[a][1] += 1
                    place4.setdefault(a, [0, 0, 0, 0])
                    place4[a][rank] += 1
            if done % 10 == 0:
                print("  [%3d/%3d] mean game %.0fs" % (done, len(tasks), sum(durs) / len(durs)),
                      flush=True)

    lines = []
    lines.append("==== 2p H2H: %s (A) vs each v9 agent | %d games/pairing, seats alternated ===="
                 % (os.path.basename(args.ckpt), args.games2))
    lines.append("%-8s | %-4s %-4s %-4s | A's score (95%% CI)" % ("opp", "W", "L", "D"))
    lines.append("-" * 52)
    for op in opps:
        t = tally[op]; n = t["w"] + t["l"] + t["d"]
        wr = (t["w"] + 0.5 * t["d"]) / max(1, n)
        lines.append("%-8s | %-4d %-4d %-4d | %.3f +/- %.3f" % (op, t["w"], t["l"], t["d"], wr, _ci(wr, n)))
    if pw4:
        lines.append("")
        lines.append("==== 4p FFA (A + all three v9, %d games, cyclic seat rotation) ====" % args.games4)
        lines.append("%-8s | %-9s | %-6s %-6s %-6s %-6s" % ("agent", "mean pw", "1st", "2nd", "3rd", "4th"))
        lines.append("-" * 56)
        order = sorted(pw4, key=lambda a: -(pw4[a][0] / pw4[a][1]))
        for a in order:
            mp = pw4[a][0] / pw4[a][1]
            pl = place4[a]; n = sum(pl)
            lines.append("%-8s | %9.3f | %5.1f%% %5.1f%% %5.1f%% %5.1f%%"
                         % (a, mp, *[100.0 * c / n for c in pl]))
    if bad:
        lines.append("\n!! %d game(s) had an agent ERROR status -- inspect before trusting" % bad)
    txt = "\n".join(lines)
    print("\n" + txt)
    od = os.path.join("runs", "h2h_v5v9")
    os.makedirs(od, exist_ok=True)
    open(os.path.join(od, "results.txt"), "w").write(txt + "\n")


def _play2_or_4(task):
    return _play2(task) if task[0] == "2p" else _play4(task)


if __name__ == "__main__":
    main()
