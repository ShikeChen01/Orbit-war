"""Render a game between two SCRIPTED agents in the official Kaggle engine, and report the head-to-head
record over N games. Uses a defs notebook that actually HAS the agents (default: setup1_target_ppo.ipynb).

    python scripts/render_scripted_match.py --a medium --b ultimate --games 20 --render-seed 0
Outputs runs/viz/scripted/<a>_vs_<b>.gif + _keyframes.png (seat0=a=blue, seat1=b=red).
"""
from __future__ import annotations
import argparse, os, sys, json, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import viz_target_ckpt as V          # noqa: E402
import scripted_tournament as T      # noqa: E402  (make_scripted_agent, KIND2CODE)


def to_ticks(out):
    ticks = []
    for t, st in enumerate(out):
        obs = st[0]["observation"]
        planets = [[int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(round(p[5])), int(round(p[6]))]
                   for p in (obs["planets"] or [])]
        fleets = [[int(f[1]), float(f[2]), float(f[3]), float(f[4]), int(round(f[6]))]
                  for f in (obs.get("fleets", []) or [])]
        comets = [int(c) for c in (obs.get("comet_planet_ids", []) or [])]
        ticks.append({"t": t, "planets": planets, "fleets": fleets, "comets": comets})
    return ticks


def ships(obs, owner):
    return sum(p[5] for p in (obs["planets"] or []) if p[1] == owner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="medium"); ap.add_argument("--b", default="ultimate")
    ap.add_argument("--defs-nb", default="notebooks/setup1_target_ppo.ipynb")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--render-seed", type=int, default=0, help="which game index to render")
    args = ap.parse_args()
    os.chdir(REPO)
    V.NB = args.defs_nb
    g = V.load_notebook_defs(); g["DEVICE"] = "cpu"
    agA = T.make_scripted_agent(g, T.KIND2CODE[args.a])
    agB = T.make_scripted_agent(g, T.KIND2CODE[args.b])
    from kaggle_environments import make

    wa = wb = dr = 0
    render = None                                       # capture a game where `a` BEATS `b`, a in seat 0 (clean colors)
    for i in range(args.games):
        env = make("orbit_wars", configuration={"episodeSteps": args.steps}, debug=False)
        seat = i % 2                                    # alternate seats
        roster = [agA, agB] if seat == 0 else [agB, agA]
        out = env.run(roster)
        ra = out[-1][seat]["reward"]
        if ra is None or ra == 0:
            dr += 1
        elif ra > 0:
            wa += 1
        else:
            wb += 1
        if render is None and seat == 0 and ra is not None and ra > 0:   # a (seat0) won -> render it
            o = out[-1][0]["observation"]
            render = (out, ships(o, 0), ships(o, 1))
        print("  game %2d (a=%-8s seat%d): a-reward %s" % (i, args.a, seat, ra))

    n = wa + wb + dr
    print("\nH2H over %d games | %s %d - %d %s | draws %d | %s win-rate %.3f"
          % (n, args.a, wa, wb, args.b, dr, args.a, (wa + 0.5 * dr) / max(1, n)))

    if render is not None:
        out2, sA, sB = render
        ticks = to_ticks(out2)
        winner = args.a if sA > sB else (args.b if sB > sA else "draw")
        outdir = os.path.join("runs", "viz", "scripted"); os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, "%s_vs_%s" % (args.a, args.b))
        js = base + ".json"
        json.dump({"board_size": g["BOARD_SIZE"], "sun_radius": g["SUN_RADIUS"],
                   "ego": args.a, "opp": args.b, "outcome": sA - sB, "ticks": ticks}, open(js, "w"))
        print("rendered game: %s(blue,p0)=%d  vs  %s(red,p1)=%d  -> %s | %d ticks"
              % (args.a, sA, args.b, sB, winner, len(ticks)))
        env = dict(os.environ); env.pop("SSLKEYLOGFILE", None)
        subprocess.run([sys.executable, "scripts/render_recording.py", js, "--out", base, "--stride", str(args.stride)],
                       check=True, env=env)
        print("-> %s.gif  %s_keyframes.png" % (base, base))


if __name__ == "__main__":
    main()
