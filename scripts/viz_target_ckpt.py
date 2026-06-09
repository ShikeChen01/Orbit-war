"""Visualize a v5 TARGET-actor (.pt, notebook-format) checkpoint actually playing a game,
then render it to GIF + keyframe PNG via scripts/render_recording.py.

Loads the policy/env definitions straight from notebooks/setup1_target_ppo.ipynb (no training run),
builds a PolicyNet matching the CHECKPOINT's own config, and loads the ACTOR weights with
strict=False (the value head is ignored -- it is not needed to act, and may be an older variant).

    python scripts/viz_target_ckpt.py --ckpt notebooks/1400iter-20hidden-MLPs/weights/setup1_target_policy_best.pt \
        --opponent starter --seed 0 --stride 3
    python scripts/viz_target_ckpt.py --ckpt <best.pt> --opponent self        # vs a copy of itself
    python scripts/viz_target_ckpt.py --ckpt <best.pt> --opponent <other.pt>  # vs another checkpoint
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import torch

NB = "notebooks/setup1_target_ppo.ipynb"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_notebook_defs():
    """Exec the notebook's code cells up to (but not including) the training run -> a namespace."""
    doc = json.load(open(os.path.join(REPO, NB), encoding="utf-8"))
    parts = []
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "= train(" in s and "def train(" not in s:        # stop before the training cell
            break
        parts.append(s)
    g = {"__name__": "__viz__"}
    exec(compile("\n\n".join(parts), "nb_defs", "exec"), g)
    return g


def build_actor(g, path):
    """Build a PolicyNet matching the ckpt config; load actor weights (strict=False -> skip value head)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob.get("config") or {}                       # config may be absent OR explicitly None
    # config-less checkpoints (e.g. E801): infer arch from the weight keys themselves
    import re
    mkeys = list(blob["model"].keys())
    has_attn_w = any(k.startswith("attn_") for k in mkeys)
    has_glu_w = any(k.startswith("glu_") for k in mkeys)
    res_idx = [int(m.group(1)) for k in mkeys for m in [re.match(r"res_a\.(\d+)\.", k)] if m]
    n_res_w = (max(res_idx) + 1) if res_idx else g["N_RES_BLOCKS"]
    PolicyNet = g["PolicyNet"]
    net = PolicyNet(
        g["F_DIM"], g["G_DIM"], cfg.get("HIDDEN", g["HIDDEN"]), cfg.get("D_G", g["D_G"]),
        g["PLANET_CAP"], cfg.get("N_RES_BLOCKS", n_res_w), cfg.get("USE_GLU", has_glu_w),
        cfg.get("STD_STATE_DEP", g["STD_STATE_DEP"]), g["LOGSTD_MIN"], g["LOGSTD_MAX"],
        g["INIT_MU_SCALE"], g["INIT_PHI_BIAS"],
        use_attention=cfg.get("USE_ATTENTION", has_attn_w), value_res_blocks=0,
    )
    # phi-bucket count must match what the actor was trained with
    npx = blob["model"]["phi_cat.bias"].shape[0]
    assert npx == g["N_PHI"], "checkpoint has %d phi buckets but notebook N_PHI=%d" % (npx, g["N_PHI"])
    missing, unexpected = net.load_state_dict(blob["model"], strict=False)
    actor_missing = [k for k in missing if not k.startswith(("val_in", "val_res", "val_out"))]
    assert not actor_missing, "actor weights missing from ckpt: %s" % actor_missing
    net.logstd_max = g["LOGSTD_MAX_POST"]
    return net.to(g["DEVICE"]).eval(), blob.get("config", {}), blob


def frame(env, t):
    pa = env.p_alive[0].cpu().numpy() > 0.5
    po = env.p_owner[0].cpu().numpy(); px = env.p_x[0].cpu().numpy(); py = env.p_y[0].cpu().numpy()
    pr = env.p_radius[0].cpu().numpy(); ps = env.p_ships[0].cpu().numpy(); pp = env.p_prod[0].cpu().numpy()
    pc = env.p_is_comet[0].cpu().numpy() > 0.5
    planets = [[i, int(po[i]), float(px[i]), float(py[i]), float(pr[i]), int(round(ps[i])), int(round(pp[i]))]
               for i in range(len(pa)) if pa[i]]
    fa = env.f_alive[0].cpu().numpy() > 0.5
    fo = env.f_owner[0].cpu().numpy(); fx = env.f_x[0].cpu().numpy(); fy = env.f_y[0].cpu().numpy()
    fang = env.f_angle[0].cpu().numpy(); fs = env.f_ships[0].cpu().numpy()
    fleets = [[int(fo[j]), float(fx[j]), float(fy[j]), float(fang[j]), int(round(fs[j]))]
              for j in range(len(fa)) if fa[j]]
    comets = [i for i in range(len(pc)) if pc[i]]
    return {"t": t, "planets": planets, "fleets": fleets, "comets": comets}


def play(g, net0, opp, steps, seed):
    """opp: 'starter'|'random'|'noop' (scripted) or a PolicyNet (self/other). Returns ticks, (s0,s1)."""
    GpuEnv, env_encode, act, env_step, settle = g["GpuEnv"], g["env_encode"], g["act"], g["env_step"], g["settle"]
    OPP = {"random": 0, "starter": 1, "noop": 2}
    env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], steps, g["SHIP_SPEED"], g["DEVICE"])
    env.reset([g["generate_world"](seed)])
    ticks = [frame(env, 0)]
    for t in range(1, steps + 1):
        ent, em, am, gl = env_encode(env, 0)
        a_t, _, _ = act(net0, ent, em, am, gl, greedy=True)
        if isinstance(opp, str):
            env_step(env, a_t, OPP[opp], g["ACT_THRESHOLD"], None)
        else:
            o = env_encode(env, 1)
            oa, _, _ = act(opp, *o, greedy=True)
            env_step(env, a_t, 1, g["ACT_THRESHOLD"], oa)
        ticks.append(frame(env, t))
        s0, s1, a0, a1 = settle(env)
        if not (bool(a0[0]) and bool(a1[0])) or env.step_ct[0].item() >= env.T - 1:
            break
    s0, s1, _, _ = settle(env)
    return ticks, (float(s0[0]), float(s1[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", default="starter", help="starter|random|noop|self|<path-to-ckpt.pt>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default=None, help="output prefix (default: runs/viz/target/<name>)")
    args = ap.parse_args()
    os.chdir(REPO)

    g = load_notebook_defs()
    net0, cfg, blob = build_actor(g, args.ckpt)
    print("loaded %s  iter=%s win_rate=%s  | HIDDEN=%s N_RES_BLOCKS=%s ATTN=%s GLU=%s"
          % (os.path.basename(args.ckpt), blob.get("iter"), blob.get("win_rate"),
             cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"), cfg.get("USE_ATTENTION"), cfg.get("USE_GLU")))

    if args.opponent in ("starter", "random", "noop"):
        opp, opp_tag = args.opponent, args.opponent
    elif args.opponent == "self":
        opp, opp_tag = build_actor(g, args.ckpt)[0], "self"
    else:
        opp, opp_tag = build_actor(g, args.opponent)[0], os.path.splitext(os.path.basename(args.opponent))[0]

    ticks, (s0, s1) = play(g, net0, opp, args.steps, args.seed)
    winner = "p0 (agent)" if s0 > s1 else ("p1 (%s)" % opp_tag if s1 > s0 else "draw")
    print("game: %d ticks | final ships  p0=%.0f  p1=%.0f  -> %s" % (len(ticks), s0, s1, winner))

    name = "%s_vs_%s_s%d" % (os.path.splitext(os.path.basename(args.ckpt))[0], opp_tag, args.seed)
    out = args.out or os.path.join("runs", "viz", "target", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    js = out + ".json"
    json.dump({"board_size": g["BOARD_SIZE"], "sun_radius": g["SUN_RADIUS"], "ticks": ticks}, open(js, "w"))
    print("recording ->", js)

    env = dict(os.environ); env.pop("SSLKEYLOGFILE", None)
    subprocess.run([sys.executable, "scripts/render_recording.py", js, "--out", out, "--stride", str(args.stride)],
                   check=True, env=env)
    print("-> %s.gif   %s_keyframes.png" % (out, out))


if __name__ == "__main__":
    main()
