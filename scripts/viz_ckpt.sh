#!/usr/bin/env bash
# ON-DEMAND: render one saved checkpoint's actual gameplay (greedy p0 vs scripted opp) to GIF + PNG.
# Training only saves weights; call this when you want to watch a specific checkpoint play.
#
#   bash scripts/viz_ckpt.sh runs/grpo/ppo_curric/ckpt/it0200.owc        # n_res=2, vs starter
#   bash scripts/viz_ckpt.sh <ckpt.owc> [n_res_blocks] [opponent] [stride] [seed]
set -u
CK="${1:?usage: viz_ckpt.sh <ckpt.owc> [n_res_blocks] [opponent] [stride] [seed]}"
NRES="${2:-2}"; OPP="${3:-starter}"; STRIDE="${4:-3}"; SEED="${5:-0}"
run=$(basename "$(dirname "$(dirname "$CK")")"); tag=$(basename "$CK" .owc)
OUT="runs/viz/$run"; mkdir -p "$OUT"
MSYS2_ARG_CONV_EXCL='*' cmd /c "native\\run.cmd ow_reward_trace --ego policy --ckpt $CK --n-res-blocks $NRES --opponent $OPP --stage 2 --greedy 1 --seed $SEED --record-json $OUT/$tag.json --out $OUT/$tag.csv" 2>&1 | grep -E "launches/st|RETURN|frames" || true
env -u SSLKEYLOGFILE .venv/Scripts/python.exe scripts/render_recording.py "$OUT/$tag.json" --stride "$STRIDE"
echo "-> $OUT/$tag.gif   $OUT/${tag}_keyframes.png"
