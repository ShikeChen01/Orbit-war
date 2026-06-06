#!/usr/bin/env bash
# Generate behavior visualizations for every checkpoint of a native run:
#   for each run_dir/ckpt/itNNNN.owc  ->  record one greedy game (CPU) -> render GIF + keyframe PNG,
# then plot the training curves. Re-runnable (skips re-recording if the JSON already exists).
#
#   bash scripts/make_viz.sh runs/grpo/ppo_curric 2            # n_res_blocks=2, opp=starter
#   bash scripts/make_viz.sh runs/grpo/grpo_curric 2 starter 3
set -u
RUN="${1:?usage: make_viz.sh <run_dir> <n_res_blocks> [opponent] [stride]}"
NRES="${2:-2}"; OPP="${3:-starter}"; STRIDE="${4:-3}"
NAME=$(basename "$RUN")
OUT="runs/viz/$NAME"; mkdir -p "$OUT"
shopt -s nullglob
for ck in "$RUN"/ckpt/*.owc; do
  tag=$(basename "$ck" .owc)
  if [ ! -f "$OUT/$tag.json" ]; then
    MSYS2_ARG_CONV_EXCL='*' cmd /c "native\\run.cmd ow_reward_trace --ego policy --ckpt $ck --n-res-blocks $NRES --opponent $OPP --stage 2 --greedy 1 --record-json $OUT/$tag.json --out $OUT/$tag.csv" >/dev/null 2>&1 || { echo "REC FAIL $tag"; continue; }
  fi
  env -u SSLKEYLOGFILE .venv/Scripts/python.exe scripts/render_recording.py "$OUT/$tag.json" --stride "$STRIDE" >/dev/null 2>&1 \
    && echo "viz $tag -> $OUT/$tag.gif" || echo "RENDER FAIL $tag"
done
env -u SSLKEYLOGFILE .venv/Scripts/python.exe scripts/plot_metrics.py "$RUN/metrics.csv" --stage1 100 --stage2 100 --out "$OUT/metrics.png"
echo "DONE: $OUT"
