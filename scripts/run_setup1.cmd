@echo off
REM ----------------------------------------------------------------------------
REM docs/set-ups/1.md : PPO+GAE with the event reward set + 4-stage self-play curriculum.
REM
REM Reward (REPLACES the production-shaping reward; all legacy weights forced to 0):
REM   1. win +1000 / loss -1000 / draw 0
REM   2. win AND loss decay by game length: +/-1000 * 0.995^steps  (a flip of winning; win/lose fast)
REM   3. capture a planet +50, lose a planet -50
REM   4. the first 50 fleet dispatches get +1 each (per game)
REM   5. any ego fleet that hits a planet gets 5 + ships, capped at 300 per game
REM   (PPO value targets rescaled by --reward-scale 100 so the critic stays in a fittable range;
REM    advantages are re-normalized, so the policy objective is unchanged.)
REM
REM Curriculum (iteration-driven, 4 stages):
REM   stage 1 vs stationary (noop)   iters   1-100
REM   stage 2 vs random              iters 101-200
REM   stage 3 vs starter             iters 201-400
REM   stage 4 self-play + starter    iters 401-1400   (frozen snapshot refreshed every 100 iters)
REM
REM Model: per-planet trunk, width 1024, 20 residual blocks (GLU on). lr 1e-4. PPO+GAE on GPU env.
REM ----------------------------------------------------------------------------
cd /d "%~dp0.."

call native\run.cmd ow_train_grpo --algo ppo --gpu-env 1 ^
  --hidden 1024 --n-res-blocks 20 --lr 1e-4 ^
  --num-groups 16 --group-size 8 --minibatches 64 --update-epochs 1 ^
  --stage1-iters 100 --stage2-iters 100 --stage3-iters 200 --total-iters 1400 ^
  --ckpt-every-early 50 --ckpt-every-late 100 ^
  --win-bonus 1000 --loss-penalty 1000 --win-decay 0.995 --loss-decay 0.995 ^
  --capture-reward 50 --dispatch-reward 1 --dispatch-count 50 ^
  --fleet-hit-base 5 --fleet-hit-ship-weight 1 --fleet-hit-cap 300 ^
  --reward-scale 100 --selfplay-prob 0.5 ^
  --prod-reward-weight 0 --valid-launch-reward 0 --illegal-launch-penalty 0 ^
  --miss-launch-penalty 0 --enemy-growth-weight 0 ^
  --logstd-max 0 --logstd-max-end -1.2 --logstd-max-post -1.0 --sigma-decay-iters 100 --ent-coef 0.005 ^
  --gae-lambda 0.95 --ppo-gamma 0.99 --vf-coef 0.5 --clip 0.2 ^
  --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp ^
  --run-dir runs/grpo/setup1
