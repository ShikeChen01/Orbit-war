@echo off
REM ----------------------------------------------------------------------------
REM verify4, trained long. Decaying-loss reward (the "flip of winning"):
REM   win  = +1000 * 0.995^steps,  loss = -1000 * 0.995^steps,  draw = 0
REM   capture +/-50,  first-50 dispatch +1,  fleet-hit 5+ships cap 300/game
REM   (PPO value targets /100 for MSE stability; legacy prod/valid/miss/illegal off)
REM
REM Curriculum (iteration-driven; NO stationary stage):
REM   stage 1  vs random                       iters   1-100   (--stage1-iters 0 --stage2-iters 100)
REM   stage 2  vs starter                       iters 101-400   (--stage3-iters 300)
REM   stage 3  self-play + starter + random     iters 401-1400  (selfplay 50%, starter 25%, random 25%)
REM
REM Model: per-planet trunk, width 512, 20 residual blocks (= the verify4 model). lr 1e-4. PPO+GAE.
REM (docs/set-ups/1.md lists model dim 1024 for the full run; this keeps verify4's 512.)
REM ----------------------------------------------------------------------------
cd /d "%~dp0.."

call native\run.cmd ow_train_grpo --algo ppo --gpu-env 1 ^
  --hidden 512 --n-res-blocks 20 --lr 1e-4 ^
  --num-groups 16 --group-size 8 --minibatches 64 --update-epochs 1 ^
  --stage1-iters 0 --stage2-iters 100 --stage3-iters 300 --total-iters 1400 ^
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
  --run-dir runs/grpo/verify4_long
