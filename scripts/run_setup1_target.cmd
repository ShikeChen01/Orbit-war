@echo off
REM ----------------------------------------------------------------------------
REM set-ups/1.md, v5 TARGET actor: the model picks (destination planet, phi=% of that
REM planet's ships); a heuristic aims the launch straight at the destination (atan2). NO
REM continuous aiming to learn -> launching is finally tractable. One launch per planet per step.
REM
REM Reward (replaces the old production-shaping reward entirely):
REM   win/loss = +/-1000, FLAT for the first 100 steps then * 0.995^(steps-100) (a "flip of winning")
REM   capture a planet +/-30
REM   first 50 VALID launches: +1 each
REM   production milestone: +20 every time ego total ships doubles (100 -> 200 -> 400 -> ...)
REM   (fleet-hit reward dropped; PPO value targets /100 for MSE stability)
REM
REM Curriculum (4-stage, iteration-driven):
REM   stage 2  vs random                       iters   1-100   (--stage1-iters 0 --stage2-iters 100)
REM   stage 3  vs starter                       iters 101-400   (--stage3-iters 300)
REM   stage 4  self-play + starter + random     iters 401-1400  (selfplay 50%, starter 25%, random 25%)
REM
REM Model: per-planet trunk, width 512, 6 residual blocks. lr 1e-4. PPO+GAE. target_actor forces
REM max_entities = planet_cap (the destination categorical is over the 48 obs slots).
REM ----------------------------------------------------------------------------
cd /d "%~dp0.."

call native\run.cmd ow_train_grpo --algo ppo --gpu-env 1 ^
  --target-actor 1 --fleets 1 ^
  --hidden 512 --n-res-blocks 6 --lr 1e-4 ^
  --num-groups 16 --group-size 8 --minibatches 64 --update-epochs 1 ^
  --stage1-iters 0 --stage2-iters 100 --stage3-iters 300 --total-iters 1400 ^
  --ckpt-every-early 50 --ckpt-every-late 100 ^
  --win-bonus 1000 --loss-penalty 1000 --win-decay 0.995 --loss-decay 0.995 --decay-start-step 100 ^
  --capture-reward 30 --dispatch-reward 1 --dispatch-count 50 ^
  --prod-milestone-reward 20 --prod-milestone-base 100 ^
  --reward-scale 100 --selfplay-prob 0.5 ^
  --prod-reward-weight 0 --valid-launch-reward 0 --illegal-launch-penalty 0 ^
  --miss-launch-penalty 0 --enemy-growth-weight 0 ^
  --logstd-max 0 --logstd-max-end -1.2 --logstd-max-post -1.0 --sigma-decay-iters 100 --ent-coef 0.005 ^
  --gae-lambda 0.95 --ppo-gamma 0.99 --vf-coef 0.5 --clip 0.2 ^
  --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp ^
  --run-dir runs/grpo/setup1_target
