@echo off
REM ----------------------------------------------------------------------------
REM Sequential PPO+GAE then GRPO under the PRODUCTION-ONLY reward redesign.
REM Reward kept: production gain + enemy-suppression + win/loss/draw outcome.
REM Reward REMOVED (firing penalties backfired -> passivity): valid-launch carrot,
REM   miss penalty, illegal-launch penalty (all weights forced to 0).
REM Model: per-planet trunk with a 5-block ResNet MLP (hidden d=256), GLU on.
REM Curriculum (iteration-driven): stage 2 (vs starter) iters 1-300, stage 3 (mix)
REM   iters 300-1300. Heartbeat log every iter; full eval + ckpt/itNNNN.owc every 100.
REM Both runs share IDENTICAL reward/model/curriculum so the algo comparison is clean.
REM ----------------------------------------------------------------------------
cd /d "%~dp0.."

set "COMMON=--gpu-env 1 --hidden 256 --n-res-blocks 5 --num-groups 32 --group-size 8 --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp --stage1-iters 0 --stage2-iters 300 --total-iters 1300 --ckpt-every-early 100 --ckpt-every-late 100 --update-epochs 1 --minibatches 32 --prod-reward-weight 2.5 --prod-reward-cap 100 --prod-reward-decay 0.997 --valid-launch-reward 0 --illegal-launch-penalty 0 --miss-launch-penalty 0 --enemy-growth-weight 0.5 --enemy-growth-warmup 50 --enemy-growth-ema 0.1 --win-bonus 10 --loss-penalty 10 --logstd-max 0 --logstd-max-end -1.2 --logstd-max-post -1.0 --sigma-decay-iters 60 --ent-coef 0.005 --lr 3e-4"

echo ============================================================
echo [1/2] PPO+GAE  to  runs/grpo/ppo_prod1
echo ============================================================
call native\run.cmd ow_train_grpo --algo ppo %COMMON% --run-dir runs/grpo/ppo_prod1
if errorlevel 1 (
  echo PPO run FAILED ^(errorlevel %errorlevel%^) -- skipping GRPO.
  exit /b 1
)

echo ============================================================
echo [2/2] GRPO     to  runs/grpo/grpo_prod1
echo ============================================================
call native\run.cmd ow_train_grpo %COMMON% --gamma 1.0 --run-dir runs/grpo/grpo_prod1
