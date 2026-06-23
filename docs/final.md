here is my instruction for the final notebook which I will train for submission:
base: v12 ( battle tested heuristic engine with complete info, comet trivial for the most part)

method: PPO+GAE value resnet = 2
Architecture:
16res, 1attn, 16res, 1attn, 32res, 1attn, 32res, 1attn, 32res, 1attn, 32res, h=256 (should be at the edge of packup and upload)

deployment: MCTS on critical steps (trigger every 10 steps, search for depth of 10 and breadth of whatever it can) for 10s, for 5 times, reserve 10s in banks for the rest of the game inference

AUX_reward=ON
Potential_Shape_reward=ON

reward design: 

wins from a pool of 1200 reward. gives 2 reward each step from the pool. winning wins the rest of the pool
losing stays the same, decay at 0.9995 on 1000

betting wins maximum of extra 1 reward each step, single sided winning, model can't bet on loses

the sub of the rest of the reward have a hard cap on 250 (capture and production and launch), they share the cap

flag me if anything. be very careful about this implementation. 





