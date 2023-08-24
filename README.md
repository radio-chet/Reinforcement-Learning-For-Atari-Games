# Reinforcement-Learning-For-Atari-Games
As a part of the mini-project, we tried to implement different policy gradient reinforcement learning
(RL) algorithms for the breakout, ping-pong, and space-invaders atari games. Initially, we used
the stable-baselines3 codes to train the RL agents from scratch in the atari environment. However,
we were unable to get the expected results or make the algorithm work for different environments.
Hence we used the algorithms implemented in the Ray rllib library for Advantage actor-critic (A2C),
Asynchronous actor-critic (A3C), and Proximal Policy Optimization (PPO) agents. We trained the
agents from scratch and tried to match the results with the benchmark results given on their website
for different atari games.
