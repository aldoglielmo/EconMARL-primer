# EconMARL-primer
A short course on RL and multi-agent RL with applications in economics. The course covers the foundations of reinforcement learning and multi-agent reinforcement learning (MARL).

## L1: Intro and RL Basics
- Multi-agent systems overview and motivation for MARL.
- RL fundamentals: agent-environment loop and MDP definition.
- MDP assumptions (Markov property, full observability, stationarity, unknown dynamics).
- Discounted returns, value functions, and Bellman equations.
- Dynamic programming and temporal-difference learning (introductory coverage).
- Examples: level-based foraging and economy as a multi-agent system.

## L2: Deep Reinforcement Learning
- Recap of tabular Q-learning and Bellman updates.
- Deep Q-learning with function approximation.
- Moving target problem and target networks.
- Correlated experience problem and replay buffers.
- DQN algorithm and training loop.
- Overestimation bias and Double DQN.
- Policy gradient methods (intro section follows DQN content).

## L3: Games and MARL Basics
- From MDPs to game models for multi-agent settings.
- Normal-form games, matrix games, and repeated games.
- Stochastic games and POSGs, with formal definitions and processes.
- Game classes: zero-sum, common-reward, general-sum.
- Assumptions in game theory vs MARL.
- Solution concepts and joint policy reasoning (outlined in the deck).

## L4: MARL Algorithms and Challenges
- MARL learning framework and convergence notions.
- Inputs to policies depending on game model (normal-form, repeated, stochastic, POSG).
- Single-agent RL reductions: central learning vs independent learning.
- Central Q-learning and Independent Q-learning (IQL).
- Modes of learning: self-play and mixed-play.
- Key challenges: non-stationarity, equilibrium selection, credit assignment, scaling.

