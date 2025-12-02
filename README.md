# IOTA 5201 Assignment: Application of Reinforcement Learning on Cyber-Physical System(CPS)
This is the code submission for the IOTA 5201 course. Core content includes:
- Implementation of SAC with Improved HER buffer
- Application of IHER-SAC on FetchPickAndPlace-v4 environment

![iher_sac](https://github.com/xxkklose/rl_cps_course/raw/master/assets/iher_sac.gif)
![iher_sac_fetch](https://github.com/xxkklose/rl_cps_course/raw/master/assets/iher_sac_fetch.gif)

# Installation
git clone this repository:
```bash
git clone https://github.com/xxkklose/rl_cps_course.git
```

we recommend using uv to install the dependencies:
```bash
cd /path/to/rl_cps_course
uv sync
```
or you can use requirements.txt for other package managers as you like:
```bash
cd /path/to/rl_cps_course
pip install -e .
```

# Usage
To train IHER-SAC on FetchPickAndPlace-v4 environment:
```bash
bash train.sh iher_sac FetchPickAndPlace-v4
```
## Benchmark Algorithms Training
To train SAC:
```bash
bash train.sh sac FetchPickAndPlace-v4
```
To train HER-SAC:
```bash
bash train.sh her_sac FetchPickAndPlace-v4
```
To train PPO:
```bash
bash train.sh ppo FetchPickAndPlace-v4
```
Parameters:
- `iher_sac`, `her_sac`, `sac`, `ppo`: Algorithms to train.
- `FetchPickAndPlace-v4`: Environment to train on.
- `SEED`: Random seed for reproducibility.
- `TOTAL_TIMESTEPS`: Total timesteps to train.
- `BUFFER_SIZE`: Size of the replay buffer.
- `LEARNING_RATE`: Learning rate for the optimizer.
- `BATCH_SIZE`: Batch size for training.
- `TAU`: Soft update coefficient for target networks.
- `GAMMA`: Discount factor for future rewards.
- `DEVICE`: Device to use for training (cuda or cpu).

# Evaluation
To evaluate the trained models, you can use the following command:
```bash
bash eval.sh iher_sac FetchPickAndPlace-v4 202511281553
```
You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1uKdLiq7VHnL6fQFf4btCON1dNm_XQWmp) to the folder `checkpoints/`.

Parameters:
- `iher_sac`, `her_sac`, `sac`, `ppo`: Algorithms to evaluate.
- `FetchPickAndPlace-v4`: Environment to evaluate on.
- `202511281553`: Checkpoint name to load.

# Inspiration
This repository drew inspiration from the following resources:
- [OpenAI Baselines](https://github.com/openai/baselines)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
- [mujoco_learning](https://github.com/Albusgive/mujoco_learning.git)

If this repository is helpful to you, please give it a star and cite it in your work.
