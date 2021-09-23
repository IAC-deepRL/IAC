## IAC: [Integrated Actor-Critic for Deep Reinforcement Learning - ICANN 2020](https://www.springerprofessional.de/en/integrated-actor-critic-for-deep-reinforcement-learning/19652718)
## SIAC: Stochastic Integrated Actor-Critic for Deep Reinforcement Learning


I am the author of [ElegantRL (over 1k stars)](https://github.com/AI4Finance-Foundation/ElegantRL). 

You can find the code of IAC: [Integrated Actor-Critic for Deep Reinforcement Learning - ICANN 2020](https://www.springerprofessional.de/en/integrated-actor-critic-for-deep-reinforcement-learning/19652718) in this repo. And you can also find a latest version in [ElegantRL Agent.py](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agent.py):
- IAC(Intergrated Actor-Critic), or call it SharedAC
- SIAC(Stochastic Integrated Actor-Critic), or call it SharedSAC



## Framework
![File_structure](https://github.com/Yonv1943/ElegantRL/blob/master/figs/File_structure.png)

   An agent (in **agent.py**) uses networks (in **net.py**) and is trained (in **run.py**) by interacting with an environment (in **env.py**).
   
A high-level overview:
+ 1). Instantiate an environment in **Env.py**, and an agent in **Agent.py** with an Actor network and a Critic network in **Net.py**; 
+ 2). In each training step in **Run.py**, the agent interacts with the environment, generating transitions that are stored into a Replay Buffer; 
+ 3). The agent fetches a batch of transitions from the Replay Buffer to train its networks; 
+ 4). After each update, an evaluator evaluates the agent's performance (e.g., fitness score or cumulative return) and saves the agent if the performance is good.

## Code Structure
### Core Codes
+ **elegantrl/net.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Neural networks.
   + Q-Net,
   + Actor network,
   + Critic network, 
+ **elegantrl/agent.py**  &nbsp;&nbsp;# RL algorithms. 
   + AgentBase, 
+ **elegantrl/run.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# run DEMO 1 ~ 4
   + Parameter initialization,
   + Training loop,
   + Evaluator.



## Start to Train

### Initialization:
+ hyper-parameters `args`.
+ `env = PreprocessEnv()` : creates an environment (in the OpenAI gym format).
+ `agent = agent.XXX()` : creates an agent for a DRL algorithm.
+ `buffer = ReplayBuffer()` : stores the transitions.
+ `evaluator = Evaluator()` : evaluates and stores the trained model.

### Training (a while-loop):
+ `agent.explore_env(…)`: the agent explores the environment within target steps, generates transitions, and stores them into the ReplayBuffer.
+ `agent.update_net(…)`: the agent uses a batch from the ReplayBuffer to update the network parameters.
+ `evaluator.evaluate_save(…)`: evaluates the agent's performance and keeps the trained model with the highest score.

The while-loop will terminate when the conditions are met, e.g., achieving a target score, maximum steps, or manually breaks.

## Experiment

### Experiment 5.1 Comparisons with Benchmark Algorithms

I am the author of [ElegantRL (over 1k stars)](https://github.com/AI4Finance-Foundation/ElegantRL). 
You can find other DRL algorithms in this repo. (DDPG, TD3, SAC, PPO, ...)

### Experiment 5.2 Self-comparisons

#### Table 1. Levels of IAC for self-comparisons

It is easy to run self-comparisons. Just replace the network to corresponding verison.

| Level    |  Network| Description |
| -- | ----------- | ----------------------------------------------------------------------|
| L1 | haredDPG    | Integrated network + Adaptive objective                               |
| L2 | L2SharedDPG | L1 + Modified exploration strategy                                    |
| L3 | L3DenseNet  | L2 + Target policy smoothing + Spectral normalization                 |
| L4 | DenseNet    | L3 + Hard-swish + Dropout + Adjusting batch size and iteration number |


### Experimental Demos 

[LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/)

![LunarLanderTwinDelay3](https://github.com/Yonv1943/ElegantRL/blob/master/figs/LunarLanderTwinDelay3.gif)

[BipedalWalkerHardcore-v2](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/)

<img src="https://github.com/Yonv1943/ElegantRL/blob/master/figs/BipedalWalkerHardcore-v2-total-668kb.gif" width="150" height="100"/>

Note: BipedalWalkerHardcore is a difficult task in continuous action space. There are only a few RL implementations can reach the target reward. Check out a experiment video: [Crack the BipedalWalkerHardcore-v2 with total reward 310 using IntelAC](https://www.bilibili.com/video/BV1wi4y187tC).

## Requirements

    Necessary:
    | Python 3.6+     |           
    | PyTorch 1.6+    |    

    Not necessary:
    | Numpy 1.18+     | For ReplayBuffer. Numpy will be installed along with PyTorch.
    | gym 0.17.0      | For env. Gym provides tutorial env for DRL training. (env.render() bug in gym==1.18 pyglet==1.6. Change to gym==1.17.0, pyglet==1.5)
    | pybullet 2.7+   | For env. We use PyBullet (free) as an alternative of MuJoCo (not free).
    | box2d-py 2.3.8  | For gym. Use pip install Box2D (instead of box2d-py)
    | matplotlib 3.2  | For plots. Evaluate the agent performance.
    
    pip3 install gym==1.17.0 pybullet Box2D matplotlib


## Citation:

@inproceedings{ZhengJiaHao2021intergratedAC,
  title={Integrated Actor-Critic for Deep Reinforcement Learning},
  author={Zheng, Jiaohao and Kurt, Mehmet Necip and Wang, Xiaodong},
  booktitle={International Conference on Artificial Neural Networks},
  pages={505--518},
  year={2021},
  organization={Springer}
}

Or:

@InProceedings{10.1007/978-3-030-86380-7_41,
author="Zheng, Jiaohao
and Kurt, Mehmet Necip
and Wang, Xiaodong",
editor="Farka{\v{s}}, Igor
and Masulli, Paolo
and Otte, Sebastian
and Wermter, Stefan",
title="Integrated Actor-Critic for Deep Reinforcement Learning",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="505--518",
abstract="We propose a new deep deterministic actor-critic algorithm with an integrated network architecture and an integrated objective function. We address stabilization of the learning procedure via a novel adaptive objective that roughly ensures keeping the actor unchanged while the critic makes large errors. We reduce the number of network parameters and propose an improved exploration strategy over bounded action spaces. Moreover, we incorporate some recent advances in deep learning to our algorithm. Experiments illustrate that our algorithm speeds up the learning process and reduces the sample complexity considerably over the state-of-the-art algorithms including TD3, SAC, PPO, and A2C in continuous control tasks.",
isbn="978-3-030-86380-7"
}
