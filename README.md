## IAC: [Integrated Actor-Critic for Deep Reinforcement Learning](https://doi.org/10.1007/978-3-030-86380-7_41)

## Framework
![File_structure](https://github.com/Yonv1943/ElegantRL/blob/master/figs/File_structure.png)

An agent (in **agent.py**) uses networks (in **net.py**) and it is trained (in **run.py**) by interacting with an environment (in **env.py**).
   
A high-level overview:
1) Instantiate an environment in **Env.py** and an agent in **Agent.py** with an Actor network and a Critic network in **Net.py**, 
2) In each training step in **Run.py**, the agent interacts with the environment, generating transitions that are stored into a Replay Buffer, 
3) The agent fetches a batch of transitions from the Replay Buffer to train its networks, 
4) After each update, an evaluator evaluates the agent's performance (e.g., fitness score or cumulative return) and saves the agent if the performance is good.

## Code Structure
### Core Codes
+ **elegantrl/net.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Neural networks.
   + Q-Net,
   + Actor network,
   + Critic network, 
+ **elegantrl/agent.py**  &nbsp;&nbsp;# DRL algorithms. 
   + AgentBase, 
+ **elegantrl/run.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# run DEMO 1 ~ 4
   + Parameter initialization,
   + Training loop,
   + Evaluator.

## Training Procedure

### Initialization:
+ hyperparameters `args`.
+ `env = PreprocessEnv()` : creates an environment (in the OpenAI gym format).
+ `agent = agent.XXX()` : creates an agent for a DRL algorithm.
+ `buffer = ReplayBuffer()` : stores the transitions.
+ `evaluator = Evaluator()` : evaluates and stores the trained model.

### Training (a while loop):
+ `agent.explore_env(…)`: the agent explores the environment within target steps, generates transitions, and stores them into the ReplayBuffer.
+ `agent.update_net(…)`: the agent uses a batch from the ReplayBuffer to update the network parameters.
+ `evaluator.evaluate_save(…)`: evaluates the agent's performance and keeps the trained model with the highest score.

The while loop terminates when the conditions are met (e.g., achieving a target score or maximum steps) or manually breaks.

## Experiments

### Comparisons with Benchmark Algorithms

You can find the implementation of benchmark DRL algorithms (TD3, SAC, PPO, and A2C) in this repository.

You can change `args.agent = AgentXXX()` to run the benchmark algorithms. 
- Run the off-policy with `demo_continuous_action_off_policy()`
- Run the on-policy with `demo_continuous_action_on_policy()`

In addition, you can change `if_train_lunar_lander = 0, if_train_bipedal_walker = 0` to train different env.

```
def demo_continuous_action_off_policy():
    args = Arguments()
    args.gpu_id = sys.argv[-1][-4]

    from elegantrl2.agent import AgentSAC  # AgentDDPG AgentTD3
    args.agent = AgentSAC()
    
    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        ...
        
    if_train_bipedal_walker = 1
    if if_train_bipedal_walker:
        ...
        
    train_and_evaluate(args)     # train in single processing
    train_and_evaluate_mp(args)  # train in multiprocessing
```



### Self-comparisons


For the self-comparisons, just replace the network with an algorithm level (see Table 1 below for L1 to L4).

```
def demo_continuous_action_off_policy():
    args = Arguments()
    args.gpu_id = sys.argv[-1][-4]

    from elegantrl2.agent import AgentIAC  # L3IAC, L2IAC, L1IAC
    args.agent = AgentIAC()
    
    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        ...
        
    if_train_bipedal_walker = 1
    if if_train_bipedal_walker:
        ...
        
    train_and_evaluate(args)     # train in single processing
    train_and_evaluate_mp(args)  # train in multiprocessing
```



#### Table 1. IAC levels for self-comparisons


| Level | Agent    | Description |
| ----- | -------- | ----------------------------------------------------------------------|
| L1    | L1IAC    | Integrated network + Adaptive objective                               |
| L2    | L2IAC    | L1 + Modified exploration strategy                                    |
| L3    | L3IAC    | L2 + Target policy smoothing + Spectral normalization                 |
| L4    | AgentIAC | L3 + Hard-swish + Dropout + Adjusting batch size and iteration number |


### Demos 

[LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/)

![LunarLanderTwinDelay3](https://github.com/Yonv1943/ElegantRL/blob/master/figs/LunarLanderTwinDelay3.gif)

[BipedalWalkerHardcore-v2](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/)

<img src="https://github.com/Yonv1943/ElegantRL/blob/master/figs/BipedalWalkerHardcore-v2-total-668kb.gif" width="150" height="100"/>

Note: BipedalWalkerHardcore is a difficult task in a continuous action domain. There are only a few DRL implementations that can reach the target reward. Check out an experiment video here: [Crack the BipedalWalkerHardcore-v2 with total reward 310 using IntelAC](https://www.bilibili.com/video/BV1wi4y187tC).

## Requirements


    | Python 3.6+     |           
    | PyTorch 1.6+    |    

    | Numpy 1.18+     | For ReplayBuffer. Numpy will be installed along with PyTorch.
    | gym 0.17.0      | For env. Gym provides tutorial env for DRL training. (env.render() bug in gym==1.18 pyglet==1.6. Change to gym==1.17.0, pyglet==1.5)
    | pybullet 2.7+   | For env. We use PyBullet (free) as an alternative of MuJoCo (not free).
    | box2d-py 2.3.8  | For gym. Use pip install Box2D (instead of box2d-py)
    | matplotlib 3.2  | For plots. Evaluate the agent's performance.
    
    pip3 install gym==1.17.0 pybullet Box2D matplotlib
    

## Notes

You can find the IAC codes also in [ElegantRL (over 1k stars)](https://github.com/AI4Finance-Foundation/ElegantRL) (Jiaohao Zheng is an author of this repository).

## Citation Information

Zheng J., Kurt M.N., Wang X. (2021) Integrated Actor-Critic for Deep Reinforcement Learning. In: Farkaš I., Masulli P., Otte S., Wermter S. (eds) Artificial Neural Networks and Machine Learning – ICANN 2021. ICANN 2021. Lecture Notes in Computer Science, vol 12894. Springer, Cham. https://doi.org/10.1007/978-3-030-86380-7_41.

@inproceedings{IAC2021, <br />
  title={Integrated Actor-Critic for Deep Reinforcement Learning}, <br />
  author={Zheng, Jiaohao and Kurt, Mehmet Necip and Wang, Xiaodong}, <br />
  booktitle={International Conference on Artificial Neural Networks}, <br />
  pages={505--518}, <br />
  year={2021}, <br />
  organization={Springer} <br />
}
