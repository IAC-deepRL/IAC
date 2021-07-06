import sys

import gym
from elegantrl2.env import PreprocessEnv, PreprocessVecEnv
from elegantrl2.run import Arguments, train_and_evaluate, train_and_evaluate_mp

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


def demo_discrete_action_off_policy():
    args = Arguments()
    from elegantrl2.agent import AgentDoubleDQN  # AgentDQN
    args.agent = AgentDoubleDQN()

    '''choose environment'''
    if_train_cart_pole = 0
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.net_dim = 2 ** 7
        args.target_step = args.env.max_step * 2

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 2e5, TargetReturn: 200, UsedTime: 400s, LunarLander-v2, PPO"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim

    '''train and evaluate'''
    train_and_evaluate(args)


def demo_discrete_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    from elegantrl2.agent import AgentDiscretePPO
    args.agent = AgentDiscretePPO()

    '''choose environment'''
    if_train_cart_pole = 0
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 4
        args.target_step = args.env.max_step * 8
        args.if_per_or_gae = True

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 2e5, TargetReturn: 200, UsedTime: 400s, LunarLander-v2, PPO"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.agent.cri_target = False
        args.reward_scale = 2 ** -1
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 4
        args.target_step = args.env.max_step * 4
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    '''train and evaluate'''
    train_and_evaluate(args)


def demo_continuous_action_off_policy():
    args = Arguments()
    args.gpu_id = sys.argv[-1][-4]

    from elegantrl2.agent import AgentSAC  # AgentDDPG AgentTD3
    args.agent = AgentSAC()

    '''choose environment'''
    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        env = gym.make('Pendulum-v0')
        env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim
        args.target_step = args.env.max_step * 4

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334
        args.gamma = 0.97
        args.if_per_or_gae = True

    '''train and evaluate'''
    train_and_evaluate(args)


def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    from elegantrl2.agent import AgentPPO
    args.agent = AgentPPO()
    args.gpu_id = sys.argv[-1][-4]
    args.agent.cri_target = True
    args.learning_rate = 2 ** -14
    args.random_seed = 1943

    '''choose environment'''
    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        env = gym.make('Pendulum-v0')
        env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReward: 200, UsedTime: 800s"
        env_name = 'LunarLanderContinuous-v2'
        # args.env = PreprocessEnv(env=env_name)
        args.env = PreprocessVecEnv(env=env_name, env_num=2)
        args.env_eval = PreprocessEnv(env=env_name)
        args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302
        args.break_step = int(8e6)
        args.if_per_or_gae = True
        args.target_step = args.env.max_step * 8
        args.repeat_times = 2 ** 4

    if_train_bipedal_walker = 1
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        env_name = 'BipedalWalker-v3'
        # args.env = PreprocessEnv(env=env_name)
        args.env = PreprocessVecEnv(env=env_name, env_num=2)
        args.env_eval = PreprocessEnv(env=env_name)
        args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334
        args.gamma = 0.97
        args.target_step = args.env.max_step * 8
        args.repeat_times = 2 ** 4
        args.if_per_or_gae = True
        args.agent.lambda_entropy = 0.05
        args.break_step = int(8e6)

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.worker_num = 2
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    # demo_continuous_action_off_policy()
    demo_continuous_action_on_policy()
