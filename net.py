import torch
import torch.nn as nn
import numpy as np

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""

'''Q Network'''


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_value = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net_value(state)  # Q value


class QNetDuel(nn.Module):  # Dueling DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))  # Q value
        self.net_adv = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)  # tensor of encoded state
        q_val = self.net_val(t_tmp)
        q_adv = self.net_adv(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling Q value


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        q1 = self.net_q1(tmp)
        q2 = self.net_q2(tmp)
        return q1, q2  # two Q values


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q1 value
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q2 value
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


'''Policy Network (Actor)'''


class Actor(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=True):
        super().__init__()
        if if_use_dn:
            nn_dense = DenseNet(mid_dim // 2)
            inp_dim = nn_dense.inp_dim
            out_dim = nn_dense.out_dim
            self.net_state = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                           nn_dense, )
        else:
            out_dim = mid_dim
            self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                           nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                           nn.Linear(mid_dim, mid_dim), nn.Hardswish(), )

        self.net_a_avg = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.num_logprob = -np.log(action_dim)  # SAC will adjust alpha to let it's logprob get close to num_logprob
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.log_alpha = nn.Parameter(torch.zeros((1, action_dim)) - np.log(action_dim), requires_grad=True)

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)
        a_std = self.net_a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)
        a_std_log = self.net_a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        action = a_avg + a_std * noise
        a_tan = action.tanh()
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute logprob according to mean and std of action (stochastic policy)'''
        logprob = -(a_std_log + self.log_sqrt_2pi + ((a_avg - action) / a_std).pow(2) * 0.5)
        # same as below:
        # from torch.distributions.normal import Normal
        # logprob = Normal(a_avg, a_std).log_prob(a_noise)

        logprob = logprob - (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        # same as below:
        # logprob = logprob - ((1 - a_noise_tanh.pow(2)) + 1e-6).log()
        return a_tan, logprob.sum(1, keepdim=True)

    def get_obj_alpha(self, logprob):
        return -(self.log_alpha * (logprob - self.num_logprob).detach()).mean()

    def get_a_std_log(self, t_tmp):
        a_std_log = self.net_a_std(t_tmp)
        return a_std_log.clamp(self.min_std_log, self.max_std_log)


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std.expand_as(a_avg)
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        a_prob = self.soft_max(self.net(state))
        # dist = Categorical(a_prob)
        # action = dist.sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state.size(0))
        return action, a_prob

    def get_new_logprob_entropy(self, state, action):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        a_int = action.squeeze(1).long()
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, action, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(action.long().squeeze(1))


'''Value Network (Critic)'''


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_dense = DenseNet(mid_dim // 2)
            inp_dim = nn_dense.inp_dim
            out_dim = nn_dense.out_dim

            self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                     nn_dense,
                                     nn.Linear(out_dim, 1), )
        else:
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1), )
        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        return self.net(state)  # Q value


class CriticTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        if if_use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense = DenseNet(mid_dim // 2)
            inp_dim = nn_dense.inp_dim
            out_dim = nn_dense.out_dim

            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, inp_dim), nn.ReLU(),
                                        nn_dense, )  # state-action value function
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU())
            out_dim = mid_dim

        self.net_q1 = nn.Linear(out_dim, 1)
        self.net_q2 = nn.Linear(out_dim, 1)
        layer_norm(self.net_q1, std=0.1)
        layer_norm(self.net_q2, std=0.1)

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


'''Parameter sharing Network'''


class SharedDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        nn_dense = DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))

        self.net = nn_dense

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh())
        self.dec_q = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)))

    @staticmethod
    def add_noise(a, noise_std):
        a_temp = torch.normal(a, noise_std)
        mask = torch.tensor((a_temp < -1.0) + (a_temp > 1.0), dtype=torch.float32).cuda()

        noise_uniform = torch.rand_like(a)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q = self.dec_q(q_)
        return q

    def next_q_action(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        '''q_target (without noise)'''
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        '''q_target (with noise)'''
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a


class L1SharedDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        nn_dense = L3DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))

        self.net = nn_dense

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh())
        self.dec_q = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)))

    @staticmethod
    def add_noise(a, noise_std):
        a_temp = torch.normal(a, noise_std)
        mask = torch.tensor((a_temp < -1.0) + (a_temp > 1.0), dtype=torch.float32).cuda()

        noise_uniform = torch.rand_like(a)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q = self.dec_q(q_)
        return q

    def next_q_action(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        '''q_target (without noise)'''
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        '''q_target (with noise)'''
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a


class L2SharedDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        nn_dense = L3DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))

        self.net = nn_dense

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh())
        self.dec_q = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, 1), )

    @staticmethod
    def add_noise(a, noise_std):
        a_temp = torch.normal(a, noise_std)
        mask = torch.tensor((a_temp < -1.0) + (a_temp > 1.0), dtype=torch.float32).cuda()

        noise_uniform = torch.rand_like(a)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q = self.dec_q(q_)
        return q

    def next_q_action(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        '''q_target (without noise)'''
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        '''q_target (with noise)'''
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a

    
"""utils"""


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)


class L3DenseNet(nn.Module):
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.ReLU())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.ReLU())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)


class ConcatNet(nn.Module):  # concatenate
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense2 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense3 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense4 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x0):
        x1 = self.dense1(x0)
        x2 = self.dense2(x0)
        x3 = self.dense3(x0)
        x4 = self.dense4(x0)

        return torch.cat((x1, x2, x3, x4), dim=1)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
