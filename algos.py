import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import os
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from logger import logger
import copy


class Actor(nn.Module):
    """Actor used in BCQ"""

    def __init__(
        self, state_dim, action_dim, max_action, hidden_dim=[400, 300], threshold=0.05
    ):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)

        self.max_action = max_action
        self.threshold = threshold

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.threshold * self.max_action * torch.tanh(self.l3(a))
        # return (a + action).clamp(-self.max_action, self.max_action)
        return torch.max(torch.min(self.max_action, a + action), -self.max_action)


class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorTD3, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x, preval=False):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        pre_tanh_val = x
        x = self.max_action * torch.tanh(self.l3(x))
        if not preval:
            return x
        return x, pre_tanh_val


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
    ):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.FloatTensor(
            np.random.normal(0, 1, size=(std_a.size()))
        ).to(device)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) + std_a.unsqueeze(1) * torch.FloatTensor(
            np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))
        ).to(device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
        return log_pis


class Critic(nn.Module):
    """Regular critic used in off-policy RL"""

    def __init__(self, state_dim, action_dim, hidden_dim=[400, 300]):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


from torch.optim.lr_scheduler import CosineAnnealingLR


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)  # uses uniform


def mlp(
    dims,
    activation=nn.ReLU,
    output_activation=None,
    squeeze_output=False,
    dropout=0,
    layer_norm=False,
):
    n_dims = len(dims)
    assert n_dims >= 2, "MLP requires at least two dims (input and output)"

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
    layers.append(nn.Linear(dims[-2], dims[-1]))

    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2, dropout=0):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True, dropout=dropout)
        self.apply(weight_init)

    def forward(self, state):
        return self.v(state)


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, dropout=0):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True, dropout=dropout)
        self.q2 = mlp(dims, squeeze_output=True, dropout=dropout)
        self.apply(weight_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa).reshape(-1, 1), self.q2(sa).reshape(-1, 1)


class DeterministicPolicy(nn.Module):
    def __init__(
        self, obs_dim, act_dim, max_action, activation, hidden_dim=256, n_hidden=2
    ):
        super().__init__()

        self.activation = activation
        if activation == "relu":
            self.latent = mlp(
                [obs_dim, *([hidden_dim] * n_hidden), hidden_dim],
                output_activation=nn.ReLU,
            )
            self.out_layer = mlp(
                [hidden_dim, *([hidden_dim] * 1), act_dim], output_activation=None
            )
        elif activation == "tanh":
            self.net = mlp(
                [obs_dim, *([hidden_dim] * n_hidden), act_dim],
                output_activation=nn.Tanh,
            )

        self.max_action = max_action
        self.apply(weight_init)

    def forward(self, obs):
        if self.activation == "tanh":
            return torch.max(
                torch.min(self.max_action, self.net(obs)), -self.max_action
            )
        elif self.activation == "relu":
            return torch.max(
                torch.min(self.max_action, self.out_layer(self.latent(obs))),
                -self.max_action,
            )

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)


from torch.distributions import MultivariateNormal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(
        self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, dropout=0, layer_norm=False
    ):
        super().__init__()
        if isinstance(hidden_dim, list):
            self.net = mlp(
                [obs_dim, *(hidden_dim * n_hidden), act_dim],
                dropout=dropout,
                layer_norm=layer_norm,
            )
        else:
            self.net = mlp(
                [obs_dim, *([hidden_dim] * n_hidden), act_dim],
                dropout=dropout,
                layer_norm=layer_norm,
            )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.apply(weight_init)

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


def norm_penalty(loss, net):
    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    grad_norm = torch.sqrt(grad_norm + 1e-12)
    loss = loss + 10 * grad_norm
    return loss


# IQL
import types


def unhook(model):
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
        model.modules(),
    )
    for layer in prunable_layers:
        layer.weight._backward_hooks = OrderedDict()


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def monkey_patch(model, mask_layers):
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
        model.modules(),
    )
    count_mask_layers = len(mask_layers)
    for i, (layer, mask) in enumerate(zip(prunable_layers, mask_layers)):
        layer.weight_mask = mask
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)
    assert (
        count_mask_layers == i + 1
    )  # this checks if all the mask layers are being used


# -------------------------------------------


def apply_prune_mask(net, keep_masks, fixed_weight=0.0):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
        net.modules(),
    )

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert layer.weight.shape == keep_mask.shape

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        if fixed_weight == -1:
            pass
        else:
            layer.weight.data[keep_mask == 0.0] = 0.0
        layer.weight.register_hook(
            hook_factory(keep_mask)
        )  # register hook is backward hook


class IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=256,
        activation="tanh",
        discount=0.99,
        batch_size=256,
        actor_grad_norm_lambda=0,
        max_steps=1000000,
        policy_type="gaussian",
        clip_grad=None,
        lr=0.001,
        beta=3.0,
        tau=0.7,
        dropout=0,
        L1_coeff=0,
        L2_coeff=0,
    ):
        latent_dim = action_dim * 2
        self.policy_type = policy_type
        if policy_type == "gaussian":
            self.actor = GaussianPolicy(
                state_dim, action_dim, hidden_dim, dropout=dropout
            ).to(device)

        elif policy_type == "deterministic":
            self.actor = DeterministicPolicy(
                state_dim,
                action_dim,
                max_action,
                activation=activation,
                hidden_dim=hidden_dim,
            ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = TwinQ(
            state_dim, action_dim, hidden_dim=hidden_dim, dropout=dropout
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.value = ValueFunction(state_dim, hidden_dim, dropout=dropout).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.max_action = max_action
        self.action_dim = action_dim
        self.policy_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.discount = discount
        self.beta = beta
        self.EXP_ADV_MAX = 100.0
        self.alpha = 0.005
        self.tau = tau
        self.total_it = 0
        self.actor_grad_norm_lambda = actor_grad_norm_lambda
        self.clip_grad = clip_grad
        self.L1_coeff = L1_coeff
        self.L2_coeff = L2_coeff

    def init_optim(self, lr=None):
        if lr is None:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
            self.value_optimizer = torch.optim.Adam(self.value.parameters())
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    def init_target(self):
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)

    def policy_loss_(self, state, perturbed_actions, y=None):
        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()
        return actor_loss

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        if self.policy_type == "deterministic":
            return self.actor(obs).cpu().data.numpy().flatten()
        else:
            dist = self.actor(obs)
            return (
                dist.mean.cpu().detach().numpy().flatten()
                if deterministic
                else dist.sample().cpu().detach().numpy().flatten()
            )

    def update_exponential_moving_average(self, target, source, alpha):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            # if source.grad is not None:
            target_param.data.mul_(1.0 - alpha).add_(source_param.data, alpha=alpha)

    def asymmetric_l2_loss(self, u, tau):
        return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

    def compute_grad_norm(self, net):
        grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            grad_norm.append(p.grad.flatten())
        grad_norm = torch.cat(grad_norm).norm(2).sum() + 1e-12
        return grad_norm

    def compute_normed_loss(self, loss, net):
        ac_grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            ac_grad_norm.append(p.grad.flatten())
        ac_grad_norm = torch.cat(ac_grad_norm).norm(2).sum() + 1e-12
        loss = loss + self.actor_grad_norm_lambda * ac_grad_norm
        return loss

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        using_snip=False,
    ):
        for it in range(iterations):
            self.total_it += 1
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))
                next_v = self.value(next_state).reshape(-1, 1)

            # Update value function
            v = self.value(state)
            adv = target_q - v
            v_loss = self.asymmetric_l2_loss(adv, self.tau)
            self.value_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.value_optimizer.step()

            # Update Q function
            true_Q = reward + done * discount * next_v.detach()
            current_Q1, current_Q2 = self.critic(state, action)
            q_loss = (
                F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q)
            ) / 2

            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            self.critic_optimizer.step()

            # Update policy
            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))  # default
                v = self.value(state)
                adv = target_q - v

            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.EXP_ADV_MAX)
            policy_out = self.actor(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(action)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError
            actor_loss = torch.mean(exp_adv * bc_losses)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.policy_lr_schedule.step()

            self.update_exponential_moving_average(
                self.critic_target, self.critic, self.alpha
            )

        logger.record_tabular("Train/Actor Loss", actor_loss.cpu().data.numpy())
        logger.record_tabular("Train/BC Loss", bc_losses.mean().cpu().data.numpy())
        logger.record_tabular("Train/Action-Value Loss", q_loss.cpu().data.numpy())
        logger.record_tabular("Train/Value Loss", v_loss.cpu().data.numpy())
        logger.record_tabular(
            "Train/True Actor Loss",
            F.mse_loss(policy_out.mean, action).cpu().data.numpy(),
        )

        logger.record_tabular("Train/Target_q", target_q.mean().cpu().data.numpy())
        logger.record_tabular("Train/reward", reward.mean().cpu().data.numpy())
        logger.record_tabular("Train/adv", adv.mean().cpu().data.numpy())

    def mask_agent(self, keep_masks):
        unhook(self.actor)
        unhook(self.critic)
        unhook(self.value)
        monkey_patch(self.actor, keep_masks["actor"])
        monkey_patch(self.critic, keep_masks["critic"])
        monkey_patch(self.critic_target, keep_masks["critic"])
        monkey_patch(self.value, keep_masks["value"])
        apply_prune_mask(self.actor, keep_masks["actor"], fixed_weight=-1)
        apply_prune_mask(self.critic, keep_masks["critic"], fixed_weight=-1)
        apply_prune_mask(self.value, keep_masks["value"], fixed_weight=-1)

    def train_no_grad_update(
        self, replay_buffer, iterations, batch_size=256, discount=0.99
    ):
        for it in range(iterations):
            self.total_it += 1
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))
                next_v = self.value(next_state).reshape(-1, 1)

            # Update value function
            v = self.value(state)
            adv = target_q - v
            v_loss = self.asymmetric_l2_loss(adv, self.tau)
            self.value_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()

            # Update Q function
            true_Q = reward + done * discount * next_v.detach()
            current_Q1, current_Q2 = self.critic(state, action)
            q_loss = (
                F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q)
            ) / 2
            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()

            # Update policy
            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))  # default
                v = self.value(state)
                adv = target_q - v

            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.EXP_ADV_MAX)
            policy_out = self.actor(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(action)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError
            actor_loss = torch.mean(exp_adv * bc_losses)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()

        logger.record_tabular("Train/Actor Loss", actor_loss.cpu().data.numpy())
        logger.record_tabular("Train/BC Loss", bc_losses.mean().cpu().data.numpy())
        logger.record_tabular("Train/Action-Value Loss", q_loss.cpu().data.numpy())
        logger.record_tabular("Train/Value Loss", v_loss.cpu().data.numpy())
        logger.record_tabular(
            "Train/True Actor Loss",
            F.mse_loss(policy_out.mean, action).cpu().data.numpy(),
        )
        logger.record_tabular("Train/Target_q", target_q.mean().cpu().data.numpy())
        logger.record_tabular("Train/reward", reward.mean().cpu().data.numpy())
        logger.record_tabular("Train/adv", adv.mean().cpu().data.numpy())

    def evaluate_validation_performance(self, replay_buffer):
        true_critic_loss = 0
        true_MC_loss = 0
        critic_loss = 0
        true_actor_loss = 0
        actor_loss = 0
        TD_error = 0

        idx = 0
        valid_sample_size = replay_buffer.storage["valid_state"].shape[0]
        sample_size = 10000
        itr = int(valid_sample_size / sample_size)
        for i in range(itr):
            # state, action, next_state, reward, not_done, true_Q, next_action = replay_buffer.validation_samples()
            # state, action, true_Q = replay_buffer.validation_samples(idx=idx, sample_size=sample_size)
            (
                state,
                action,
                next_state,
                next_action,
                rewards,
                not_done,
                true_Q,
                true_MC,
            ) = replay_buffer.validation_samples(idx=idx, sample_size=sample_size)
            idx += sample_size

            # check Q estimate on validation set
            Q_estimate = torch.min(*self.critic(state, action))
            true_critic_loss += F.mse_loss(Q_estimate, true_Q).cpu().data.numpy()

            true_MC_loss += F.mse_loss(Q_estimate, true_MC).cpu().data.numpy()

            # common critic loss
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss += (
                (F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q))
                .cpu()
                .data.numpy()
            )

            # TD -error
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # float64; should be float32
            target_Q = rewards + not_done * self.discount * target_Q
            TD_error += (
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
                .cpu()
                .data.numpy()
            )

            pi = self.actor(state).mean
            true_actor_loss += F.mse_loss(pi, action).cpu().data.numpy()

            # common actor loss
            Q = torch.min(*self.critic(state, pi))
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss += -lmbda * Q.mean() + F.mse_loss(pi, action)

            # value loss
            v = self.value(state)
            adv = target_Q - v
            v_loss = self.asymmetric_l2_loss(adv, self.tau).cpu().data.numpy()

        logger.record_tabular("Valid/True Critic Loss", true_critic_loss / itr)
        logger.record_tabular("Valid/True MC Loss", true_MC_loss / itr)
        logger.record_tabular("Valid/True Actor Loss", true_actor_loss / itr)
        logger.record_tabular("Valid/Critic Loss", critic_loss / itr)
        logger.record_tabular("Valid/TD Error", TD_error / itr)
        logger.record_tabular("Valid/Actor Loss", actor_loss.cpu().data.numpy() / itr)
        logger.record_tabular("Valid/V Loss", v_loss / itr)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        # torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))
        torch.save(self.value.state_dict(), "%s/%s_value.pth" % (directory, filename))

        # optimizer
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.value_optimizer.state_dict(), filename + "_value_optimizer")

    def get_net_weight_size(self, filename, directory):
        return (
            os.path.getsize("%s/%s_actor.pth" % (directory, filename)),
            os.path.getsize("%s/%s_critic.pth" % (directory, filename)),
            os.path.getsize("%s/%s_value.pth" % (directory, filename)),
        )

    def sparse_weights(self, directory, filename):

        def compress(model):
            res = OrderedDict()
            for name, param in model.named_parameters():
                res[name] = param.to_sparse()
            return res

        torch.save(compress(self.actor), "%s/%s_actor.pth" % (directory, filename))
        torch.save(compress(self.critic), "%s/%s_critic.pth" % (directory, filename))
        torch.save(compress(self.value), "%s/%s_value.pth" % (directory, filename))

    def remove_weights(self, filename, directory):
        os.remove("%s/%s_actor.pth" % (directory, filename))
        os.remove("%s/%s_critic.pth" % (directory, filename))
        os.remove("%s/%s_value.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.value.load_state_dict(
            torch.load("%s/%s_value.pth" % (directory, filename))
        )

        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))


# --------------- AWAC ------------------


class AWAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=256,
        activation="tanh",
        discount=0.99,
        batch_size=256,
        actor_grad_norm_lambda=0,
        max_steps=1000000,
        policy_type="gaussian",
        clip_grad=None,
        lr=0.001,
        tau=5e-3,
        awac_lambda=0.3333,
        exp_adv_max=100,
        dropout=0,
        L1_coeff=0,
        L2_coeff=0,
    ):
        latent_dim = action_dim * 2
        self.policy_type = policy_type
        if policy_type == "gaussian":
            self.actor = GaussianPolicy(
                state_dim, action_dim, hidden_dim, dropout=dropout
            ).to(device)

        elif policy_type == "deterministic":
            self.actor = DeterministicPolicy(
                state_dim,
                action_dim,
                max_action,
                activation=activation,
                hidden_dim=hidden_dim,
            ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinQ(
            state_dim, action_dim, hidden_dim=hidden_dim, dropout=dropout
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.value = ValueFunction(state_dim, hidden_dim, dropout=dropout).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        #

        self.policy_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.discount = discount
        self.awac_lambda = awac_lambda  # 3.0
        self.exp_adv_max = exp_adv_max
        self.alpha = 0.005
        self.tau = tau
        self.total_it = 0
        # self.batch_size = batch_size

        #
        self.actor_grad_norm_lambda = actor_grad_norm_lambda
        self.clip_grad = clip_grad
        # regs
        self.L1_coeff = L1_coeff
        self.L2_coeff = L2_coeff

    def init_optim(self, lr=None):
        if lr is None:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
            self.value_optimizer = torch.optim.Adam(self.value.parameters())
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    def init_target(self):
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)

    def policy_loss_(self, state, perturbed_actions, y=None):
        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()
        return actor_loss

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        if self.policy_type == "deterministic":
            return self.actor(obs).cpu().data.numpy().flatten()
        else:
            dist = self.actor(obs)
            return (
                dist.mean.cpu().detach().numpy().flatten()
                if deterministic
                else dist.sample().cpu().detach().numpy().flatten()
            )

    def update_exponential_moving_average(self, target, source, alpha):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            # if source.grad is not None:
            target_param.data.mul_(1.0 - alpha).add_(source_param.data, alpha=alpha)

    def asymmetric_l2_loss(self, u, tau):
        return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

    def compute_grad_norm(self, net):
        grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            grad_norm.append(p.grad.flatten())
        grad_norm = torch.cat(grad_norm).norm(2).sum() + 1e-12
        return grad_norm

    def compute_normed_loss(self, loss, net):
        ac_grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            ac_grad_norm.append(p.grad.flatten())
        ac_grad_norm = torch.cat(ac_grad_norm).norm(2).sum() + 1e-12
        loss = loss + self.actor_grad_norm_lambda * ac_grad_norm
        return loss

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        using_snip=False,
    ):
        for it in range(iterations):
            self.total_it += 1
            # print ('Iteration : ', it)
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))
            with torch.no_grad():
                next_pi_action = self.actor(next_state).sample()
                q_next = torch.min(
                    *self.critic_target(next_state, next_pi_action)
                ).reshape(-1, 1)

            # Update Q function
            true_Q = reward + done * discount * q_next.detach()
            current_Q1, current_Q2 = self.critic(state, action)
            q_loss = (
                F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q)
            ) / 2
            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            self.critic_optimizer.step()

            # Update policy
            with torch.no_grad():
                pi_action = self.actor(state).sample()
                v = torch.min(*self.critic(state, pi_action))
                q = torch.min(*self.critic(state, action))
                adv = q - v
                weights = torch.clamp_max(
                    torch.exp(adv / self.awac_lambda), self.exp_adv_max
                )

            policy_out = self.actor(state)
            bc_losses = -policy_out.log_prob(action)
            actor_loss = torch.mean(weights * bc_losses)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.policy_lr_schedule.step()
            self.update_exponential_moving_average(
                self.critic_target, self.critic, self.alpha
            )

        logger.record_tabular("Train/Actor Loss", actor_loss.cpu().data.numpy())
        logger.record_tabular("Train/BC Loss", bc_losses.mean().cpu().data.numpy())
        logger.record_tabular("Train/Action-Value Loss", q_loss.cpu().data.numpy())
        logger.record_tabular(
            "Train/True Actor Loss",
            F.mse_loss(policy_out.mean, action).cpu().data.numpy(),
        )

        logger.record_tabular("Train/Target_q", target_q.mean().cpu().data.numpy())
        logger.record_tabular("Train/reward", reward.mean().cpu().data.numpy())
        logger.record_tabular("Train/adv", adv.mean().cpu().data.numpy())

    def mask_agent(self, keep_masks):
        unhook(self.actor)
        unhook(self.critic)
        monkey_patch(self.actor, keep_masks["actor"])
        monkey_patch(self.critic, keep_masks["critic"])
        monkey_patch(self.critic_target, keep_masks["critic"])
        apply_prune_mask(self.actor, keep_masks["actor"], fixed_weight=-1)
        apply_prune_mask(self.critic, keep_masks["critic"], fixed_weight=-1)

    def train_no_grad_update(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        using_snip=False,
    ):
        for it in range(iterations):
            self.total_it += 1
            # print ('Iteration : ', it)
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))

            # --- critic loss ---

            with torch.no_grad():
                next_pi_action = self.actor(next_state).sample()
                q_next = torch.min(
                    *self.critic_target(next_state, next_pi_action)
                ).reshape(-1, 1)

            # Update Q function
            true_Q = reward + done * discount * q_next.detach()
            current_Q1, current_Q2 = self.critic(state, action)
            q_loss = (
                F.mse_loss(current_Q1, true_Q.flatten())
                + F.mse_loss(current_Q2, true_Q.flatten())
            ) / 2
            q_loss.backward()

            # ------ Update policy --------
            with torch.no_grad():
                pi_action = self.actor(state).sample()
                v = torch.min(*self.critic(state, pi_action))
                q = torch.min(*self.critic(state, action))
                adv = q - v
                weights = torch.clamp_max(
                    torch.exp(adv / self.awac_lambda), self.exp_adv_max
                )

            policy_out = self.actor(state)
            bc_losses = -policy_out.log_prob(action)

            actor_loss = torch.mean(weights * bc_losses)
            actor_loss.backward()

        logger.record_tabular("Train/Actor Loss", actor_loss.cpu().data.numpy())
        logger.record_tabular("Train/BC Loss", bc_losses.mean().cpu().data.numpy())
        logger.record_tabular("Train/Action-Value Loss", q_loss.cpu().data.numpy())
        logger.record_tabular(
            "Train/True Actor Loss",
            F.mse_loss(policy_out.mean, action).cpu().data.numpy(),
        )
        logger.record_tabular("Train/Target_q", target_q.mean().cpu().data.numpy())
        logger.record_tabular("Train/reward", reward.mean().cpu().data.numpy())
        logger.record_tabular("Train/adv", adv.mean().cpu().data.numpy())

    def evaluate_validation_performance(self, replay_buffer):
        true_critic_loss = 0
        true_MC_loss = 0
        critic_loss = 0
        true_actor_loss = 0
        actor_loss = 0
        TD_error = 0

        idx = 0
        valid_sample_size = replay_buffer.storage["valid_state"].shape[0]
        sample_size = 10000
        itr = int(valid_sample_size / sample_size)
        for i in range(itr):
            # state, action, next_state, reward, not_done, true_Q, next_action = replay_buffer.validation_samples()
            # state, action, true_Q = replay_buffer.validation_samples(idx=idx, sample_size=sample_size)
            (
                state,
                action,
                next_state,
                next_action,
                rewards,
                not_done,
                true_Q,
                true_MC,
            ) = replay_buffer.validation_samples(idx=idx, sample_size=sample_size)
            idx += sample_size

            # check Q estimate on validation set
            Q_estimate = torch.min(*self.critic(state, action))
            true_critic_loss += F.mse_loss(Q_estimate, true_Q).cpu().data.numpy()
            true_MC_loss += F.mse_loss(Q_estimate, true_MC).cpu().data.numpy()

            # common critic loss
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss += (
                (F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q))
                .cpu()
                .data.numpy()
            )

            # TD -error
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # float64; should be float32
            target_Q = rewards + not_done * self.discount * target_Q
            TD_error += (
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
                .cpu()
                .data.numpy()
            )

            pi = self.actor(state).mean
            true_actor_loss += F.mse_loss(pi, action).cpu().data.numpy()

            # common actor loss
            Q = torch.min(*self.critic(state, pi))
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss += -lmbda * Q.mean() + F.mse_loss(pi, action)

        logger.record_tabular("Valid/True Critic Loss", true_critic_loss / itr)
        logger.record_tabular("Valid/True MC Loss", true_MC_loss / itr)
        logger.record_tabular("Valid/True Actor Loss", true_actor_loss / itr)
        logger.record_tabular("Valid/Critic Loss", critic_loss / itr)
        logger.record_tabular("Valid/TD Error", TD_error / itr)
        logger.record_tabular("Valid/Actor Loss", actor_loss.cpu().data.numpy() / itr)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(self.value.state_dict(), "%s/%s_value.pth" % (directory, filename))

    def get_net_weight_size(self, filename, directory):
        return (
            os.path.getsize("%s/%s_actor.pth" % (directory, filename)),
            os.path.getsize("%s/%s_critic.pth" % (directory, filename)),
            os.path.getsize("%s/%s_value.pth" % (directory, filename)),
        )

    def sparse_weights(self, directory, filename):

        def compress(model):
            res = OrderedDict()
            for name, param in model.named_parameters():
                res[name] = param.to_sparse()
            return res

        torch.save(compress(self.actor), "%s/%s_actor.pth" % (directory, filename))
        torch.save(compress(self.critic), "%s/%s_critic.pth" % (directory, filename))
        torch.save(compress(self.value), "%s/%s_value.pth" % (directory, filename))

    def remove_weights(self, filename, directory):
        os.remove("%s/%s_actor.pth" % (directory, filename))
        os.remove("%s/%s_critic.pth" % (directory, filename))
        os.remove("%s/%s_value.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.value.load_state_dict(
            torch.load("%s/%s_value.pth" % (directory, filename))
        )


class TD3Actor(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim, max_action, activation="tanh"
    ):
        super(TD3Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.activation = activation

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        if self.activation == "tanh":
            return torch.max(
                torch.min(self.max_action, torch.tanh(self.l3(a))), -self.max_action
            )
        elif self.activation == "relu":
            return torch.max(torch.min(self.max_action, self.l3(a)), -self.max_action)


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(TD3Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        activation="tanh",
        hidden_dim=256,
        lr=0.001,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        self.discounts = 0.99

        self.actor = TD3Actor(
            state_dim, action_dim, hidden_dim, max_action, activation
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TD3Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

    def mask_agent(self, keep_masks):
        unhook(self.actor)
        unhook(self.critic)
        monkey_patch(self.actor, keep_masks["actor"])
        monkey_patch(self.actor_target, keep_masks["actor"])
        monkey_patch(self.critic, keep_masks["critic"])
        monkey_patch(self.critic_target, keep_masks["critic"])
        apply_prune_mask(self.actor, keep_masks["actor"], fixed_weight=-1)
        apply_prune_mask(self.critic, keep_masks["critic"], fixed_weight=-1)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        state = state.unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        using_snip=False,
    ):
        for it in range(iterations):
            self.total_it += 1
            # Sample replay buffer
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = torch.max(
                    torch.min(self.max_action, (self.actor_target(next_state) + noise)),
                    -self.max_action,
                )
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                # float64; should be float32
                target_Q = reward + done * self.discount * target_Q
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha / Q.abs().mean().detach()
                actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

    def train_no_grad_update(self, replay_buffer, iterations, batch_size=256):

        for it in range(iterations):
            self.total_it += 1

            # Sample replay buffer
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )

                # next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                next_action = torch.max(
                    torch.min(self.max_action, (self.actor_target(next_state) + noise)),
                    -self.max_action,
                )

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                # float64; should be float32
                target_Q = reward + done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

    def evaluate_validation_performance(self, replay_buffer):
        true_critic_loss = 0
        true_MC_loss = 0
        critic_loss = 0
        true_actor_loss = 0
        actor_loss = 0
        TD_error = 0

        idx = 0
        valid_sample_size = replay_buffer.storage["valid_state"].shape[0]
        sample_size = 10000
        itr = int(valid_sample_size / sample_size)
        for i in range(itr):
            (
                state,
                action,
                next_state,
                next_action,
                rewards,
                not_done,
                true_Q,
                true_MC,
            ) = replay_buffer.validation_samples(idx=idx, sample_size=sample_size)
            idx += sample_size

            # check Q estimate on validation set
            Q_estimate = self.critic.Q1(state, action)
            true_critic_loss += F.mse_loss(Q_estimate, true_Q).cpu().data.numpy()

            true_MC_loss += F.mse_loss(Q_estimate, true_MC).cpu().data.numpy()

            # common critic loss
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss += (
                (F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q))
                .cpu()
                .data.numpy()
            )

            # TD -error
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # float64; should be float32
            target_Q = rewards + not_done * self.discount * target_Q
            TD_error += (
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
                .cpu()
                .data.numpy()
            )
            pi = self.actor(state)
            true_actor_loss += F.mse_loss(pi, action).cpu().data.numpy()

            # common actor loss
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss += -lmbda * Q.mean() + F.mse_loss(pi, action)

        logger.record_tabular("Valid/True Critic Loss", true_critic_loss / itr)
        logger.record_tabular("Valid/True MC Loss", true_MC_loss / itr)
        logger.record_tabular("Valid/True Actor Loss", true_actor_loss / itr)
        logger.record_tabular("Valid/Critic Loss", critic_loss / itr)
        logger.record_tabular("Valid/TD Error", TD_error / itr)
        logger.record_tabular("Valid/Actor Loss", actor_loss.cpu().data.numpy() / itr)


class BCActor(nn.Module):
    def __init__(
        self, obs_dim, act_dim, max_action, hidden_dim=256, n_hidden=2, dropout=0
    ):
        super(BCActor, self).__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], dropout=dropout)
        self.max_action = max_action

    def forward(self, obs):
        a = self.net(obs)
        return self.max_action * torch.tanh(a)


class BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=256,
        activation="tanh",
        discount=0.99,
        batch_size=256,
        actor_grad_norm_lambda=0,
        max_steps=1000000,
        policy_type="gaussian",
        clip_grad=None,
        lr=0.001,
        beta=3.0,
        tau=0.7,
        dropout=0,
        L1_coeff=0,
        L2_coeff=0,
        layer_norm=False,
    ):
        latent_dim = action_dim * 2
        self.policy_type = policy_type
        if policy_type == "gaussian":
            self.actor = GaussianPolicy(
                state_dim,
                action_dim,
                hidden_dim,
                dropout=dropout,
                layer_norm=layer_norm,
            ).to(device)
        elif policy_type == "deterministic":
            self.actor = DeterministicPolicy(
                state_dim,
                action_dim,
                max_action,
                activation=activation,
                hidden_dim=hidden_dim,
            ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.policy_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.discount = discount
        self.beta = beta
        self.EXP_ADV_MAX = 100.0
        self.alpha = 0.005
        self.tau = tau
        self.total_it = 0
        self.actor_grad_norm_lambda = actor_grad_norm_lambda
        self.clip_grad = clip_grad
        self.L1_coeff = L1_coeff
        self.L2_coeff = L2_coeff
        self.layer_norm = layer_norm

    def init_optim(self, lr=None):
        if lr is None:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def init_target(self):
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)

    def policy_loss_(self, state, perturbed_actions, y=None):
        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()
        return actor_loss

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        if self.policy_type == "deterministic":
            return self.actor(obs).cpu().data.numpy().flatten()
        else:
            dist = self.actor(obs)
            return (
                dist.mean.cpu().detach().numpy().flatten()
                if deterministic
                else dist.sample().cpu().detach().numpy().flatten()
            )

    def update_exponential_moving_average(self, target, source, alpha):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            # if source.grad is not None:
            target_param.data.mul_(1.0 - alpha).add_(source_param.data, alpha=alpha)

    def asymmetric_l2_loss(self, u, tau):
        return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

    def compute_grad_norm(self, net):
        grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            grad_norm.append(p.grad.flatten())
        grad_norm = torch.cat(grad_norm).norm(2).sum() + 1e-12
        return grad_norm

    def compute_normed_loss(self, loss, net):
        ac_grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            ac_grad_norm.append(p.grad.flatten())
        ac_grad_norm = torch.cat(ac_grad_norm).norm(2).sum() + 1e-12
        loss = loss + self.actor_grad_norm_lambda * ac_grad_norm
        return loss

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        using_snip=False,
    ):
        for it in range(iterations):
            self.total_it += 1
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            policy_out = self.actor(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(action)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError
            actor_loss = torch.mean(bc_losses)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.policy_lr_schedule.step()

        logger.record_tabular("Train/Actor Loss", actor_loss.cpu().data.numpy())
        logger.record_tabular("Train/BC Loss", bc_losses.mean().cpu().data.numpy())
        logger.record_tabular(
            "Train/True Actor Loss",
            F.mse_loss(policy_out.mean, action).cpu().data.numpy(),
        )

    def mask_agent(self, keep_masks):
        unhook(self.actor)
        monkey_patch(self.actor, keep_masks["actor"])
        apply_prune_mask(self.actor, keep_masks["actor"], fixed_weight=-1)

    def train_no_grad_update(
        self, replay_buffer, iterations, batch_size=256, discount=0.99
    ):
        for it in range(iterations):
            self.total_it += 1
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(
                batch_size
            )
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            policy_out = self.actor(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(action)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError
            actor_loss = torch.mean(bc_losses)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()

        logger.record_tabular("Train/Actor Loss", actor_loss.cpu().data.numpy())
        logger.record_tabular("Train/BC Loss", bc_losses.mean().cpu().data.numpy())
        logger.record_tabular(
            "Train/True Actor Loss",
            F.mse_loss(policy_out.mean, action).cpu().data.numpy(),
        )

    def evaluate_validation_performance(self, replay_buffer):
        true_actor_loss = 0
        idx = 0
        valid_sample_size = replay_buffer.storage["valid_state"].shape[0]
        sample_size = 10000
        itr = int(valid_sample_size / sample_size)
        for i in range(itr):
            (
                state,
                action,
                next_state,
                next_action,
                rewards,
                not_done,
                true_Q,
                true_MC,
            ) = replay_buffer.validation_samples(idx=idx, sample_size=sample_size)
            idx += sample_size
            pi = self.actor(state).mean
            true_actor_loss += F.mse_loss(pi, action).cpu().data.numpy()
        logger.record_tabular("Valid/True Actor Loss", true_actor_loss / itr)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        # optimizer
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def get_net_weight_size(self, filename, directory):
        return (
            os.path.getsize("%s/%s_actor.pth" % (directory, filename)),
            os.path.getsize("%s/%s_critic.pth" % (directory, filename)),
            os.path.getsize("%s/%s_value.pth" % (directory, filename)),
        )

    def sparse_weights(self, directory, filename):
        def compress(model):
            res = OrderedDict()
            for name, param in model.named_parameters():
                res[name] = param.to_sparse()
            return res

        torch.save(compress(self.actor), "%s/%s_actor.pth" % (directory, filename))

    def remove_weights(self, filename, directory):
        os.remove("%s/%s_actor.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
