import gym
import numpy as np
import torch
import argparse
import os
import utils
import algos
from logger import logger, setup_logger
import copy
from sparse_utils import sparse_net_func
from data_utils import d4rl_subsample_trajectories
import d4rl
from d4rl import gym_mujoco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="mujoco", type=str)
    parser.add_argument("--env_name", default="hopper-expert-v2")
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--eval_freq", default=1e3, type=float
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=1e6, type=float
    )  # Max time steps to run environment for
    parser.add_argument(
        "--algo_name", default="IQL", type=str
    )  # Which algo to run (see the options below in the main function)
    parser.add_argument(
        "--log_dir", default="./data_tmp/", type=str
    )  # Logging directory
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--experiment", type=str, default="None")
    parser.add_argument("--buffer_size", default=100000, type=int)
    parser.add_argument("--action_repeat", default=1, type=int)
    parser.add_argument("--actor_grad_norm_lambda", default=0.0, type=float)
    parser.add_argument("--use_validation_set", action="store_true", default=False)
    parser.add_argument("--experiment_name", default="sample_complexity")
    parser.add_argument("--clip_grad", default=None)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--TD3_BC_alpha", default=2.5, type=float)
    parser.add_argument("--IQL_beta", default=3.0, type=float)
    # pruning
    parser.add_argument("--activate_sparse_reg", action="store_true", default=False)
    parser.add_argument("--keep_ratio", default=0.05, type=float)
    parser.add_argument("--turn_off_sparse_reg_at", default=200000, type=int)
    args = parser.parse_args()

    return args


class Workspace(object):
    def __init__(self, args):

        self.args = args
        self.init_essentials()
        self.activation = "tanh"

    def init_essentials(self):
        # initialize variables
        self.env = gym.make(self.args.env_name)
        self.env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = torch.FloatTensor(self.env.action_space.high).to(device)

    def load_dataset(self):
        # Load dataset
        self.replay_buffer = utils.ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            validation_set=self.args.use_validation_set,
        )
        dataset = self.env.unwrapped.get_dataset()
        d4rl_subsample_trajectories(
            dataset,
            self.env,
            self.replay_buffer,
            buffer_size=self.args.buffer_size,
            AR=self.args.action_repeat,
            validation_set=False,
        )
        if self.args.use_validation_set:
            from data_utils import get_validation_data

            get_validation_data(dataset, self.replay_buffer)
        mean, std = self.replay_buffer.normalize_states()
        self.mean = mean
        self.std = std

    def set_logging(self):
        hparam_str_dict = dict(
            algo=self.args.algo_name,
            seed=self.args.seed,
            env=self.args.env_name,
            keep_ratio=self.args.keep_ratio,
            gamma=self.args.gamma,
            batch_size=self.args.batch_size,
            buffer_size=self.args.buffer_size,
            SNIP=self.args.activate_SNIP,
            hidden_dim=self.args.hidden_dim,
        )
        file_name = ",".join(
            [
                "%s=%s" % (k, str(hparam_str_dict[k]))
                for k in sorted(hparam_str_dict.keys())
            ]
        )
        print("---------------------------------------")
        print("Settings: " + file_name)
        print("---------------------------------------")

        variant = dict(
            algorithm=self.args.algo_name,
            env_name=self.args.env_name,
            seed=self.args.seed,
        )
        setup_logger(
            file_name,
            variant=variant,
            log_dir=os.path.join(self.args.log_dir, file_name),
        )

    def offline_algo(self):
        if self.args.algo_name == "IQL":
            self.policy = algos.IQL(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                hidden_dim=self.args.hidden_dim,
                actor_grad_norm_lambda=self.args.actor_grad_norm_lambda,
                activation=self.activation,
                lr=self.args.lr,
                beta=self.args.IQL_beta,
                tau=0.7,
            )
        elif self.args.algo_name == "AWAC":
            self.policy = algos.AWAC(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                hidden_dim=self.args.hidden_dim,
                actor_grad_norm_lambda=self.args.actor_grad_norm_lambda,
                activation=self.activation,
                clip_grad=self.args.clip_grad,
                lr=self.args.lr,
                tau=0.7,
            )
        elif self.args.algo_name == "BC":
            self.policy = algos.BC(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                hidden_dim=self.args.hidden_dim,
                actor_grad_norm_lambda=self.args.actor_grad_norm_lambda,
                activation=self.activation,
                clip_grad=self.args.clip_grad,
                lr=self.args.lr,
                beta=0,
                tau=0.7,
            )
        elif self.args.algo_name == "TD3_BC":
            self.policy = algos.TD3_BC(
                self.state_dim,
                self.action_dim,
                self.max_action,
                hidden_dim=self.args.hidden_dim,
                lr=self.args.lr,
                alpha=self.args.TD3_BC_alpha,
            )

    # Runs policy for X episodes and returns average reward
    def evaluate_policy(self, eval_episodes=10):
        avg_reward = 0.0
        all_rewards = []

        for t in range(eval_episodes):
            obs = self.env.reset()
            done = False
            ep_len = 0
            while not done:
                obs = (np.array(obs).reshape(1, -1) - self.mean) / self.std
                action = self.policy.select_action(obs)
                obs, reward, done, info = self.env.step(action)
                ep_len += 1
                if ep_len + 1 == self.env._max_episode_steps:
                    done = True
                if done:
                    break
                avg_reward += reward
            all_rewards.append(avg_reward)

        avg_reward /= eval_episodes
        for j in range(eval_episodes - 1, 1, -1):
            all_rewards[j] = all_rewards[j] - all_rewards[j - 1]
        all_rewards = np.array(all_rewards)
        std_rewards = np.std(all_rewards)
        median_reward = np.median(all_rewards)
        print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        print("---------------------------------------")
        return avg_reward, std_rewards, median_reward

    def apply_sparse_reg(self):

        sparse_net = sparse_net_func(self.args.keep_ratio, algo=self.args.algo_name)
        keep_masks = sparse_net.get_masks(
            copy.deepcopy(self.policy), self.replay_buffer
        )
        self.policy.mask_agent(keep_masks)

    def train(self):
        training_iters = 0
        while training_iters < self.args.max_timesteps:
            self.policy.train(
                self.replay_buffer,
                iterations=int(self.args.eval_freq),
                batch_size=self.args.batch_size,
            )
            training_iters += self.args.eval_freq
            ret_eval, var_ret, median_ret = self.evaluate_policy(eval_episodes=10)
            print("Training iterations: " + str(training_iters))
            logger.record_tabular(
                "Training Epochs", int(training_iters // int(self.args.eval_freq))
            )
            logger.record_tabular("Eval/AverageReturn", ret_eval)
            if self.args.use_validation_set:
                self.policy.evaluate_validation_performance(self.replay_buffer)
            logger.dump_tabular()

            # iterative update of sparse-reg
            if (
                self.args.activate_sparse_reg
                and training_iters < self.args.turn_off_sparse_reg_at
            ):
                self.apply_sparse_reg()


def main():
    args = get_args()

    # import offline rl algorithm
    agent = Workspace(args)
    agent.offline_algo()

    # import dataset
    agent.load_dataset()

    # mask a sparse network
    if args.activate_sparse_reg:
        agent.apply_sparse_reg()

    # train offline
    agent.train()


if __name__ == "__main__":
    main()
