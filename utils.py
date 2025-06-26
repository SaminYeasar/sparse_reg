import numpy as np
import pickle
import gzip
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim=10, action_dim=4, validation_set=False):
        self.storage = dict()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = 1000000
        self.init_buffer()
        self.ctr = 0
        self.data_size = 0
        self.mean = 0
        self.std = 1
        self.validation_set = validation_set

    def init_buffer(self):
        self.storage["observations"] = np.zeros(
            (self.buffer_size, self.state_dim), np.float32
        )
        self.storage["next_observations"] = np.zeros(
            (self.buffer_size, self.state_dim), np.float32
        )
        self.storage["actions"] = np.zeros(
            (self.buffer_size, self.action_dim), np.float32
        )
        self.storage["rewards"] = np.zeros((self.buffer_size, 1), np.float32)
        self.storage["terminals"] = np.zeros((self.buffer_size, 1), np.float32)
        self.storage["next_actions"] = np.zeros(
            (self.buffer_size, self.action_dim), np.float32
        )

    def add(self, data):
        try:
            self.storage["observations"][self.ctr] = data[0]
        except:
            pass
        self.storage["next_observations"][self.ctr] = data[1]
        self.storage["actions"][self.ctr] = data[2]
        self.storage["rewards"][self.ctr] = data[3]
        self.storage["terminals"][self.ctr] = data[4]
        self.ctr += 1
        self.data_size += 1
        self.ctr = self.ctr % self.buffer_size

    def sample(self, batch_size, with_data_policy=False):
        ind = np.random.randint(0, self.data_size, size=batch_size)
        s = self.storage["observations"][ind]
        a = self.storage["actions"][ind]
        r = self.storage["rewards"][ind]
        s2 = self.storage["next_observations"][ind]
        d = self.storage["terminals"][ind]

        if with_data_policy:
            data_mean = self.storage["data_policy_mean"][ind]
            data_cov = self.storage["data_policy_logvar"][ind]
            return (
                np.array(s),
                np.array(s2),
                np.array(a),
                np.array(r).reshape(-1, 1),
                np.array(d).reshape(-1, 1),
                np.array(data_mean),
                np.array(data_cov),
            )
        else:
            return (
                np.array(s),
                np.array(s2),
                np.array(a),
                np.array(r).reshape(-1, 1),
                np.array(d).reshape(-1, 1),
            )

    def validation_samples(self, idx, sample_size=10000):
        ind = list(np.arange(idx, idx + sample_size, 1))
        return (
            torch.FloatTensor(self.storage["valid_state"][ind]).to(device),
            torch.FloatTensor(self.storage["valid_action"][ind]).to(device),
            torch.FloatTensor(self.storage["valid_next_state"][ind]).to(device),
            torch.FloatTensor(self.storage["valid_next_action"][ind]).to(device),
            torch.FloatTensor(self.storage["valid_rewards"][ind]).to(device),
            torch.FloatTensor(1 - self.storage["valid_terminals"][ind]).to(device),
            torch.FloatTensor(self.storage["valid_true_Q"][ind]).to(device),
            torch.FloatTensor(self.storage["valid_true_MC"][ind]).to(device),
        )

    def normalize_states(self, eps=1e-3):
        mean = self.storage["observations"][: self.data_size].mean(0, keepdims=True)
        std = self.storage["observations"][: self.data_size].std(0, keepdims=True) + eps
        self.storage["observations"] = (self.storage["observations"] - mean) / std
        self.storage["next_observations"] = (
            self.storage["next_observations"] - mean
        ) / std
        if self.validation_set:
            self.storage["valid_state"] = (self.storage["valid_state"] - mean) / std
            self.storage["valid_next_state"] = (
                self.storage["valid_next_state"] - mean
            ) / std
        self.mean = mean
        self.std = std
        return mean, std

    def re_normalize(self):
        self.storage["observations"][: self.data_size] = (
            self.storage["observations"][: self.data_size] * self.std
        ) + self.mean
        self.storage["next_observations"][: self.data_size] = (
            self.storage["next_observations"][: self.data_size] * self.std
        ) + self.mean
        if self.validation_set:
            self.storage["valid_state"] = (
                self.storage["valid_state"] * self.std
            ) + self.mean
            self.storage["valid_next_state"] = (
                self.storage["valid_next_state"] * self.std
            ) + self.mean
        self.normalize_states()
        print("done re-normalize")

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def load(self, filename, bootstrap_dim=None):
        """Deprecated, use load_hdf5 in main.py with the D4RL environments"""
        with gzip.open(filename, "rb") as f:
            self.storage = pickle.load(f)

        sum_returns = self.storage["rewards"].sum()
        num_traj = self.storage["terminals"].sum()
        if num_traj == 0:
            num_traj = 1000
        average_per_traj_return = sum_returns / num_traj
        print("Average Return: ", average_per_traj_return)
        # import ipdb; ipdb.set_trace()

        num_samples = self.storage["observations"].shape[0]
        if bootstrap_dim is not None:
            self.bootstrap_dim = bootstrap_dim
            bootstrap_mask = np.random.binomial(
                n=1,
                size=(
                    1,
                    num_samples,
                    bootstrap_dim,
                ),
                p=0.8,
            )
            bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
            self.storage["bootstrap_mask"] = bootstrap_mask[:num_samples]
