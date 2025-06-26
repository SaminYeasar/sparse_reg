import random
import numpy as np
import torch
from collections import deque


def d4rl_subsample_trajectories(
    dataset,
    env,
    replay_buffer,
    buffer_size=2000000,
    num_trajectories=None,
    AR=1,
    validation_set=False,
    rew_scale=[1, 0],
    top10percent=False,
):
    all_obs = dataset["observations"]
    all_act = dataset["actions"]
    if top10percent:
        N = all_obs.shape[0]
    else:
        N = min(all_obs.shape[0], buffer_size)
    expert_states = all_obs[:N]
    expert_actions = all_act[:N]
    if "next_observations" in dataset:
        expert_next_states = dataset["next_observations"][:N]
    else:
        expert_next_states = np.concatenate(
            [all_obs[1:N, :], np.zeros_like(expert_states[0])[np.newaxis, :]], axis=0
        )
    expert_reward = dataset["rewards"][:N]
    expert_reward = (
        expert_reward * rew_scale[0] + np.ones_like(expert_reward) * rew_scale[1]
    )
    expert_dones = dataset["terminals"][:N]
    expert_timeouts = dataset["timeouts"][:N]
    expert_dones[np.where(dataset["timeouts"][:N] == 1)] = True

    expert_states_traj = [[]]
    expert_actions_traj = [[]]
    expert_rewards_traj = [[]]
    expert_next_states_traj = [[]]
    expert_dones_traj = [[]]
    expert_next_actions_traj = [[]]
    Z = AR
    action_que = deque(maxlen=Z)
    traj_terminate = Z

    for i in range(expert_states.shape[0] - (Z - 1)):
        if traj_terminate < Z:
            traj_terminate += 1
        else:
            expert_states_traj[-1].append(expert_states[i])
            r = 0
            action_que = deque(maxlen=Z)
            for j in range(Z):
                action_que.append(
                    expert_actions[i + j]
                )  # this will add from t+0, t+1, t+2, t+3
                r += expert_reward[i + j]  # accumulate rewards
            expert_rewards_traj[-1].append(r)
            expert_actions_traj[-1].append(np.mean(action_que, axis=0))
            expert_next_states_traj[-1].append(expert_next_states[i + Z - 1])
            expert_dones_traj[-1].append(expert_dones[i + Z - 1])

        if (expert_dones[i + Z - 1]) or (expert_timeouts[i + Z - 1]):
            expert_states_traj.append([])
            expert_actions_traj.append([])
            expert_rewards_traj.append([])
            expert_next_states_traj.append([])
            expert_dones_traj.append([])
            traj_terminate = 0

    expert_states_traj = [
        expert_states_traj[i]
        for i in range(len(expert_states_traj))
        if len(expert_states_traj[i]) > 10
    ]
    expert_actions_traj = [
        expert_actions_traj[i]
        for i in range(len(expert_actions_traj))
        if len(expert_actions_traj[i]) > 10
    ]
    expert_rewards_traj = [
        expert_rewards_traj[i]
        for i in range(len(expert_rewards_traj))
        if len(expert_rewards_traj[i]) > 10
    ]
    expert_next_states_traj = [
        expert_next_states_traj[i]
        for i in range(len(expert_next_states_traj))
        if len(expert_next_states_traj[i]) > 10
    ]
    expert_dones_traj = [
        expert_dones_traj[i]
        for i in range(len(expert_dones_traj))
        if len(expert_dones_traj[i]) > 10
    ]

    # expert next action
    for per_traj_actions in expert_actions_traj:
        for i in range(len(per_traj_actions) - 1):
            expert_next_actions_traj[-1].append(per_traj_actions[i + 1])
        expert_next_actions_traj[-1].append(np.zeros_like(per_traj_actions[i + 1]))
        expert_next_actions_traj.append([])
    expert_next_actions_traj = [
        expert_next_actions_traj[i]
        for i in range(len(expert_next_actions_traj))
        if len(expert_next_actions_traj[i]) != 0
    ]

    if top10percent:
        return_list = [
            sum(expert_rewards_traj[i]) for i in range(len(expert_rewards_traj))
        ]
        top_inds = np.argsort(return_list)[
            : int(len(expert_rewards_traj) * 0.1)
        ]  # number of traj for 10% int(len(expert_rewards_traj)*0.1)
        random.shuffle(top_inds)
        expert_states_traj = [expert_states_traj[i] for i in top_inds]
        expert_actions_traj = [expert_actions_traj[i] for i in top_inds]
        expert_rewards_traj = [expert_rewards_traj[i] for i in top_inds]
        expert_next_states_traj = [expert_next_states_traj[i] for i in top_inds]
        expert_dones_traj = [expert_dones_traj[i] for i in top_inds]
        expert_next_actions_traj = [expert_next_actions_traj[i] for i in top_inds]
    else:
        shuffle_inds = list(range(len(expert_states_traj)))
        random.shuffle(shuffle_inds)
        expert_states_traj = [expert_states_traj[i] for i in shuffle_inds]
        expert_actions_traj = [expert_actions_traj[i] for i in shuffle_inds]
        expert_rewards_traj = [expert_rewards_traj[i] for i in shuffle_inds]
        expert_next_states_traj = [expert_next_states_traj[i] for i in shuffle_inds]
        expert_dones_traj = [expert_dones_traj[i] for i in shuffle_inds]
        expert_next_actions_traj = [expert_next_actions_traj[i] for i in shuffle_inds]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    traj_reward = [
        sum(expert_rewards_traj[i])
        for i in range(len(expert_rewards_traj))
        if len(expert_rewards_traj[i]) != 0
    ]
    env.avg_score = sum(traj_reward) / len(traj_reward)
    print(f"expert score: {sum(traj_reward) / len(traj_reward)}")
    print("len : {}".format(len(expert_states_traj)))
    replay_buffer.data_size = min(
        concat_trajectories(expert_states_traj).shape[0], buffer_size
    )
    replay_buffer.storage["observations"][: replay_buffer.data_size] = (
        concat_trajectories(expert_states_traj).astype(np.float32)
    )
    replay_buffer.storage["actions"][: replay_buffer.data_size] = concat_trajectories(
        expert_actions_traj
    ).astype(np.float32)
    replay_buffer.storage["rewards"][: replay_buffer.data_size] = (
        concat_trajectories(expert_rewards_traj).reshape(-1, 1).astype(np.float32)
    )
    replay_buffer.storage["next_observations"][: replay_buffer.data_size] = (
        concat_trajectories(expert_next_states_traj).astype(np.float32)
    )
    replay_buffer.storage["terminals"][: replay_buffer.data_size] = (
        concat_trajectories(expert_dones_traj).reshape(-1, 1).astype(np.float32)
    )
    replay_buffer.storage["next_actions"][: replay_buffer.data_size] = (
        concat_trajectories(expert_next_actions_traj).astype(np.float32)
    )
    replay_buffer.storage["expert_score"] = (
        sum(traj_reward) / len(traj_reward)
    ).astype(np.float32)


def get_validation_data(dataset, replay_buffer):
    replay_buffer.storage["valid_state"] = np.array(
        dataset["observations"][: -800 * 1000]
    ).astype(np.float32)
    replay_buffer.storage["valid_action"] = np.array(
        dataset["actions"][: -800 * 1000]
    ).astype(np.float32)
    replay_buffer.storage["valid_rewards"] = (
        np.array(dataset["rewards"][: -800 * 1000]).reshape(-1, 1).astype(np.float32)
    )
    replay_buffer.storage["valid_next_state"] = np.array(
        dataset["next_observations"][: -800 * 1000]
    ).astype(np.float32)
    replay_buffer.storage["valid_terminals"] = (
        np.array(dataset["terminals"][: -800 * 1000]).reshape(-1, 1).astype(np.float32)
    )
    replay_buffer.storage["valid_terminals"][
        np.where(dataset["timeouts"][: -800 * 1000] == 1)
    ] = True
    next_action = np.array(dataset["actions"][: -800 * 1000]).astype(np.float32)
    replay_buffer.storage["valid_next_action"] = np.concatenate(
        [next_action[1:, :], np.zeros_like(next_action[0])[np.newaxis, :]], axis=0
    )
    rewards = (
        np.array(dataset["rewards"][: -800 * 1000]).reshape(-1, 1).astype(np.float32)
    )

    # compute true Q values
    replay_buffer.storage["valid_true_Q"] = [[]]
    replay_buffer.storage["valid_true_MC"] = [[]]
    R = 0
    MC = 0
    for r, d in zip(rewards[::-1], replay_buffer.storage["terminals"][::-1]):
        if d:
            replay_buffer.storage["valid_true_Q"].append([])
            R = 0
            MC = 0
        R = float(r) + 0.99 * R
        MC = float(r) + MC
        replay_buffer.storage["valid_true_Q"][-1].insert(
            0, R
        )  # r0 = \sum_0^t \gamma^t  r_t: insert from left, this makes sure we calculate rewards backwards
        replay_buffer.storage["valid_true_MC"][-1].insert(0, MC)
    replay_buffer.storage["valid_true_Q"] = (
        np.concatenate(replay_buffer.storage["valid_true_Q"])
        .reshape(-1, 1)
        .astype(np.float32)
    )
    replay_buffer.storage["valid_true_MC"] = (
        np.concatenate(replay_buffer.storage["valid_true_MC"])
        .reshape(-1, 1)
        .astype(np.float32)
    )
    assert (
        replay_buffer.storage["valid_true_Q"].shape
        == replay_buffer.storage["valid_rewards"].shape
    )
