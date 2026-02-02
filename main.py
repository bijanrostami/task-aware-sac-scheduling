import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from env_step import env_step
from replay_memory import ReplayMemory
from sac import SAC

DATASET_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
    parser.add_argument("--env-name", default="Scheduling_SAC", help="Scheduling_SAC")
    parser.add_argument(
        "--policy",
        default="Gaussian",
        help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=False,
        help="Evaluates a policy a policy every 10 episode (default: True)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for reward (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        metavar="G",
        help="target smoothing coefficient (default: 0.005)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00003,
        metavar="G",
        help="learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=0.00002,
        metavar="G",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        metavar="G",
        help=(
            "Temperature parameter alpha determines the relative importance of the entropy "
            "term against the reward (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--automatic_entropy_tuning",
        type=bool,
        default=True,
        metavar="G",
        help="Automatically adjust alpha (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        metavar="N",
        help="random seed (default: 123456)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        metavar="N",
        help="batch size (default: 256)",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=499,
        metavar="N",
        help="maximum number of steps (TTI) (default: 500)",
    )
    parser.add_argument(
        "--max_episode",
        type=int,
        default=1500,
        metavar="N",
        help="maximum number of episodes (default: 1300)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        metavar="N",
        help="hidden size (default: 256)",
    )
    parser.add_argument(
        "--updates_per_step",
        type=int,
        default=1,
        metavar="N",
        help="model updates per simulator step (default: 1)",
    )
    parser.add_argument(
        "--save_per_epochs",
        type=int,
        default=15,
        metavar="N",
        help="save_per_epochs",
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=10000,
        metavar="N",
        help="Steps sampling random actions (default: 10000)",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=1,
        metavar="N",
        help="Value target update per no. of updates per step (default: 1)",
    )
    parser.add_argument(
        "--replay_size",
        type=int,
        default=1000000,
        metavar="N",
        help="size of replay buffer (default: 10000000)",
    )
    parser.add_argument("--cuda", default=1, help="run on CUDA (default: True)")
    parser.add_argument("--gpu_nums", type=int, default=1, help="#GPUs to use (default: 1)")
    return parser.parse_args()


def plot_series(title: str, values: np.ndarray) -> None:
    plt.plot(values, label=title)
    plt.title(title)
    plt.grid(True)
    plt.show()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path.append(str(DATASET_DIR))
    from main_data_generation import config  # pylint: disable=import-error

    signal_power = np.load(DATASET_DIR / "Channel_matrix_gain.npy")

    num_states = 420
    max_actions = 32
    num_actions = 4
    random_prob = 1.0
    epsilon = 10000.0

    max_delay_idx = np.random.rand(config.num_of_subnetworks) + 0.8
    max_data = 60 * np.random.randint(3, 6, config.num_of_subnetworks)
    data_ratio = 3
    h_std = 1.0
    h_mean = 1.0

    agent = SAC(num_states, num_actions, max_actions, args)
    memory = ReplayMemory(args.replay_size, args.seed)

    total_numsteps = 0
    updates = 0

    reward_record_tr = np.array([])
    reward_record_test = np.array([])
    t_slot_max = np.array([])
    t_slot_min = np.array([])
    t_slot_mean = np.array([])
    t_slot_std = np.array([])
    norm_delay_mean = np.array([])
    norm_delay_std = np.array([])
    rate_record_tr = np.array([])
    rate_record_test = np.array([])

    reward_scale = 0.4 / h_mean

    for i_episode in range(args.max_episode):
        evaluate = (i_episode + 1) % 50 == 0
        channel_ind = 0 if evaluate else np.random.randint(4)
        sample_idx = np.random.randint(10000)

        episode_q_loss = 0
        episode_log_prob = 0
        episode_steps = 0
        done = False
        episode_reward = 0
        total_rate = 0

        ue_history = np.zeros(config.num_of_subnetworks) + 0.00001
        fairness_coef_all = h_mean * np.ones(config.num_of_subnetworks)

        sig_scaled = signal_power[sample_idx, channel_ind] * 1e7
        sig_norm = sig_scaled.T / sig_scaled.max(axis=1)
        state = np.concatenate(
            (sig_norm.T.reshape(1, -1), fairness_coef_all.reshape(1, -1)), axis=1
        )

        t_slot_converged = 500 * np.ones(config.num_of_subnetworks)

        while not done:
            if random_prob > np.random.rand(1):
                action, actions_int = agent.random_action()
            else:
                action, actions_int, log_pi, _, _ = agent.select_action(state, evaluate)
                episode_log_prob += log_pi

            if actions_int == 0:
                continue

            random_prob -= 1 / epsilon

            action_bit_array = np.array(
                list(np.binary_repr(actions_int, width=config.num_of_subnetworks)), dtype=int
            )
            ue_select = np.where(action_bit_array == 1)[0]

            p_alloc = np.ones(len(ue_select))
            fairness_coef = fairness_coef_all[ue_select]
            h_full = signal_power[sample_idx, channel_ind]
            h_selected = h_full[ue_select, :][:, ue_select]

            final_power = h_selected * p_alloc

            rate, max_rate = env_step(
                final_power * 1e7,
                h_full * 1e7,
                config.noise_power * 1e7,
                config.ch_bandwidth,
            )

            modified_rate = rate * fairness_coef
            reward = (modified_rate.sum() / np.sort(max_rate)[5:].sum()) * reward_scale
            total_rate += (rate.squeeze() * 1e-8).sum()
            sample_idx += 1

            ue_history[ue_select] += rate.squeeze() * 1e-8
            normalized_data = ue_history / max_data
            new_idx = (data_ratio - normalized_data) / (
                max_delay_idx - (episode_steps * 0.0016)
            )
            t_slot_converged[(normalized_data >= 1.5) * (t_slot_converged == 500)] = (
                episode_steps
            )
            norm_delay = t_slot_converged / max_delay_idx

            fairness_temp_all = (new_idx - new_idx.mean()) / h_std
            fairness_coef_all = fairness_temp_all + h_mean

            sig_scaled = signal_power[sample_idx, channel_ind] * 1e7
            sig_norm = sig_scaled.T / sig_scaled.max(axis=1)
            next_state = np.concatenate(
                (sig_norm.T.reshape(1, -1), fairness_coef_all.reshape(1, -1)), axis=1
            )

            mask = 1
            if args.max_episode_steps and episode_steps >= args.max_episode_steps - 1:
                done = True

            if len(memory) > args.batch_size and not evaluate:
                for _ in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = (
                        agent.update_parameters(memory, args.batch_size, updates)
                    )
                    episode_q_loss += (critic_1_loss + critic_2_loss) / 2
                    updates += 1

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            if not evaluate:
                memory.push(state, action, reward, next_state, mask)

            state = next_state

        print(
            "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
            )
        )
        if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
            agent.save_checkpoint('SAC-v1')

        if evaluate:
            rate_record_test = np.append(rate_record_test, total_rate)
            reward_record_test = np.append(reward_record_test, episode_reward)
            t_slot_min = np.append(t_slot_min, t_slot_converged.min())
            t_slot_max = np.append(t_slot_max, t_slot_converged.max())
            t_slot_mean = np.append(t_slot_mean, t_slot_converged.mean())
            t_slot_std = np.append(t_slot_std, t_slot_converged.std())
            norm_delay_mean = np.append(norm_delay_mean, norm_delay.mean())
            norm_delay_std = np.append(norm_delay_std, norm_delay.std())
        else:
            rate_record_tr = np.append(rate_record_tr, total_rate)
            reward_record_tr = np.append(reward_record_tr, episode_reward)

    print("Training is finished")

    with open("power_refine.pkl", "wb") as f:
        pickle.dump(reward_record_tr, f)
        pickle.dump(reward_record_test, f)
        pickle.dump(rate_record_tr, f)
        pickle.dump(rate_record_test, f)
        pickle.dump(t_slot_min, f)
        pickle.dump(t_slot_max, f)
        pickle.dump(t_slot_mean, f)
        pickle.dump(t_slot_std, f)
        pickle.dump(norm_delay_mean, f)
        pickle.dump(norm_delay_std, f)


if __name__ == "__main__":
    main()
