import argparse
import heapq
import os
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from cem_auxiliar import setup_env, plot_history, setup_policy


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_elite_indicies(num_elite, rewards):
    return heapq.nlargest(num_elite, range(len(rewards)), key=rewards.__getitem__)


def evaluate_theta(theta, env_id, monitor=False):
    env, _, _ = setup_env(env_id)

    if monitor:
        env = gym.wrappers.RecordVideo(env, video_folder="./{}/videos/".format(env_id), name_prefix="cem")

    policy = setup_policy(env, theta)

    terminated = False
    truncated = False
    observation, info = env.reset()
    rewards = []

    while not terminated and not truncated:
        action = policy.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        observation = next_observation

    if monitor:
        env.close()

    return sum(rewards)


def run_cem(
    env_id,
    epochs=10,
    batch_size=4096,
    elite_frac=0.2,
    extra_std=2.0,
    extra_decay_time=10,
    num_process=8,
):
    ensure_dir("./{}/".format(env_id))
    ensure_dir("./{}/videos/".format(env_id))

    start = time.time()
    num_episodes = epochs * num_process * batch_size
    print("expt of {} total episodes".format(num_episodes))

    num_elite = int(batch_size * elite_frac)
    history = defaultdict(list)

    env, obs_shape, act_shape = setup_env(env_id)
    theta_dim = (obs_shape + 1) * act_shape
    means = np.random.uniform(size=theta_dim)
    stds = np.ones(theta_dim)

    for epoch in range(epochs):
        extra_cov = max(1.0 - epoch / extra_decay_time, 0) * extra_std**2

        thetas = np.random.multivariate_normal(
            mean=means, cov=np.diag(np.array(stds**2) + extra_cov), size=batch_size
        )

        with Pool(num_process) as p:
            rewards = p.map(partial(evaluate_theta, env_id=env_id), thetas)

        rewards = np.array(rewards)

        indicies = get_elite_indicies(num_elite, rewards)
        elites = thetas[indicies]

        means = elites.mean(axis=0)
        stds = elites.std(axis=0)

        history["epoch"].append(epoch)
        history["avg_rew"].append(np.mean(rewards))
        history["std_rew"].append(np.std(rewards))
        history["avg_elites"].append(np.mean(rewards[indicies]))
        history["std_elites"].append(np.std(rewards[indicies]))

        print(
            "epoch {} - {:2.1f} {:2.1f} pop - {:2.1f} {:2.1f} elites".format(
                epoch,
                history["avg_rew"][-1],
                history["std_rew"][-1],
                history["avg_elites"][-1],
                history["std_elites"][-1],
            )
        )

    end = time.time()
    expt_time = end - start
    print("expt took {:2.1f} seconds".format(expt_time))

    plot_history(history, env_id, num_episodes, expt_time)
    num_optimal = 3
    print("epochs done - evaluating {} best thetas".format(num_optimal))

    best_theta_rewards = [
        evaluate_theta(theta, env_id, monitor=True) for theta in elites[:num_optimal]
    ]
    print("best rewards - {} across {} samples".format(best_theta_rewards, num_optimal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env")
    parser.add_argument("--num_process", default=2, nargs="?", type=int)
    parser.add_argument("--epochs", default=5, nargs="?", type=int)
    parser.add_argument("--batch_size", default=4096, nargs="?", type=int)
    args = parser.parse_args()
    print(args)

    run_cem(
        args.env,
        num_process=args.num_process,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )