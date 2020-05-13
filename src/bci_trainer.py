import os
import time
import logging
import argparse

import numpy as np
import tensorflow as tf
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.dict import Dict

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.envs.utils import is_discrete
from tf2rl.misc.get_replay_buffer import get_space_size, get_default_rb_dict
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer
from tf2rl.experiments.trainer import Trainer


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


def get_replay_buffer(policy, env, use_prioritized_rb=False,
                      use_nstep_rb=False, n_step=1, size=None):
    if policy is None or env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if size is not None:
        kwargs["size"] = size

    # on-policy policy
    if not issubclass(type(policy), OffPolicyAgent):
        kwargs["size"] = policy.horizon
        kwargs["env_dict"].pop("next_obs")
        kwargs["env_dict"].pop("rew")
        kwargs["env_dict"]["logp"] = {}
        kwargs["env_dict"]["ret"] = {}
        kwargs["env_dict"]["adv"] = {}
        if is_discrete(env.action_space):
            kwargs["env_dict"]["act"]["dtype"] = np.int32
        return ReplayBuffer(**kwargs)

    # N-step prioritized
    if use_prioritized_rb and use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return PrioritizedReplayBuffer(**kwargs)

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return ReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)


class BCITrainer(Trainer):
    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = self._env.reset()

        while total_steps < self._max_steps:
            # Replay collected data to get around live training
            if hasattr(self, 'rep_buff'):
                replay_buffer.add(obs=self.rep_buff["obs"][total_steps % len(self.rep_buff["obs"])],
                                  act=self.rep_buff["act"][total_steps % len(self.rep_buff["act"])],
                                  next_obs=self.rep_buff["rew"][(total_steps + 1) % len(self.rep_buff["rew"])],
                                  rew=self.rep_buff["rew"][total_steps % len(self.rep_buff["rew"])],
                                  done=True)
                if total_steps % self._policy.update_interval == 0:
                    samples = replay_buffer.sample(self._policy.batch_size)
                    with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                        self._policy.train(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32),
                            None if not self._use_prioritized_rb else samples["weights"])
                    if self._use_prioritized_rb:
                        td_error = self._policy.compute_td_error(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32))
                        replay_buffer.update_priorities(
                            samples["indexes"], np.abs(td_error) + 1e-6)
            else:
                if total_steps < self._policy.n_warmup:
                    action = self._env.action_space.sample()
                else:
                    action = self._policy.get_action(obs)
                    

                next_obs, reward, done, _ = self._env.step(action)
                if self._show_progress:
                    self._env.render()
                episode_steps += 1
                episode_return += reward
                total_steps += 1
                tf.summary.experimental.set_step(total_steps)

                
                done_flag = done
                            
                if hasattr(self._env, "_max_episode_steps") and \
                        episode_steps == self._env._max_episode_steps:
                    done_flag = False
                    
                replay_buffer.add(obs=obs, act=action,
                                  next_obs=next_obs, rew=reward, done=done_flag)
                obs = next_obs
                if done or episode_steps == self._episode_max_steps:
                    # obs = self._env.reset() # EPS are length 1

                    n_episode += 1
                    fps = episode_steps / (time.perf_counter() - episode_start_time)
                    self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                        n_episode, total_steps, episode_steps, episode_return, fps))
                    tf.summary.scalar(
                        name="Common/training_return", data=episode_return)

                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.perf_counter()

                if total_steps < self._policy.n_warmup:
                    continue

                if total_steps % self._policy.update_interval == 0:
                    samples = replay_buffer.sample(self._policy.batch_size)
                    with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                        self._policy.train(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32),
                            None if not self._use_prioritized_rb else samples["weights"])
                    if self._use_prioritized_rb:
                        td_error = self._policy.compute_td_error(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32))
                        replay_buffer.update_priorities(
                            samples["indexes"], np.abs(td_error) + 1e-6)

                if total_steps % self._test_interval == 0:
                    avg_test_return = self.evaluate_policy(total_steps)
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes))
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    tf.summary.scalar(name="Common/fps", data=fps)
                    self.writer.flush()

                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

        tf.summary.flush()

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(
                *self._env.normalizer.get_params())
        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done, _ = self._test_env.step(action)
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action,
                                      next_obs=next_obs, rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images,)
        return avg_test_return / self._test_episodes