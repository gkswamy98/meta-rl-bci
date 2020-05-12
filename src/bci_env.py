import gym
from gym import spaces
from .data_utils import format_datasets
import numpy as np
import time

def inf_loop_gen(arr):
    i = 0
    while True:
        yield arr[i % len(arr)]
        i += 1

class BCIEnv(gym.Env):
    def __init__(self,
                 state_dim=(1, 16, 125),
                 num_actions=2,
                 is_live=False,
                 data_idx=7,
                 delay=2.5,
                 freq=125,
                 streamer=None,
                 reward_dec=None,
                 cursor_ctrl=None,
                 task="erp",
                 is_testing=False):
        super(BCIEnv, self).__init__()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-10, high=10, shape=state_dim, dtype=np.float64)
        self.ep_len = 1
        self.t = 0
        self.delay = delay
        self.freq = freq
        if is_live:
            self.streamer = streamer
            self.reward_dec = reward_dec
            self.cursor_ctrl = cursor_ctrl
        else:
            dataset = format_datasets([data_idx], task=task)[0]
            x_train, y_train, x_test, y_test = dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]
            if is_testing:
                self.x_train = x_train
                self.y_train = y_train
            else:
                self.x_train = x_test
                self.y_train = y_test
                self.state_gen = inf_loop_gen(self.x_train)
                self.opt_act_gen = inf_loop_gen(self.y_train)
            
    def step(self, action):
        if not self.is_live:
            obs = next(self.state_gen)
            opt_action = next(self.opt_act_gen)
            if action == np.argmax(opt_action): 
                reward = 1. / self.ep_len # effectively SQIL during warmup
            else:
                reward = 0.
        else:
            self.cursor_ctrl.execute_ctrl(action)
            time.sleep(self.delay)
            data = self.steamer.get_data(int(self.delay * self.freq), [(0.5, 40.), (1., 60.)])
            is_error = np.argmax(self.reward_dec.predict(data[0][:, :, :self.freq]), axis=-1)
            reward = np.abs(1 - is_error) / self.ep_len
            obs = data[1][:, :, int(1.4 * self.freq): int(2.4 * self.freq)]
        if self.t % self.ep_len == 0:
            done = True
        else:
            done = False
        info = dict()
        self.t += 1
        return obs, reward, done, info
    
    def reset(self):
        if not self.is_live:
            self.state_gen = inf_loop_gen(self.x_train)
            self.opt_act_gen = inf_loop_gen(self.y_train)
            return next(self.state_gen)
        else:
            data = self.steamer.get_data(int(self.delay * self.freq), [(1., 60.)])
            return data[0][:, :, int(1.4 * self.freq): int(2.4 * self.freq)]

    def render(self, mode='human'):
        if self.is_live:
            self.cursor_ctrl.render()
    
    def close(self):
        pass