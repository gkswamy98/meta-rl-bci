import gym
from gym import spaces
import numpy as np
import time
from src.data_utils import denoise

def inf_loop_gen(arr):
    i = 0
    while True:
        yield arr[i % len(arr)]
        i += 1

class BCIEnv(gym.Env):
    def __init__(self,
                 state_dim=(1, 8, 250),
                 num_actions=4,
                 is_live=False,
                 data_idx=7,
                 delay=2.5,
                 freq=250,
                 streamer=None,
                 reward_dec=None,
                 cursor_ctrl=None,
                 task="erp",
                 is_testing=False,
                 dataset=None):
        super(BCIEnv, self).__init__()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-10, high=10, shape=state_dim, dtype=np.float64)
        self.ep_len = 1
        self.t = 0
        self.delay = delay
        self.freq = freq
        self.is_live = is_live
        self.last_opt = -1
        self.true_errors = []
        self.pred_errors = []
        if is_live:
            self.streamer = streamer
            self.reward_dec = reward_dec
            self.cursor_ctrl = cursor_ctrl
        else:
            if dataset is None:
                dataset = format_datasets([data_idx], task=task)[0]
            x_train, y_train, x_test, y_test = dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]
            if is_testing:
                self.x_train = x_test
                self.y_train = y_test
            else:
                self.x_train = x_train
                self.y_train = y_train
            self.state_gen = inf_loop_gen(self.x_train)
            self.opt_act_gen = inf_loop_gen(self.y_train)
            
    def step(self, action, sim=True):
        if not self.is_live:
            obs = next(self.state_gen)
            opt_action = next(self.opt_act_gen)
            if action == np.argmax(opt_action): 
                reward = 1. / self.ep_len # effectively SQIL during warmup
            else:
                reward = 0.
        else:
            self.cursor_ctrl.action_buffer.append(action)
            if sim:
                true_error = (action != self.last_opt)
                opt = self.cursor_ctrl.get_optimal_action()
                data = self.streamer.get_data(int(self.delay * self.freq),
                                             [(0.5, 40.), (6., 50.)],
                                             freq_to_add=self.cursor_ctrl.ctrl_freqs[opt],
                                             add_error=true_error)
                self.last_opt = opt
            else:
                data = self.streamer.get_data(int(self.delay * self.freq), [(0.5, 40.), (6., 50.)])
            rew_dec_input = np.expand_dims(denoise(data[0][:, :, :self.freq]), axis=0)
            is_error = np.argmax(self.reward_dec.predict(rew_dec_input), axis=-1)
            self.true_errors.append(true_error)
            self.pred_errors.append(np.sum(is_error))
            # reward = 1 - int(true_error)
            # if np.random.rand() > 0.65: # works at 75, doesn't at 70% (works at 75 for simple task)
            #     reward = 1 - reward # flip reward
            reward = np.abs(1 - is_error) / self.ep_len
            obs = data[1][:, :, int(0.5 * self.freq): int(1.5 * self.freq)]
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
            time.sleep(self.delay + 1.)
            data = self.streamer.get_data(int(self.delay * self.freq), [(6., 50.)])
            obs = data[0][:, :, int(0.5 * self.freq): int(1.5 * self.freq)]
            return obs

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass