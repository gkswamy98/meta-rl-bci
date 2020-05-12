import gym
from gym import spaces
from .data_utils import format_datasets

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
                 streamer=None,
                 task="erp",
                 is_testing=False):
        super(BCIEnv, self).__init__()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-10, high=10, shape=state_dim, dtype=np.float64)
        self.ep_len = 1
        self.t = 0
        if is_live:
            self.streamer = streamer
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
        obs = next(self.state_gen)
        opt_action = next(self.opt_act_gen)
        # TODO: classify using ERP classifier
        if action == np.argmax(opt_action): 
            reward = 1. / self.ep_len # effectively SQIL during warmup
        else:
            reward = 0.
        if self.t % self.ep_len == 0:
            done = True
        else:
            done = False
        info = dict()
        self.t += 1
        return obs, reward, done, info
    
    def reset(self):
        self.state_gen = inf_loop_gen(self.x_train)
        self.opt_act_gen = inf_loop_gen(self.y_train)
        return next(self.state_gen)
    
    def render(self, mode='human'):
        pass # TODO: incorporate w/ pygame
    
    def close(self):
        pass # TODO: need to close brainflow connection