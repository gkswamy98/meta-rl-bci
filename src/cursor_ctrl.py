import pygame
import sys
import numpy as np
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 255, 50)
BLUE=(0,0,255)
RED=(255,0,0)
CIRCLE_RADIUS = 20
SCREEN = 600, 600
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class CursorCtrl:
    def __init__(self, num_actions=2, delay=2.5, data_idx=101, streamer=None):
        self.data_idx = data_idx
        self.num_actions = num_actions
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN)
        self.delay = delay
        pygame.display.set_caption('Cursor CTRL')
        self.action_buffer = []
        # Data Recording
        self.streamer = streamer
        self.ctrl_buffer = []
        self.erp_buffer = []
        self.ts_buffer = []
        # SSVEP
        self.ctrl_freqs = [10.25, 14.0]
        self.ctrl_vis = [True, True]
        self.ctrl_ts = [time.time(), time.time()]
        self.ctrl_locs = [(0,250,50,100), (550,250,50,100)]
        self.ctrl_col = [BLUE, RED]
        self.reset()
    def render(self):
        self.screen.fill(BLACK)
        pygame.draw.circle(self.screen, WHITE, self.cursor_pos, CIRCLE_RADIUS, 0)
        pygame.draw.circle(self.screen, GREEN, self.goal_pos, CIRCLE_RADIUS, 0)
        for c in range(self.num_actions):
            if self.ctrl_vis[c]:
                pygame.draw.rect(self.screen, self.ctrl_col[c], self.ctrl_locs[c])
        pygame.display.update()
    def execute_ctrl(self, ctrl):
        if ctrl == LEFT:
            self.cursor_pos[0] -= 20 
        if ctrl == RIGHT:
            self.cursor_pos[0] += 20 
        self.cursor_pos[0] = min(max(self.cursor_pos[0], 0), SCREEN[0])
        self.cursor_pos[1] = min(max(self.cursor_pos[1], 0), SCREEN[1])
        if np.sqrt((self.cursor_pos[0] - self.goal_pos[0]) ** 2 + (self.cursor_pos[1] - self.goal_pos[1]) ** 2) < 30.:
            self.goal_pos = [np.random.randint(SCREEN[0]), 300]
    def run_game(self, game_len=1800):
        start_time = time.time()
        while time.time() - start_time < game_len:#1200: # 20 minutes
            human_ctrl = -1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if self.cursor_pos[0] > self.goal_pos[0] and abs(self.cursor_pos[0] - self.goal_pos[0]) > 10:
                human_ctrl = LEFT
                robot_ctrl = RIGHT
            if self.cursor_pos[0] < self.goal_pos[0] and abs(self.cursor_pos[0] - self.goal_pos[0]) > 10:
                human_ctrl = RIGHT
                robot_ctrl = LEFT
            if self.cursor_pos[1] > self.goal_pos[1] and abs(self.cursor_pos[1] - self.goal_pos[1]) > 10:
                human_ctrl = UP
                robot_ctrl = DOWN
            if self.cursor_pos[1] < self.goal_pos[1] and abs(self.cursor_pos[1] - self.goal_pos[1]) > 10:
                human_ctrl = DOWN
                robot_ctrl = UP
            if human_ctrl < 0:
                ctrl = human_ctrl
            else:
                if human_ctrl == LEFT:
                    self.action_times[LEFT].append(str(time.time()))
                    np.save("./data/left_times_{0}.npy".format(self.data_idx), self.action_times[LEFT])
                if human_ctrl == RIGHT:
                    self.action_times[RIGHT].append(str(time.time()))
                    np.save("./data/right_times_{0}.npy".format(self.data_idx), self.action_times[RIGHT])
                if np.random.rand() < 0.8:
                    ctrl = human_ctrl
                    self.correct_times.append(str(time.time()))
                    np.save("./data/correct_times_{0}.npy".format(self.data_idx), self.correct_times)
                else:
                    ctrl = robot_ctrl
                    self.incorrect_times.append(str(time.time()))
                    np.save("./data/incorrect_times_{0}.npy".format(self.data_idx), self.incorrect_times)
            self.execute_ctrl(ctrl)
            self.render_for(self.delay)
            if self.streamer is not None:
                samples, ts = self.streamer.get_data(int(self.delay * 125), [(0.5, 40.), (9., 50.)], time=True)
                self.erp_buffer.append(samples[0])
                self.ctrl_buffer.append(samples[1])
                self.ts_buffer.append(ts)
    def render_for(self, game_len=1800):
        start_time = time.time()
        for c in range(self.num_actions):
            self.ctrl_ts[c] = start_time
        t = time.time()
        while t - start_time < game_len:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if len(self.action_buffer) > 0:
                action = self.action_buffer[0]
                self.action_buffer = self.action_buffer[1:]
                self.execute_ctrl(action)
            for c in range(self.num_actions):
                if t > self.ctrl_ts[c]:
                    self.ctrl_vis[c] = not self.ctrl_vis[c]
                    self.ctrl_ts[c] += 1. / self.ctrl_freqs[c]
            self.render()
            t = time.time()
    def delay_for(self, t):
        pygame.time.delay(int(t * 1000))
    def reset(self):
        self.cursor_pos = [int(SCREEN[0] / 2), int(SCREEN[1] / 2)]
        self.goal_pos = [np.random.randint(SCREEN[0]), int(SCREEN[1] / 2)]
        self.correct_times = []
        self.incorrect_times = []
        self.action_times = [[] for _ in range(self.num_actions)]
    def close(self):
        if self.streamer is not None:
            d1 = np.concatenate(self.erp_buffer, axis=-1)
            d2 = np.concatenate(self.ctrl_buffer, axis=-1)
            d3 = np.concatenate(self.ts_buffer, axis=-1)
            np.save("./data/eeg_data_{0}.npy".format(self.data_idx), np.transpose(d1))
            np.save("./data/eeg_fft_data_{0}.npy".format(self.data_idx), np.transpose(d2))
            np.save("./data/eeg_timestamps_{0}.npy".format(self.data_idx), d3)
        pygame.display.quit()
        pygame.quit()

