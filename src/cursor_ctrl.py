import pygame
import sys
import numpy as np
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 255, 50)
CIRCLE_RADIUS = 20
SCREEN = 600, 600
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class CursorCtrl:
    def __init__(self, num_actions=2, delay=2.5, data_idx=101):
        self.data_idx = data_idx
        self.num_actions = num_actions
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN)
        self.delay = delay
        pygame.display.set_caption('Cursor CTRL')
        self.reset()
    def render(self):
        self.screen.fill(BLACK)
        pygame.draw.circle(self.screen, WHITE, self.cursor_pos, CIRCLE_RADIUS, 0)
        pygame.draw.circle(self.screen, GREEN, self.goal_pos, CIRCLE_RADIUS, 0)
        pygame.display.update()
        pygame.time.delay(int(1000 * self.delay))
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
            if ctrl == LEFT:
                self.cursor_pos[0] -= 20 
            if ctrl == RIGHT:
                self.cursor_pos[0] += 20 
            self.cursor_pos[0] = min(max(self.cursor_pos[0], 0), SCREEN[0])
            self.cursor_pos[1] = min(max(self.cursor_pos[1], 0), SCREEN[1])
            if np.sqrt((self.cursor_pos[0] - self.goal_pos[0]) ** 2 + (self.cursor_pos[1] - self.goal_pos[1]) ** 2) < 30.:
                self.goal_pos = [np.random.randint(SCREEN[0]), 300]
            self.render()
    def reset(self):
        self.cursor_pos = [int(SCREEN[0] / 2), int(SCREEN[1] / 2)]
        self.goal_pos = [np.random.randint(SCREEN[0]), int(SCREEN[1] / 2)]
        self.correct_times = []
        self.incorrect_times = []
        self.action_times = [[] for _ in range(self.num_actions)]
    def close(self):
        pygame.display.quit()
        pygame.quit()

