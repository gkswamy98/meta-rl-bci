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


# Initialization
pygame.init()
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption('Cursor CTRL')
clock = pygame.time.Clock()
start_time = time.time()

# Cursor setup
cursor_pos = [int(SCREEN[0] / 2), int(SCREEN[1] / 2)]
goal_pos = [np.random.randint(SCREEN[0]), 300]

# Data
correct_times = []
incorrect_times = []
left_times = []
right_times = []

def render():
    screen.fill(BLACK)
    pygame.draw.circle(screen, WHITE, cursor_pos, CIRCLE_RADIUS, 0)
    pygame.draw.circle(screen, GREEN, goal_pos, CIRCLE_RADIUS, 0)
    pygame.display.update()
    pygame.time.delay(2500)


while time.time() - start_time < 1800:#1200: # 20 minutes
    human_ctrl = -1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    if cursor_pos[0] > goal_pos[0] and abs(cursor_pos[0] - goal_pos[0]) > 10:
        human_ctrl = LEFT
        robot_ctrl = RIGHT
    if cursor_pos[0] < goal_pos[0] and abs(cursor_pos[0] - goal_pos[0]) > 10:
        human_ctrl = RIGHT
        robot_ctrl = LEFT
    # if cursor_pos[1] > goal_pos[1] and abs(cursor_pos[1] - goal_pos[1]) > 10:
    #     human_ctrl = UP
    #     robot_ctrl = DOWN
    # if cursor_pos[1] < goal_pos[1] and abs(cursor_pos[1] - goal_pos[1]) > 10:
    #     human_ctrl = DOWN
    #     robot_ctrl = UP
    if human_ctrl < 0: # no feedback, whoops
        ctrl = human_ctrl
    else:
        if human_ctrl == LEFT:
            left_times.append(str(time.time()))
            np.save("left_times_8.npy", left_times)
        if human_ctrl == RIGHT:
            right_times.append(str(time.time()))
            np.save("right_times_8.npy", right_times)
        if np.random.rand() < 0.8: # (switch back to 0.5 if necc) follow person
            ctrl = human_ctrl
            correct_times.append(str(time.time()))
            np.save("correct_times_8.npy", correct_times)
        else: # disagree w/ person
            ctrl = robot_ctrl
            incorrect_times.append(str(time.time()))
            np.save("incorrect_times_8.npy", incorrect_times)
    if ctrl == LEFT:
        cursor_pos[0] -= 20 
    if ctrl == RIGHT:
        cursor_pos[0] += 20 
    # if ctrl == UP:
    #     cursor_pos[1] -= 20
    # if ctrl == DOWN:
    #     cursor_pos[1] += 20
    cursor_pos[0] = min(max(cursor_pos[0], 0), 600)
    cursor_pos[1] = min(max(cursor_pos[1], 0), 600)
    if np.sqrt((cursor_pos[0] - goal_pos[0]) ** 2 + (cursor_pos[1] - goal_pos[1]) ** 2) < 30.:
        goal_pos = [np.random.randint(SCREEN[0]), 300]
    render()