import random

import gymnasium as gym
import numpy as np

alpha = 0.1
gamma = 0.9
epsilon = 0.3  # Higher exploration speeds up full-table convergence
tolerance = 1e-4  # Tighter convergence threshold
patience = 500  # Must stay stable for 500 episodes in a row
stable_episodes = 0


ROWS = 4
COLS = 12
START = (ROWS - 1, 0)
GOAL = (ROWS - 1, COLS - 1)

# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTION_SYMBOLS = {
    UP: "^",
    RIGHT: ">",
    DOWN: "v",
    LEFT: "<",
}

A = {UP, RIGHT, DOWN, LEFT}
Q = np.zeros((ROWS * COLS, 4))

episodes = 500000


def state_id(row, col):
    return row * COLS + col


def step(row, col, action):

    reward = -1
    done = False
    new_row = row
    new_col = col

    if action == UP:  # up
        new_row -= 1

    elif action == RIGHT:  # right
        new_col += 1

    elif action == DOWN:  # down
        new_row += 1

    elif action == LEFT:  # left
        new_col -= 1

    # stay inside grid
    new_row = max(0, min(ROWS - 1, new_row))
    new_col = max(0, min(COLS - 1, new_col))
    # clif
    if new_row == ROWS - 1 and 1 <= new_col <= COLS - 2:
        reward = -100
        new_row, new_col = ROWS - 1, 0
    # goal
    elif (new_row, new_col) == GOAL:
        done = True
        reward = 0

    return new_row, new_col, reward, done


for episode in range(episodes):
    max_delta = 0.0
    total_reward = 0
    row, col = START
    done = False
    print(f"eposide {episode}")
    # epsilon = max(0.01, epsilon*0.99)
    while not done:
        state = state_id(row, col)
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        next_row, next_col, reward, done = step(row, col, action)

        next_state = state_id(next_row, next_col)
        oldQ = Q[state, action]
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        max_delta = max(max_delta, abs(Q[state, action] - oldQ))

        if done == True:
            print(f"episode {episode} is done ")

        row, col = next_row, next_col
        total_reward += reward

        # Check convergence once per episode (outside while loop)
    if max_delta < tolerance:
        stable_episodes += 1
        if stable_episodes >= patience:
            print(f"Q-table converged at episode {episode + 1}")
            break
    else:
        stable_episodes = 0

    if (episode + 1) % 50 == 0:
        print(f"Episode{episode + 1: 3d} | Total_reward {total_reward}")
print("\nTraining finished!\n")

# print Q table
np.set_printoptions(precision=2, suppress=True)

print("Q-table:")
print(Q)

print("\nLearned Policy:\n")

for row in range(ROWS):
    for col in range(COLS):
        if (row, col) == START:
            print("S", end=" ")
        elif (row, col) == GOAL:
            print("G", end=" ")
        elif row == ROWS - 1 and 1 <= col <= COLS - 2:
            print("C", end=" ")
        else:
            state = state_id(row, col)
            best_action = np.argmax(Q[state])
            print(ACTION_SYMBOLS[best_action], end=" ")

    print()


# play game with the q table
# env = gym.make("CliffWalking-v1", render_mode="human")

# obs, info = env.reset()

# done = False
# close = False

# while not done:

#     action = np.argmax(Q[obs])

#     obs, reward, terminated, truncated, info = env.step(action)

#     done = terminated or truncated

# env.close()
