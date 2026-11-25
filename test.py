import gymnasium as gym

env = gym.make("Cliff-world")

episodes = 100

scores = []

for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0

    done = False

    while not done:

        action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

    scores.append(total_reward)

print("Average Reward:", sum(scores)/len(scores))

env.close()