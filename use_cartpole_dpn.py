import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN.load("cartpole_dqn")

scores = []

for episode in range(100):


    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated
    
    scores.append(total_reward)    

print("average: ", sum(scores)/len(scores))
print("Max: ", max(scores))
print("Min: ", min(scores))

env.close()