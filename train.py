import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=200000)

model.save("cartpole_dqn")

print("Training completed!")
