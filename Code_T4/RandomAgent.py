# RandomAgent.py
import random

import gym

from FeatureExtractor import FeatureExtractor

env = gym.make("MountainCar-v0", render_mode="human")
feature_extractor = FeatureExtractor()
observation, info = env.reset()
obs_features = feature_extractor.get_features(observation)
for _ in range(1000):
    action = random.choice([0, 1, 2])
    observation, reward, terminated, truncated, info = env.step(action)
    obs_features = feature_extractor.get_features(observation)

    if terminated or truncated:
        observation, info = env.reset()
        obs_features = feature_extractor.get_features(observation)

env.close()
