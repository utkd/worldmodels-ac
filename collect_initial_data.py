import argparse
import numpy as np
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--rollouts', type=int, default=100, help="Number of random rollouts")
parser.add_argument('--output', type=str, default="saves/samples.npz", help="Location of the output file")
args = parser.parse_args()

OUTPUT_FILE = args.output
COLLECTION_SIZE = args.rollouts

ACTIONS = [
    [-1.0, 0.0, 0.0],   # Turn left
    [1.0, 0.0, 0.0],    # Turn right
    [0.0, 0.0, 0.8],    # Brake
    [0.0, 1.0, 0.0],    # Acclerate
    [0.0, 0.0, 0.0]     # Do nothing
]

env = gym.make('CarRacing-v0')

episode_samples = []
total_sample_size = 0

for i in range(COLLECTION_SIZE):
    obs = env.reset()
    done = False
    
    # Take random actions until done
    episode_len = 0
    episode = []
    while not done:
        random_action = ACTIONS[np.random.randint(5)]
        episode.append((obs[:84, :84 ,:], random_action))
        new_obs, reward, done, info = env.step(random_action)

        episode_len += 1
        obs = new_obs
    total_sample_size += episode_len
    episode_samples.append(episode)

print("Collected", total_sample_size, "samples.")
episode_samples = np.array(episode_samples)
np.savez_compressed(OUTPUT_FILE, episode_samples=episode_samples)