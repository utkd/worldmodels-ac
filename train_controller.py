import argparse
import gym

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from models import VAE, MDNRNN, NNModule, ActorCritic
from utils.ac_trainer import ActorCriticTrainer, idx_to_action

parser = argparse.ArgumentParser()
parser.add_argument('--vaemodel', type=str, default="saves/zvalues.npz", help="VAE Model location")
parser.add_argument('--rnnmodel', type=str, default="saves/mdnrnn.mdl", help="RNN Model location")
parser.add_argument('--modelout', type=str, default=20, help="Controller model save location")
parser.add_argument('--episodes', type=int, default=1000, help="Number of Episodes to train for")


args = parser.parse_args()

# Hyper params
NUM_ACTIONS = 5 # refer to ac_trainer.py
GAMMA = 0.95
LEARNING_RATE = 0.001
NUM_TRAIN_EPISODES = args.episodes

# Model paths
VAE_MODEL_LOC = args.vaemodel
RNN_MODEL_LOC = args.rnnmodel
CONTROLLER_MODEL_LOC = args.modelout

env = gym.make('CarRacing-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Init models and load saved models
vaemodel = VAE(device=device).to(device)
vaemodel.load_state_dict(torch.load(VAE_MODEL_LOC))
vaemodel.train()

rnnmodel =  MDNRNN(z_dim=32, a_dim=3, hid_dim=64, n_components=5, n_layers=1).to(device)
rnnmodel.load_state_dict(torch.load(RNN_MODEL_LOC))
rnnmodel.train()

actor = NNModule(32+64, 16, NUM_ACTIONS)
critic = NNModule(32, 8, 1)

controller = ActorCritic(actor, critic).to(device)
controller.train()

optimizer = optim.Adam(controller.parameters(), lr=LEARNING_RATE)
trainer = ActorCriticTrainer(controller, optimizer, device, gamma=GAMMA)

print("Loaded and initialized models.")

for ep_idx in range(NUM_TRAIN_EPISODES):
    state = env.reset() 
    episode = []
   
    state = state[:84, :84 ,:]
    action_idx = 3   # accelerate
    done = False
    hidden = rnnmodel.init_hidden(1, device)

    reward_this_episode = 0
    episode_length = 0

    while not done:
        state = torch.FloatTensor(state.copy().transpose((2, 0, 1))).unsqueeze(0).to(device)
        state = state / 255.

        mu, logVar, vision_z_values = vaemodel.encode(state)
        vision_z_values = vision_z_values.unsqueeze(0)
        action_values = torch.FloatTensor([idx_to_action(action_idx)]).unsqueeze(0).to(device)
        rnn_input = torch.cat((vision_z_values, action_values), axis=-1)
        (pi, mu, sigma), y, hidden = rnnmodel(rnn_input, hidden)
        
        action_pred, value_pred = controller(vision_z_values, y)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)

        action_idx = dist.sample()
        log_prob_action = dist.log_prob(action_idx)

        action = idx_to_action(action_idx)

        state, reward, done, _ = env.step(action)
        state = state[:84, :84 ,:]

        reward_this_episode += reward
        episode_length += 1
        episode.append((log_prob_action, value_pred, reward))

    trainer.train_on_episode(episode)

    print("Episode: %d - Length: %d. Total reward: %.2f" % (ep_idx, episode_length, reward_this_episode))

torch.save(controller.state_dict(), CONTROLLER_MODEL_LOC)
