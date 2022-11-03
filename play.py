import time
import argparse
import gym
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributions as distributions

from models import VAE, MDNRNN, NNModule, ActorCritic
from utils.ac_trainer import idx_to_action

parser = argparse.ArgumentParser()
parser.add_argument('--vaemodel', type=str, default="saves/zvalues.npz", help="VAE Model location")
parser.add_argument('--rnnmodel', type=str, default="saves/mdnrnn.mdl", help="RNN Model location")
parser.add_argument('--controllermodel', type=str, default="saves/controller.mdl", help="Controller model location")

args = parser.parse_args()

# Hyper params
NUM_ACTIONS = 5 # refer to ac_trainer.py

# Model paths
VAE_MODEL_LOC = args.vaemodel
RNN_MODEL_LOC = args.rnnmodel
CONTROLLER_MODEL_LOC = args.controllermodel

env = gym.make('CarRacing-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Init models and load saved models
vaemodel = VAE(device=device).to(device)
vaemodel.load_state_dict(torch.load(VAE_MODEL_LOC))
vaemodel.eval()

rnnmodel =  MDNRNN(z_dim=32, a_dim=3, hid_dim=64, n_components=5, n_layers=1).to(device)
rnnmodel.load_state_dict(torch.load(RNN_MODEL_LOC))
rnnmodel.eval()

actor = NNModule(32+64, 16, NUM_ACTIONS)
critic = NNModule(32, 8, 1)

controller = ActorCritic(actor, critic).to(device)
controller.load_state_dict(torch.load(CONTROLLER_MODEL_LOC))

state = env.reset() 
rewards = []
cumu_rewards = []

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
    env.render(mode='human')
    time.sleep(0.1)
    state = state[:84, :84 ,:]

    reward_this_episode += reward
    rewards.append(reward)
    cumu_rewards.append(reward_this_episode)
    episode_length += 1

plt.plot(rewards)
plt.plot(cumu_rewards)
plt.show()
print("Total rewards:", reward_this_episode)
    