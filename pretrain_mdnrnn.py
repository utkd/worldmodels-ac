import argparse
import numpy as np
import torch
from torch.optim import Adam
from models import MDNRNN
from utils.losses import criterion

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="saves/zvalues.npz", help="Location of saved Z values")
parser.add_argument('--modelout', type=str, default="saves/mdnrnn.mdl", help="Model save location")
parser.add_argument('--epochs', type=int, default=20, help="Number of Epochs")

args = parser.parse_args()

DATA_FILE = args.input
BATCH_SIZE = 1 # each episode is batch due to the generate_episode_sequences function
EPOCHS = args.epochs
LR = 0.0001
SEQLEN = 999

def extract_episodes(z, a, lens):
    z = np.array(z)
    a = np.array(a)

    episodes = []
    start_idx = 0
    for l in lens:
        episode = []
        for idx in range(start_idx, start_idx+l):
            episode.append((z[idx], a[idx]))
        episodes.append(episode)
        start_idx = start_idx + l
    
    return episodes

def generate_episode_sequences(episode):
    episode_len = len(episode)
    episode_inputs = []
    episode_targets = []

    for idx in range(episode_len - 1):
        episode_inputs.append( np.concatenate( [episode[idx][0], episode[idx][1]]) )
        episode_targets.append(episode[idx+1][0])
    
    episode_inputs = np.array(episode_inputs)
    episode_targets = np.array(episode_targets)
    return episode_inputs, episode_targets

samples = None
print("Reading data for MDN training..")
loaded = np.load(DATA_FILE, allow_pickle=True)
z_values = loaded['z_values']
actions = loaded['actions']
lengths = loaded['lengths']

print("Read", len(z_values), "samples for training.")
episodes = extract_episodes(z_values, actions, lengths)
print("Extracted", len(episodes), "episodes.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

rnnmodel = MDNRNN(z_dim=32, a_dim=3, hid_dim=64, n_components=5, n_layers=1).to(device)
optimizer = torch.optim.Adam(rnnmodel.parameters(), lr=LR)

# Train the mdl
for epoch in range(EPOCHS):
    
    epoch_loss = 0.
    
    for episode in episodes:

        episode_inputs, episode_targets = generate_episode_sequences(episode)
        episode_inputs = torch.FloatTensor(episode_inputs).unsqueeze(0).to(device)
        episode_targets = torch.FloatTensor(episode_targets).unsqueeze(0).to(device)

        inputs = episode_inputs
        targets = episode_targets

        # Forward pass
        hidden = rnnmodel.init_hidden(BATCH_SIZE, device)
        (pi, mu, sigma), y, hidden = rnnmodel(inputs, hidden)
        
        loss = criterion(targets, pi, mu, sigma)
        epoch_loss += loss.item()

        # Backward and optimize
        rnnmodel.zero_grad()
        loss.backward()
        optimizer.step()
    
    print ('Epoch [{}], Loss: {:.4f}'.format(epoch, epoch_loss))

torch.save(rnnmodel.state_dict(), args.modelout)
print("RNN Model saved.")