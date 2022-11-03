import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from models import VAE
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="saves/samples.npz", help="Location of rollouts output")
parser.add_argument('--modelout', type=str, default="saves/vae.mdl", help="Model save location")
parser.add_argument('--output', type=str, default="saves/zvalues.npz", help="Location of the zvalues corresponding to the rollouts")
parser.add_argument('--batchsize', type=int, default=256, help="Batch size")
parser.add_argument('--epochs', type=int, default=10, help="Number of Epochs")

args = parser.parse_args()

DATA_FILE = args.input
BATCH_SIZE = args.batchsize
EPOCHS = args.epochs
LR = 0.0001

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def compute_losses(x, x_hat, mu, logsigma):
    recon = F.mse_loss(x_hat, x, size_average=False)
    kld = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - logsigma.exp().pow(2))
    return recon, kld

if __name__ == "__main__":
    print("Reading data..")
    loaded = np.load(DATA_FILE, allow_pickle=True)
    episode_samples = loaded['episode_samples']
    
    samples = []
    actions = []
    lengths = []
    for episode in episode_samples:
        lengths.append(len(episode))
        for state, action in episode:
            samples.append(state)
            actions.append(action)

    samples = np.array(samples)
    print("Read", len(samples), "samples for training.")
    samples = samples.transpose((0, 3, 1, 2))
    print("Samples shape:", samples.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ImageDataset(samples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(device=device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=LR)
    losses = []

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.
        for i, batch in enumerate(dataloader):
            x = torch.tensor(batch, device=device, dtype=torch.float32)
            x = x / 255.

            optimizer.zero_grad()
            x_hat, mu, logsigma, z = model(x)
            recon_loss, kld = compute_losses(x, x_hat, mu, logsigma)

            loss = recon_loss + kld
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", total_loss / ((i+1) * BATCH_SIZE))
        losses.append( total_loss / ((i+1) * BATCH_SIZE) )

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    torch.save(model.state_dict(), args.modelout)
    print("Model saved. Computing z_values ..")

    model.eval()

    z_values = []
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for i, batch in enumerate(dataloader):
        x = torch.tensor(batch, device=device, dtype=torch.float32)
        x = x / 255.

        optimizer.zero_grad()
        x_hat, mu, logVar, z = model(x)
        z_values.append(z.cpu().detach().numpy())

    z_values = np.concatenate(z_values, axis=0)
    print("Z values shape:", z_values.shape, ". Saving data ..")
    np.savez_compressed(args.output, z_values=z_values, actions=actions, lengths=lengths)