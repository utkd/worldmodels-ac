import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, device, channels=3, n_hid=2304, n_z=32):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(n_hid, n_z)
        self.fc2 = nn.Linear(n_hid, n_z)
        self.fc3 = nn.Linear(n_z, n_hid)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (2304, 1, 1)),
            # output_size = strides * (input_size-1) + kernel_size - 2*padding
            nn.ConvTranspose2d(n_hid, 256, kernel_size=3, stride=2), # 3 x 3 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), # 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), # 18 x 18
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), # 40 x 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=6, stride=2), # 84 x 84
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc1(x)
        logsigma = self.fc2(x)
        z = self.reparameterize(mu, logsigma)
        return mu, logsigma, z
    
    def reparameterize(self, mu, logsigma):
        std = logsigma.exp()
        eps = torch.randn(*mu.size(), device=self.device)
        z = mu + std * eps
        return z

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logVar, z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, logVar, z

class MDNRNN(nn.Module):
    def __init__(self, z_dim, a_dim, hid_dim=128, n_components=3, n_layers=1):
        super(MDNRNN, self).__init__()

        self.inp_dim = z_dim + a_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.n_components = n_components
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(self.inp_dim, hid_dim, n_layers, batch_first=True)

        self.fc1 = nn.Linear(hid_dim, n_components)
        self.fc2 = nn.Linear(hid_dim, n_components*z_dim)
        self.fc3 = nn.Linear(hid_dim, n_components*z_dim)
    
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, logsigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_components, 1)
        mu = mu.view(-1, rollout_length, self.n_components, self.z_dim)
        logsigma = logsigma.view(-1, rollout_length, self.n_components, self.z_dim)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(logsigma)
        return pi, mu, sigma
        
    def forward(self, x, h):
        # Forward propagate LSTM
        y, states = self.rnn(x, h)   # states is the final hidden output for rnn or (hidden, cell) in lstm
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), y, states

    def init_hidden(self, bsz, device):
        return (torch.zeros(self.n_layers, bsz, self.hid_dim).to(device),
                    torch.zeros(self.n_layers, bsz, self.hid_dim).to(device))

class NNModule(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dropout=0.25):
        super(NNModule, self).__init__()
        self.hid1 = nn.Linear(n_inp, n_hid)
        self.hid2 = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = self.hid1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.hid2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def forward(self, z_state, h_state):
        action_pred = self.actor(torch.cat((z_state, h_state), axis=-1))
        value_pred = self.critic(z_state)
        return action_pred, value_pred