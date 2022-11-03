import torch
import torch.nn.functional as F

def idx_to_action(action_idx):
    ACTION = [
        [-1.0, 0.0, 0.0],   # Turn left
        [1.0, 0.0, 0.0],    # Turn right
        [0.0, 0.0, 0.8],    # Brake
        [0.0, 1.0, 0.0],    # Acclerate
        [0.0, 0.0, 0.0]     # Do nothing
    ]

    return ACTION[action_idx]

class ActorCriticTrainer(object):

    def __init__(self, model, optimizer, device, gamma=0.95):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
    
    def train_on_episode(self, episode):
        log_prob_actions = []
        values = []
        rewards = []

        for log_prob_act, value_pred, reward in episode:
            log_prob_actions.append(log_prob_act)
            values.append(value_pred)
            rewards.append(reward)
        
        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1).squeeze(-1)
            
        returns = self._calculate_returns(rewards).to(self.device)
        policy_loss, value_loss = self._update_policy(returns, log_prob_actions, values)

        return policy_loss, value_loss
    
    def _calculate_returns(self, rewards):
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r +  R * self.gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # normalize
        returns = (returns - returns.mean()) / returns.std()
        return returns

    def _update_policy(self, returns, log_prob_actions, values):
        returns = returns.detach()

        policy_loss = - (returns * log_prob_actions).sum()
        value_loss = F.mse_loss(values, returns).sum()

        self.optimizer.zero_grad()
        
        policy_loss.backward(retain_graph=True)
        value_loss.backward()
        
        self.optimizer.step()
        return policy_loss.item(), value_loss.item()