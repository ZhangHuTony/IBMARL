"""
Multi-Agent Behavior Cloning (MABC) is a method for learning policies for multi-agent systems from demonstrations.
The method is based on the idea that a human can provide demos for a single agent, one at a time, and iterate until a desired level of performance is achieved.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path


class MiniBC(torch.nn.Module):
    def __init__(self, agent_id, in_size, out_size, hidden_size=8, hidden_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        # Policy architecture
        layers = []
        layers.append(torch.nn.Linear(in_features=in_size, out_features=hidden_size))
        layers.append(torch.nn.ReLU())
        for i in range(self.hidden_layers - 1):
            layers.append(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(in_features=hidden_size, out_features=out_size))

        self.layers = torch.nn.Sequential(*layers)
        print("LAYERS: ", layers)

        # self.l1 = torch.nn.Linear(in_features=self.in_size, out_features=hidden_size)
        # self.relu1 = torch.nn.ReLU()
        # self.l2 = torch.nn.Linear(in_features=hidden_size, out_features=self.out_size)

    def forward(self, x):
        return self.layers(x)
        # res_l1 = self.l1(x)
        # res_nonlin = self.relu1(res_l1)
        # res_l2 = self.l2(res_nonlin)
        # return res_l2

    def save_checkpoint(self, path):
        """Save policy checkpoint with model parameters and metadata"""
        checkpoint = {
            'state_dict': self.state_dict(),
            'agent_id': self.agent_id,
            'in_size': self.in_size,
            'out_size': self.out_size,
            'hidden_size': self.hidden_size,
            'model_type': 'MiniBC'
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device=None):
        """Load policy checkpoint and return initialized model"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved parameters
        model = cls(
            agent_id=checkpoint['agent_id'],
            in_size=checkpoint['in_size'],
            out_size=checkpoint['out_size'],
            hidden_size=checkpoint['hidden_size']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

class DecentralizedMiniBC(torch.nn.Module):
    def __init__(self, n, in_size, out_size, hidden_size=8, hidden_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.in_size = in_size // n
        self.out_size = out_size // n
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        print("HIDDEN LAYERS: ", hidden_layers)

        # Policies
        for i in range(n):
            setattr(self, f"pi_{i}", MiniBC(i, self.in_size, self.out_size, hidden_size, hidden_layers))

    def forward(self, x):
        # Split the input (concatenated obs) into n parts
        outputs = []
        for i in range(self.n):
            obs_i = None
            if len(x.shape) == 3:
                obs_i = x[:, :, (i * self.in_size):((i + 1) * self.in_size)]
            elif len(x.shape) == 2:
                obs_i = x[:, (i * self.in_size):((i + 1) * self.in_size)]
            pi = getattr(self, f"pi_{i}")
            outputs.append(pi.forward(obs_i).unsqueeze(1))
        ret = torch.concat(outputs, dim=2 if len(x.shape) == 3 else 1)
        return torch.reshape(ret, (x.shape[0], self.out_size * self.n))

    def save_checkpoint(self, path):
        """Save decentralized policy checkpoint with all agent policies"""
        checkpoint = {
            'state_dict': self.state_dict(),
            'n': self.n,
            'in_size': self.in_size * self.n,  # Store total input size
            'out_size': self.out_size * self.n,  # Store total output size
            'hidden_size': self.hidden_size,
            'hidden_layers': self.hidden_layers,
            'model_type': 'DecentralizedMiniBC'
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device=None):
        """Load decentralized policy checkpoint and return initialized model"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved parameters
        model = cls(
            n=checkpoint['n'],
            in_size=checkpoint['in_size'],
            out_size=checkpoint['out_size'],
            hidden_size=checkpoint['hidden_size'],
            hidden_layers=checkpoint['hidden_layers'] if 'hidden_layers' in checkpoint else 2
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

class MABC():
    def __init__(self, scenario_name: str = None, n: int = 3, discrete=False, *args, **kwargs):
        self.scenario_name = scenario_name
        self.translator = TranslatorFactory.create(self.scenario_name)
        self.n = n

        print("INP SIZE: ", n)
        print(self.translator["obs"].raw_size(n) // n)
        act_size = self.translator["act"].raw_size(n) // n if not discrete else 8
        self.policies = [MiniBC(i, self.translator["obs"].raw_size(n) // n, act_size) for i in range(self.n)]
        self.buffers = [[] for _ in range(self.n)]

    def update_buffer(self, agent_id, obs, act):
        for i in range(len(obs)):
            self.buffers[agent_id].append([obs[i], act[i]])

    def train_policy(self, agent_id, max_epochs=15):
        i = agent_id
        optimizer = torch.optim.Adam(self.policies[i].parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        ds = BCDataset(np.array(self.buffers)[i, :, 0], np.array(self.buffers)[i, :, 1])
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        for epoch in range(max_epochs):
            losses = []
            for batch, (s, a_true) in enumerate(dl):
                optimizer.zero_grad()
                a_hat = torch.clamp(self.policies[i].forward(s), -1, 1)
                loss = loss_fn(a_hat, a_true)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            print(f"Epoch {epoch}, loss: {np.mean(losses)}")

    def act(self, obs):
        """
        Given the observations, return the actions for each agent.
        """
        actions = []
        for i in range(self.n):
            obs_i = torch.tensor(obs[i])
            act = self.policies[i].forward(obs_i).detach()
            act = torch.clamp(act, -1, 1)
            actions.append(act.numpy())
        return actions