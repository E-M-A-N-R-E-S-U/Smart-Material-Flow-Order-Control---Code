import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.distributions.categorical import Categorical
import os


class Sequence:

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.terminal_states = []

    def __len__(self):
        return len(self.states)

    def set(self, state, action, reward, log_prob, state_value, terminal_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.terminal_states.append(terminal_state)

    def get(self):
        return self.states, self.actions, self.rewards, self.log_probs, self.state_values, self.terminal_states

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.terminal_states = []


class Actor(nn.Module):

    def __init__(self, n_states, n_actions, lr=0.0002, model=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states

        self.model = nn.Sequential(nn.Linear(n_states, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, n_actions)
                                   )
        if model:
            self.model.load_state_dict(torch.load(model))
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state, masked):
        # mask = np.array((state.shape[0], self.n_actions))
        mask = []
        if masked:
            features_per_order = int(self.n_states / self.n_actions)
            start = 0
            if state.ndim == 2:
                for id_of_element in range(state.shape[0]):
                    mask_element = []
                    start = 0
                    for i in range(self.n_actions):
                        stop = start+features_per_order
                        features_of_order = state[id_of_element][start:stop]
                        if np.count_nonzero(features_of_order) > 0:
                            mask_element.append(1.0)
                        else:
                            mask_element.append(0.0)
                        start += features_per_order
                    mask.append(mask_element)
            else:
                for i in range(self.n_actions):
                    stop = start + features_per_order
                    features_of_order = state[start:stop]
                    if np.count_nonzero(features_of_order) > 0:
                        mask.append(1.0)
                    else:
                        mask.append(0.0)
                    start += features_per_order
        else:
            if state.ndim == 2:
                for id_of_element in range(state.shape[0]):
                    mask_element = [1.0 for _ in range(self.n_actions)]
                    mask.append(mask_element)
            else:
                mask = [1.0 for _ in range(self.n_actions)]

        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        if isinstance(state, np.ndarray):
            state = torch.tensor([state], dtype=torch.float).to(self.device)

        logits = self.model(state)
        masked = logits + (mask == 0) * torch.tensor(-1e9).to(self.device)
        probs = nn.functional.softmax(masked, dim=-1)
        distribution = Categorical(probs=probs)
        return distribution

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))


class Critic(nn.Module):

    def __init__(self, n_states, lr=0.0005, model=None):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(n_states, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1))
        if model:
            self.model.load_state_dict(torch.load(model))
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        # state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        if isinstance(state, np.ndarray):
            state = torch.tensor([state], dtype=torch.float).to(self.device)

        state_value = self.model(state)
        return state_value

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))


class Agent:

    def __init__(self, n_actions, n_states, batch_size, alpha, beta, gamma, lambd, clip, actor_model=None,
                 critic_model=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self.policy_clip = clip

        self.batch_size = batch_size

        self.actor = Actor(n_states, n_actions, alpha, model=actor_model)
        self.critic = Critic(n_states, beta, model=critic_model)

        self.sequence = Sequence()

    def save_model(self, stage, episode):
        actor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Model_Stage_{stage}", f"ActorEP{episode}.pth")
        self.actor.save_model(file_path=actor_path)
        critic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Model_Stage_{stage}", f"CriticEP{episode}.pth")
        self.critic.save_model(file_path=critic_path)

    def get_action(self, state, masked=False):
        distribution = self.actor(state, masked)
        action = distribution.sample()

        log_prob = distribution.log_prob(action)
        log_prob = torch.squeeze(log_prob).item()
        action = torch.squeeze(action).item()

        return action, log_prob

    def get_state_value(self, state):
        state_value = self.critic(state)
        state_value = torch.squeeze(state_value).item()

        return state_value

    def compute_gae(self, rewards, state_values, terminal_states):
        advantages = []
        last_advantage = 0

        for t in reversed(range(len(self.sequence))):
            if t + 1 < len(self.sequence):
                delta = rewards[t] + self.gamma * state_values[t+1] * (1 - int(terminal_states[t+1])) - state_values[t]
            else:
                delta = rewards[t] - state_values[t]

            advantage = delta + self.gamma * self.lambd * (1 - int(terminal_states[t])) * last_advantage
            last_advantage = advantage
            advantages.insert(0, advantage)

        return advantages

    @staticmethod
    def compute_returns(advantages, state_values):
        returns = advantages + state_values
        return returns

    def perform_training(self, training_episodes, masked=False):
        states, actions, rewards, log_probs_old, state_values, terminal_states = self.sequence.get()

        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        old_log_probs = torch.tensor(log_probs_old, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)

        advantages = self.compute_gae(rewards, state_values, terminal_states)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.actor.device)
        advantages = ((advantages - advantages.mean()) / advantages.std() + 1e-8)
        state_values = torch.tensor(state_values, dtype=torch.float).to(self.actor.device)

        returns = self.compute_returns(advantages, state_values)
        returns = torch.tensor(returns, dtype=torch.float).to(self.actor.device)

        dataset = TensorDataset(states, actions, returns, advantages, old_log_probs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(training_episodes):
            for batch in dataloader:
                states_, actions_, returns_, advantages_, old_log_probs_ = batch
                distribution = self.actor(states_, masked=masked)
                current_log_probs = distribution.log_prob(actions_)
                prob_ratios = current_log_probs.exp()/old_log_probs_.exp()
                weighted_probs = prob_ratios * advantages_
                weighted_clipped_probs = torch.clamp(prob_ratios, 1-self.policy_clip, 1+self.policy_clip) * advantages_
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_value = self.critic(states_)
                critic_value = torch.squeeze(critic_value)
                critic_loss = (returns_-critic_value)**2
                critic_loss = critic_loss.mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

        self.sequence.reset()
