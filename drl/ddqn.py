import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import sys
import gymnasium as gym

from sinergym.utils.wrappers import DiscretizeEnv, WeatherForecastingWrapper
from torch.utils.tensorboard import SummaryWriter


class DDQN(nn.Module):

    def __init__(self, state_dim, n_actions, hidden_size=64):
        super(DDQN, self).__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Shared input processing
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Head that outputs V(s)
        self.value_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Head that computes Advantages
        self.advantage_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        # Cast if Needed
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        # Ensure dimensions are lined up for being a batch
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        feature_values = self.feature_layers(x)
        V_s = self.value_layers(feature_values)
        advantage = self.advantage_layers(feature_values)

        q_values = V_s + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class ReplayMemory:
    def __init__(self, capacity=1_000_000, storage_device='cpu', model_device='cpu'):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.storage_device = storage_device
        self.model_device = model_device

    def insert(self, transition):
        # Transitions are (s, a, r, t, s')
        transition = [item.to(self.storage_device) for item in transition]
        self.memory.append(transition)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        return [torch.cat(items).to(self.model_device) for items in batch]


class DDQNAgent:
    def __init__(self, state_dim, n_actions, eps_init=1.0, eps_min=0.1, eps_decay_steps=50_000, hidden_size=64,
                 memory_capacity=1_000_000, lr=1e-4, forecasting=0, gamma=0.99, device='cpu'):
        self.device = device
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma

        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_decay = (self.eps - self.eps_min) / eps_decay_steps

        self.model = DDQN(state_dim, n_actions, hidden_size).to(self.device)
        self.target_model = DDQN(state_dim, n_actions, hidden_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # I ran into issues storing 1 million transitions on my laptop. This slows down the process, but makes it
        # feasible to train...
        self.memory = ReplayMemory(capacity=memory_capacity, storage_device='cpu', model_device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.log_dir = f"./logs/ddqn_lr{lr}_decay{eps_decay_steps}_forecasting{forecasting}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.step_count = 0  # For the x-axis of Tensorboard

    def select_action(self, state, env):
        if random.random() < self.eps:
            return env.action_space.sample()
        else:
            # Return the best action, as dictated by the model
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                action = torch.argmax(self.model(state_tensor)).item()
                return action

    def store_transition(self, state, action, reward, done, next_state):
        transition = (
            torch.FloatTensor(state).unsqueeze(0),
            torch.tensor([[action]]),
            torch.tensor([[reward]], dtype=torch.float),
            torch.tensor([[done]], dtype=torch.bool),
            torch.FloatTensor(next_state).unsqueeze(0)
        )
        self.memory.insert(transition)

    def train(self, batch_size):
        if not self.memory.can_sample(batch_size):
            return None

        states, actions, rewards, dones, next_states = self.memory.sample(batch_size)

        # We want to save the torch grad graph when computing the Q values
        q_values = self.model(states).gather(1, actions)

        # But not with the target model
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=-1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (~dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_agent(env, agent, n_episodes, solved_score=-12_300.0, update_every=10, save_every=100, batch_size=32,
                model_save_path='./ddqn_datacenter'):

    returns = deque(maxlen=100)

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action = agent.select_action(state, env)
            next_state, reward, terminal, truncated, _ = env.step(action)
            done = terminal or truncated

            agent.store_transition(state, action, reward, done, next_state)

            loss = agent.train(batch_size)
            if loss is not None:  # loss is only None when batch size is too small
                agent.writer.add_scalar("Training/loss", loss, agent.step_count)
                agent.step_count += 1

            state = next_state
            ep_return += reward
            agent.step_count += 1

            # Epsilon decay
            agent.eps = max(agent.eps_min, agent.eps - agent.eps_decay)

        returns.append(ep_return)
        agent.writer.add_scalar("Training/Episode_Return", ep_return, episode)
        agent.writer.add_scalar("Training/Epsilon", agent.eps, episode)

        solved = np.mean(returns) >= solved_score and len(returns) == 100
        if solved:
            break

        if episode % update_every == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if episode % save_every == 0:
            os.makedirs(model_save_path, exist_ok=True)
            # remove "./logs/" from the log dir. It's parameterized to save unique files
            model_file = f"{agent.log_dir[len('./logs/'):]}_episode_{episode}.pth"
            model_path = os.path.join(model_save_path, model_file)
            torch.save(agent.model.state_dict(), model_path)


def test_agent(env, agent, num_episodes):
    old_eps = agent.eps
    agent.eps = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                action = torch.argmax(agent.model(state_tensor)).item()
            state, reward, terminal, truncated, _ = env.step(action)
            done = terminal or truncated
            ep_reward += reward

        agent.writer.add_scalar("Test/Episode_Return", ep_reward, episode)

    # Reset the agent epsilon to whatever it was prior to testing
    agent.eps = old_eps

def make_discrete_env(env):
    num_bins = 5
    low = env.action_space.low
    high = env.action_space.high

    # Create discrete sets of actions for each dimension
    dim_actions = [np.linspace(low[i], high[i], num_bins) for i in range(len(low))]

    # Create a grid of all possible actions (Cartesian product)
    # For example, if we have 2 dimensions and 5 bins per dimension,
    # that yields 25 discrete actions.
    if len(dim_actions) == 1:
        # If there's only one dimension
        action_grid = [(x,) for x in dim_actions[0]]
    else:
        # For multiple dimensions, we can use a nested comprehension or itertools.product
        from itertools import product
        action_grid = list(product(*dim_actions))

    discrete_space = gym.spaces.Discrete(len(action_grid))

    def action_mapping(discrete_action):
        return list(action_grid[discrete_action])

    env = DiscretizeEnv(env, discrete_space=discrete_space, action_mapping=action_mapping)
    return env


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    os.environ['EPLUS_PATH'] = '/Applications/EnergyPlus-24-2-0'
    sys.path.append(os.environ['EPLUS_PATH'])

    SOLVED_THRESHOLD = -12200.0
    TRAIN_EPISODES = 300
    TEST_EPISODES = 30
    FORECASTING = 0

    env = gym.make('Eplus-datacenter-mixed-continuous-stochastic-v1',
                   config_params={
                       'timesteps_per_hour': 3,
                       'runperiod': (1, 6, 1991, 31, 8, 1991)
                   })

    if FORECASTING > 0:
        env = WeatherForecastingWrapper(env, n=FORECASTING)

    env = make_discrete_env(env)

    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        eps_init=1.0,
        eps_min=0.1,
        eps_decay_steps=50_000,
        hidden_size=64,
        memory_capacity=1_000_000,
        lr=1e-4,
        forecasting=0,
        gamma=0.99,
        device='cpu'
    )

    train_agent(
        env=env,
        agent=agent,
        n_episodes=TRAIN_EPISODES,
        solved_score=SOLVED_THRESHOLD,
        update_every=10,
        save_every=100,
        batch_size=32
    )

    test_agent(
        env,
        agent,
        TEST_EPISODES
    )

    agent.writer.close()


if __name__ == "__main__":
    main()
