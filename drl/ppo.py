import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.distributions.normal import Normal
import gymnasium as gym
import sys
import os

from sinergym.utils.wrappers import WeatherForecastingWrapper, NormalizeAction
from torch.utils.tensorboard import SummaryWriter
import datetime


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_param.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class PPOAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            action_low,
            action_high,
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,
            lr=3e-4,
            lr_decay=0.99,
            train_iters=80,
            mini_batch_size=64,
            max_grad_norm=0.5,
            hidden_sizes=(64, 64),
            device='cpu',
            scheduler_type='exponential'
    ):
        self.device = device

        # action scaling for env 
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.value_fn = ValueNetwork(state_dim, hidden_sizes).to(self.device)

        # initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=lr)

        # initialize schedulers
        if scheduler_type.lower() == 'exponential':
            self.policy_scheduler = ExponentialLR(self.policy_optimizer, lr_decay)
            self.value_scheduler = ExponentialLR(self.value_optimizer, lr_decay)
        elif scheduler_type.lower() == 'linear':
            # add a decay function for the learning rate
            def linear_lambda(epoch):
                return 1 - epoch * (1 - lr_decay)  # for experimentation
            self.policy_scheduler = LambdaLR(self.policy_optimizer, lr_lambda=linear_lambda)
            self.value_scheduler = LambdaLR(self.value_optimizer, lr_lambda=linear_lambda)
        else:
            raise ValueError("Unsupported scheduler type: {}".format(scheduler_type))

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mean, std = self.policy(state)
        dist_ = Normal(mean, std)
        raw_action = dist_.sample()
        log_prob = dist_.log_prob(raw_action).sum(-1)

        # squish action space
        action = torch.tanh(raw_action)

        # and rescale
        scaled_action = self.scale_action(action)

        return scaled_action.cpu().numpy(), log_prob.detach().cpu().numpy(), raw_action.detach(), action.detach()

    def scale_action(self, action):
        # normalize, like rescale, action to environment.
        return self.action_low + (0.5 * (action + 1.0) * (self.action_high - self.action_low))

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        # work backwards to apply gae. 
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * (1 - dones[step]) * next_value - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]

        return advantages

    def update(self, buffer):
        """
        buffer contains trajectories per paper.
        pre-tanh states and actions, old_log_probs, returns, advantages.
        """
        states = torch.as_tensor(np.array(buffer['states']), dtype=torch.float32, device=self.device)
        old_actions = torch.as_tensor(np.array(buffer['raw_actions']), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.array(buffer['log_probs']), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(buffer['advantages'], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(buffer['returns'], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = len(states)

        for _ in range(self.train_iters):
            # shuffle for mini batch
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_old_actions = old_actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # update distribution
                mean, std = self.policy(batch_states)
                dist_ = Normal(mean, std)

                # make new log probs that align with previous distribution
                new_log_probs = dist_.log_prob(batch_old_actions).sum(axis=-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # compute loss
                value_pred = self.value_fn(batch_states).squeeze(-1)
                value_loss = ((value_pred - batch_returns) ** 2).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

    def collect_trajectories(self, env, num_steps, max_timesteps_per_episode=7000):
        """
        useful for collecting trajectories to their extent under the current policy
        """
        obs, _ = env.reset()
        buffer = {
            'states': [],
            'actions': [],
            'raw_actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

        episode_rewards = []
        episode_reward = 0
        timesteps = 0

        for _ in range(num_steps):
            with torch.no_grad():
                state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                value = self.value_fn(state_tensor.unsqueeze(0)).item()

            action, log_prob, raw_action, _ = self.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer['states'].append(obs)
            buffer['raw_actions'].append(raw_action.cpu().numpy())  # store raw pre-tanh action
            buffer['actions'].append(action)
            buffer['log_probs'].append(log_prob.item())
            buffer['rewards'].append(reward)
            buffer['values'].append(value)
            buffer['dones'].append(float(done))

            obs = next_obs
            episode_reward += reward
            timesteps += 1

            if done or timesteps == max_timesteps_per_episode:
                obs, _ = env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0
                timesteps = 0

        # Compute advantages using GAE
        with torch.no_grad():
            state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            next_value = self.value_fn(state_tensor.unsqueeze(0)).item()

        advantages = self.compute_gae(
            rewards=buffer['rewards'],
            values=buffer['values'],
            dones=buffer['dones'],
            next_value=next_value
        )

        returns = [adv + val for adv, val in zip(advantages, buffer['values'])]

        buffer['advantages'] = advantages
        buffer['returns'] = returns

        return buffer, np.mean(episode_rewards)


# PPO Training Loop
def train_ppo(env, agent, num_epochs, steps_per_epoch, save_interval, model_save_path, writer, eval_interval=10,
              eval_steps=2000):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        buffer, avg_reward = agent.collect_trajectories(env, steps_per_epoch)
        agent.update(buffer)

        writer.add_scalar("Training/Average_Reward", avg_reward, epoch)

        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(model_save_path, f"ppo_epoch_{epoch + 1}.pth")
            torch.save({
                'policy_state_dict': agent.policy.state_dict(),
                'value_state_dict': agent.value_fn.state_dict()
            }, model_path)
            print(f"Model saved at {model_path}")

        if (epoch + 1) % eval_interval == 0:
            print("Starting Evaluation")
            eval_reward = evaluate_agent(env, agent, eval_steps)
            writer.add_scalar("Evaluation/Average_Reward", eval_reward, epoch)
            print("Ending Evaluation")

    writer.close()


# evaluate the agent
def evaluate_agent(env, agent, num_steps):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(num_steps):
        with torch.no_grad():
            action, _, _, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    return total_reward / num_steps


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(device)

    # Hyperparameter grids
    # lr_list = [1e-4, 3e-4, 1e-5]
    # scheduler_list = ['exponential', 'linear']
    # hidden_layers_list = [2, 3, 4]
    # hidden_nodes_list = [64, 128, 256]
    # batch_size_list = [64, 128]
    # clip_ratio_list = [0.1, 0.2, 0.3]
    # forecasting_list = [0, 3, 5]

    experiments = [
        (0.0003, 'exponential', 2, 128, 0.2, 0),
        (0.0003, 'exponential', 2, 128, 0.2, 3),
        (0.0003, 'exponential', 2, 128, 0.3, 0),
        (0.0003, 'exponential', 2, 128, 0.3, 3),
        (0.0003, 'exponential', 3, 64, 0.1, 0),
        (0.0003, 'exponential', 3, 64, 0.1, 3),
        (0.0003, 'exponential', 3, 64, 0.2, 0),
        (0.0003, 'exponential', 3, 64, 0.2, 3),
    ]

    # training parameters, extensive. 
    num_epochs = 50
    batch_size = 64
    steps_per_epoch = 6624
    save_interval = 10
    eval_steps = 6624

    # where are we?
    model_base_path = "./ppo_models"
    os.makedirs(model_base_path, exist_ok=True)

    for lr, scheduler_type, num_layers, num_nodes, clip_ratio, forecasting in experiments:
        # make tuple for intermediate layers
        hidden_sizes = tuple([num_nodes for _ in range(num_layers)])

        # Environment setup
        os.environ['EPLUS_PATH'] = '/Applications/EnergyPlus-24-2-0'
        sys.path.append(os.environ['EPLUS_PATH'])

        env = gym.make("Eplus-datacenter-mixed-continuous-stochastic-v1",
                       config_params={
                           'timesteps_per_hour': 3,
                           'runperiod': (1, 6, 1991, 31, 8, 1991)
                       })
        if forecasting > 0:
            env = WeatherForecastingWrapper(env, n=forecasting)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high

        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            lr=lr,
            clip_ratio=clip_ratio,
            mini_batch_size=batch_size,
            hidden_sizes=hidden_sizes,
            scheduler_type=scheduler_type
        )

        # make a directory so we can save all our model, graphs, and other output uniquely.
        # important for github
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = (f"runs/lr_{lr}_sched_{scheduler_type}_layers_{num_layers}_nodes_{num_nodes}_"
                   f"batch_{batch_size}_clip_{clip_ratio}_forecast_{forecasting}_{timestamp}")
        writer = SummaryWriter(log_dir=log_dir)

        # save
        run_save_path = os.path.join(model_base_path,
                                     f"lr_{lr}_sched_{scheduler_type}_layers_{num_layers}_nodes_{num_nodes}_batch_{batch_size}_clip_{clip_ratio}_forecast_{forecasting}")
        os.makedirs(run_save_path, exist_ok=True)

        # Train
        train_ppo(
            env=env,
            agent=agent,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            save_interval=save_interval,
            model_save_path=run_save_path,
            eval_interval=5,
            eval_steps=eval_steps,
            writer=writer
        )


if __name__ == "__main__":
    main()