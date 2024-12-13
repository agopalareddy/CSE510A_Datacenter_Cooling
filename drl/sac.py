import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
import os
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter
import sinergym


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.network = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        # Q-network takes state and action as input
        layers = []
        prev_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=64, device='cpu'):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        )


class SACAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            action_low,
            action_high,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            lr=3e-4,
            lr_decay=0.99,
            hidden_sizes=(64, 64),
            device='cpu',
            learn_alpha=False,
            target_entropy=None
    ):
        self.device = device

        # rescale for the action for the environemnt
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.learn_alpha = learn_alpha
        self.hidden_sizes = hidden_sizes

        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        # init nets
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # init opts
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # for dynamic entropy learning
        if self.learn_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # scheduling mgmt
        self.policy_scheduler = ExponentialLR(self.policy_optimizer, lr_decay)
        self.q1_scheduler = ExponentialLR(self.q1_optimizer, lr_decay)
        self.q2_scheduler = ExponentialLR(self.q2_optimizer, lr_decay)
        if self.learn_alpha:
            self.alpha_scheduler = ExponentialLR(self.alpha_optimizer, lr_decay)

    def select_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            with torch.no_grad():
                mean, log_std = self.policy.forward(state)
                # deterministic = just mean action
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _, _, _, _ = self.policy.sample(state)

        scaled_action = self.scale_action(action)
        return scaled_action.squeeze(0).cpu().numpy()

    def scale_action(self, action):
        # scale action for normalization for environemnt
        return self.action_low + (0.5 * (action + 1.0) * (self.action_high - self.action_low))

    def update(self, replay_buffer, batch_size=64):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device=self.device)

        with torch.no_grad():
            # update for next action 
            next_action, next_log_prob, _, _, _ = self.policy.sample(next_states)
            # figure out q values 
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards + self.gamma * (1 - dones) * q_next

        # update q
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # update policy
        new_action, log_prob, _, _, _ = self.policy.sample(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update alpha appropriately 
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # perform soft updates 
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# obv
def evaluate_agent(env, agent, num_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(num_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    return total_reward / num_steps

# train sac 
def train_sac(env, agent, replay_buffer, writer, num_epochs, steps_per_epoch, start_steps=10000, update_after=1000, update_every=50, batch_size=64, save_interval=10, model_save_path="./sac_models", eval_interval=10, eval_steps=2000, run_name=""):
    os.makedirs(model_save_path, exist_ok=True)
    total_steps = num_epochs * steps_per_epoch
    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    for t in range(1, total_steps + 1):
        # continuously take random actions 
        if t < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.store(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            episode_reward = 0

        # update agent t-periorically 
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                agent.update(replay_buffer, batch_size)

        # end the epoch
        if t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0
            print(f"Epoch {epoch}/{num_epochs}, Steps: {t}, Avg Reward (last 10 eps): {avg_reward:.2f}")
            writer.add_scalar("Training/Average_Reward", avg_reward, epoch)

            # save perioritically 
            if epoch % save_interval == 0:
                model_path = os.path.join(model_save_path, f"{run_name}_epoch_{epoch}.pth")
                torch.save({
                    'policy_state_dict': agent.policy.state_dict(),
                    'q1_state_dict': agent.q1.state_dict(),
                    'q2_state_dict': agent.q2.state_dict(),
                    'alpha': agent.alpha
                }, model_path)
                print(f"Model saved at {model_path}")

            # evaluate perioritically 
            if epoch % eval_interval == 0:
                print("Starting Evaluation")
                eval_reward = evaluate_agent(env, agent, eval_steps)
                writer.add_scalar("Evaluation/Average_Reward", eval_reward, epoch)
                print(f"Evaluation Avg Reward: {eval_reward:.2f}")

    writer.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(device)

    # evironment  setup
    os.environ['EPLUS_PATH'] = '/Applications/EnergyPlus-24-2-0'
    sys.path.append(os.environ['EPLUS_PATH'])

    env = gym.make("Eplus-datacenter-mixed-continuous-stochastic-v1",
                   config_params={
                       'timesteps_per_hour': 3,
                       'runperiod': (1, 6, 1991, 31, 8, 1991)
                   })
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # hyperparameter
    gamma = 0.99
    alpha = 0.2
    learn_alpha = True
    lr = 3e-4
    lr_decay = 0.99
    hidden_sizes = (256, 256)
    update_every = 50
    batch_size = 64
    tau = 0.005

    start_epochs = 5
    steps_in_full_simulation = 6624
    start_steps = start_epochs * steps_in_full_simulation

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        lr=lr,
        lr_decay=lr_decay,
        hidden_sizes=hidden_sizes,
        device=device,
        learn_alpha=learn_alpha
    )

    # paramerize uniquely based on hyperparams so that we can save to github easily during grid search
    run_name = (f"sac_gamma{agent.gamma}_tau{agent.tau}_alpha{agent.alpha}_lr{agent.policy_optimizer.param_groups[0]['lr']}_"
                f"hid{'-'.join(map(str, agent.hidden_sizes))}_learnalpha{agent.learn_alpha}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    model_save_path = "./sac_models"

    train_sac(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        writer=writer,
        num_epochs=100,
        steps_per_epoch=steps_in_full_simulation,
        start_steps=start_steps,
        update_after=start_steps,
        update_every=update_every,
        batch_size=batch_size,
        save_interval=10,
        model_save_path=model_save_path,
        eval_interval=10,
        eval_steps=steps_in_full_simulation,
        run_name=run_name
    )


if __name__ == "__main__":
    main()
