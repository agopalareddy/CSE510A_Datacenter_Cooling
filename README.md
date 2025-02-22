# Datacenter Cooling Optimization using Deep Reinforcement Learning

## Overview

This project implements deep reinforcement learning algorithms to optimize datacenter cooling systems using EnergyPlus simulations. It was developed as part of Washington University in St. Louis's CSE 510A: Deep Reinforcement Learning course.

## Features

- Multiple DRL algorithm implementations:
  - Deep Q-Network (DQN)
  - Proximal Policy Optimization (PPO)
  - Soft Actor-Critic (SAC)
- Integration with EnergyPlus for accurate building energy simulation
- Custom environment for datacenter cooling optimization
- Performance metrics and energy efficiency tracking
- Configurable hyperparameters for training

## Repository Structure

### Core Files
- `drl/ddqn.py` - Deep Q-Network implementation
- `drl/ppo.py` - Proximal Policy Optimization implementation
- `drl/sac.py` - Soft Actor-Critic implementation
- `requirements.txt` - Project dependencies

## Prerequisites

1. [Anaconda](https://www.anaconda.com/products/distribution) - For environment management
2. [GitHub Desktop](https://desktop.github.com/) - For repository management
3. [EnergyPlus](https://energyplus.net/downloads) - For building energy simulation

## Getting Started

1. **Clone the repository**:
   ```sh
   git clone https://github.com/peyton-gozon/CSE510A-Datacenter-Cooling
   ```

2. **Navigate to the project directory**:
   ```sh
   cd CSE510A-Datacenter-Cooling
   ```

3. **Create and activate a conda environment**:
   ```sh
   conda create -n cooler
   conda activate cooler
   ```

4. **Install Python 3.12**:
   ```sh
   conda install python=3.12
   ```

5. **Install required packages**:
   ```sh
   pip3 install -r requirements.txt
   ```

6. **Configure EnergyPlus**:
   - Set the `EPLUS_PATH` environment variable in your chosen algorithm file to point to your EnergyPlus installation directory

7. **Run the training**:
   ```sh
   cd drl
   python3 ppo.py  # or ddqn.py/sac.py for other algorithms
   ```

## Usage

Choose between different DRL algorithms based on your needs:
- `ppo.py` - Use PPO for more stable training
- `ddqn.py` - Use DQN for discrete action spaces
- `sac.py` - Use SAC for continuous action spaces with exploration

## Technologies

- Python 3.12
- EnergyPlus
- PyTorch
- Gymnasium
- NumPy

## Acknowledgments

Developed for CSE 510A at Washington University in St. Louis.
