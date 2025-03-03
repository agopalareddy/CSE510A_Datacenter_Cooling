# Datacenter Cooling Optimization using Deep Reinforcement Learning

## Overview

This project implements deep reinforcement learning algorithms to optimize datacenter cooling systems using EnergyPlus simulations. It was developed as part of Washington University in St. Louis's CSE 510A: Deep Reinforcement Learning course.

With the growing popularity of deep learning and big data, data centers have become essential to modern infrastructure, leading to increased energy consumption both for computing operations and cooling needs. While large-scale data centers have been extensively studied, small- to mid-sized data centers remain understudied despite occupying 42.5% and 19.5% of the market share respectively. This project focuses on using Deep Reinforcement Learning (DRL) to optimize cooling efficiency in these smaller facilities.

## Research Context

As machines within a data center complete their tasks, they generate heat, creating a complex spatial cooling problem. This project leverages DRL to dynamically adapt to changing conditions such as machine workload and external temperature.

Key contributions include:
- A focus on small- to mid-sized data centers
- A novel exploration of Dueling Deep Q-Networks (DDQN) for data-center cooling
- Comparative analysis between DRL methods and traditional control approaches

## Features

- Multiple DRL algorithm implementations:
  - Dueling Deep Q-Network (DDQN) - Our novel contribution to datacenter cooling
  - Proximal Policy Optimization (PPO) with Generalized Advantage Estimation
  - Soft Actor-Critic (SAC) with automatic entropy tuning
- Integration with EnergyPlus for accurate building energy simulation
- Custom environment for datacenter cooling optimization
- Performance metrics and energy efficiency tracking
- Configurable hyperparameters for training
- Baselines for comparison (Random, Rules-Based Controller, Rules-Based Incremental)

## Repository Structure

### Core Files
- `drl/ddqn.py` - Dueling Deep Q-Network implementation
- `drl/ppo.py` - Proximal Policy Optimization implementation
- `drl/sac.py` - Soft Actor-Critic implementation
- `requirements.txt` - Project dependencies

## Prerequisites

1. [Anaconda](https://www.anaconda.com/products/distribution) - For environment management
2. [GitHub Desktop](https://desktop.github.com/) - For repository management
3. [EnergyPlus](https://energyplus.net/downloads) - For building energy simulation
4. [Sinergym](https://github.com/ugr-sail/sinergym) - Python wrapper for EnergyPlus

## Experimental Setup

This project uses the `Sinergym` Python package to simulate a small datacenter through the `Eplus-datacenter-mixed-continuous-stochastic-v1` environment. The environment simulates:

- A 491.3 m² building divided into two asymmetrical zones (west and east)
- Each zone equipped with an HVAC system
- Hosted servers as primary heat sources
- Stochastic weather conditions with 1.5 standard deviations of normal amplification
- Training period from June 1st to August 31st

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
- `ddqn.py` - Our novel approach using Dueling DQN with discretized action space (best performance in our tests)
- `ppo.py` - PPO with Generalized Advantage Estimation for more stable training
- `sac.py` - SAC with automatic entropy tuning for continuous action spaces

## Key Findings

Our experiments revealed:

- DDQN significantly outperformed other approaches, showing a 35.8% improvement over the Rules-Based Incremental Controller
- PPO achieved a 10.9% improvement over the baseline
- SAC showed limited improvement (0.284%) compared to the baseline
- Weather forecasting data generally reduced model performance across configurations
- Model-free approaches like DDQN offer promising results for small to mid-sized data centers with limited computational resources

## Recommended Hyperparameters

Based on our research, we recommend:
- **DDQN**: Learning rate of 0.0005 decayed over 50k timesteps, γ=0.99
- **PPO**: Clip ratio of 0.1, 64×64 node network, linear learning rate scheduling, batch size of 64
- **SAC**: γ=0.99, τ=0.005, α=0.2, learning rate of 0.0003, (256, 2) network architecture with learnable α and automatic entropy tuning

## Technologies

- Python 3.12
- EnergyPlus 24.2.0
- PyTorch 2.5.1
- Gymnasium 1.0.0
- Sinergym 3.7.0
- NumPy
- Tensorboard 2.18.0

## Acknowledgments

Developed by Joseph Islam, Peyton Gozon, and Aadarsha Gopala Reddy for CSE 510A at Washington University in St. Louis.
