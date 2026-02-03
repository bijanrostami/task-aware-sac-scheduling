# Task-Aware SAC Scheduling

A Soft Actor-Critic (SAC) reinforcement learning implementation for task-aware scheduling using the InF-S dataset.

## Overview

This project implements a Soft Actor-Critic (SAC) algorithm for scheduling tasks in a wireless communication environment. The implementation uses deep reinforcement learning to optimize task scheduling decisions based on network conditions and quality of service requirements.

## Features

- **Soft Actor-Critic (SAC)** reinforcement learning algorithm
- Gaussian and Deterministic policy support
- Replay memory for experience replay
- Custom action space for discrete scheduling decisions
- Integration with InF-S dataset for training and evaluation

## Project Structure

```
task-aware-sac-scheduling/
│
├── main.py                  # Main training script with argument parsing
├── sac.py                   # SAC algorithm implementation
├── model.py                 # Neural network models (Actor, Critic)
├── env_step.py              # Environment step logic
├── action_space.py          # Custom action space implementation
├── replay_memory.py         # Experience replay buffer
├── utils.py                 # Utility functions
│
└── data/           # Dataset directory
    ├── main_data_generation.py
    └── subnetwork_generate.py
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

See `requirements.txt` for complete dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bijanrostami/task-aware-sac-scheduling.git
cd task-aware-sac-scheduling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the main training script with default parameters:

```bash
python main.py
```

### Configuration Options

The training script supports various command-line arguments:

- `--env-name`: Environment name (default: Scheduling_SAC)
- `--policy`: Policy type - Gaussian or Deterministic (default: Gaussian)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target smoothing coefficient (default: 0.005)
- `--lr`: Learning rate (default: 0.00003)
- `--alpha`: Temperature parameter (default: 0.2)
- `--batch_size`: Batch size (default: 256)
- `--num_steps`: Maximum training steps
- `--eval`: Enable evaluation mode

Example with custom parameters:
```bash
python main.py --policy Gaussian --lr 0.0001 --batch_size 128
```

## Dataset

The project uses the InF-S dataset for training and evaluation. The dataset generation scripts are located in the `data/` directory:

- `main_data_generation.py`: Main data generation script
- `subnetwork_generate.py`: Subnetwork generation utilities

## Model Architecture

The implementation includes:
- **Gaussian Policy**: Stochastic policy for exploration
- **Deterministic Policy**: Deterministic policy for exploitation
- **Q-Network**: Twin Q-networks for value estimation
- **Action Space**: Custom discrete action space for scheduling decisions

## Citation

If you use this code in your research, please cite the relevant papers and the InF-S dataset.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.
