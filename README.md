# AI Car Driving Sim

An AI-powered car driving sim that uses reinforcement learning to learn how to drive around a custom track.

## Features

- Custom track creation using MS Paint
- Car simulation with basic physics
- Virtual sensors for track detection
- Reinforcement learning (DQN) for AI training
- Real-time visualization of the training process
- Lap timing and performance tracking

## Requirements

- Python 3.8+
- Pygame
- OpenCV
- NumPy
- PyTorch

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Create a track image using MS Paint:
   - Use black color for the track
   - Designate a start point with a **green** line and a finish line with a **red** line.
   - Save the image as 'track.png' in the project directory 

2. Run the training:
```bash
python main.py
```

## How it Works

The AI agent uses Deep Q-Network (DQN) to learn optimal driving strategies:
- Input: Car's position, orientation, and sensor readings
- Actions: Move forward, turn left, turn right
- Rewards: Based on staying on track and lap completion time

## Project Structure

- `main.py`: Main game loop and training logic
- `car.py`: Car physics and sensor implementation
- `track.py`: Track processing and environment setup
- `ai_agent.py`: DQN implementation and training logic 