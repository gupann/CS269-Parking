# Reverse Parking with Dynamic Obstacles - Quick Guide

## What Was Added

I've created a new environment `parking-reverse-dynObs-v0` that extends the base `parking-v0` environment with a **moving obstacle** (dynamic obstacle).

## Files Created/Modified

1. **New file**: `highway_env/envs/reverse_parking_dynObs_env.py`
   - Contains `ReverseParkingDynObsEnv` class
   - Extends `ParkingEnv` and adds moving obstacle functionality

2. **Modified**: `highway_env/__init__.py`
   - Registered the new environment as `"parking-reverse-dynObs-v0"`

3. **Updated**: `reverse_parking.ipynb`
   - Changed to use the new environment with dynamic obstacles

## How to Use

### Basic Usage

```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

# Create environment with dynamic obstacles
env = gym.make(
    "parking-reverse-dynObs-v0",
    render_mode="rgb_array",
    config={
        "vehicles_count": 3,  # Number of parked vehicles
    }
)
```

### Customizing the Moving Obstacle

You can customize the moving obstacle's behavior:

```python
env = gym.make(
    "parking-reverse-dynObs-v0",
    render_mode="rgb_array",
    config={
        "vehicles_count": 3,
        # Dynamic obstacle settings
        "moving_obstacle": True,  # Enable/disable moving obstacle
        "moving_obstacle_speed": 1.0,  # Speed in m/s (default: 1.0)
        "moving_obstacle_position": [20.0, 0.0],  # Starting [x, y] position
        "moving_obstacle_heading": 0.0,  # Heading in radians (0 = right, π/2 = up)
    }
)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `moving_obstacle` | `True` | Enable/disable the moving obstacle |
| `moving_obstacle_speed` | `1.0` | Speed of moving obstacle in m/s |
| `moving_obstacle_position` | `[20.0, 0.0]` | Starting position `[x, y]` |
| `moving_obstacle_heading` | `0.0` | Initial heading in radians (0 = pointing right) |

### Disable Moving Obstacle

To use the environment without the moving obstacle:

```python
env = gym.make(
    "parking-reverse-dynObs-v0",
    config={
        "moving_obstacle": False,  # Disable moving obstacle
        "vehicles_count": 3,
    }
)
```

Or just use the original `parking-v0` environment.

## How It Works

1. **Inheritance**: `ReverseParkingDynObsEnv` extends `ParkingEnv`
   - Inherits all base parking functionality
   - Adds moving obstacle on top

2. **Moving Obstacle**:
   - Created as a `Vehicle` object (not `ControlledVehicle`)
   - Not in `controlled_vehicles` list (moves autonomously)
   - Orange colored for visual distinction
   - Moves at constant speed with constant heading
   - Automatically updated each environment step

3. **Environment Update**:
   - Each `env.step()` call:
     - Updates ego vehicle (based on your action)
     - Updates moving obstacle (based on its speed/heading)
     - Checks for collisions
     - Returns new observation

## Example: Complete Usage

```python
import gymnasium as gym
import highway_env
from utils import record_videos, show_videos

gym.register_envs(highway_env)

# Create environment
env = gym.make(
    "parking-reverse-dynObs-v0",
    render_mode="rgb_array",
    config={
        "vehicles_count": 3,
        "moving_obstacle_speed": 1.5,  # Faster obstacle
        "moving_obstacle_position": [15.0, 0.0],  # Different starting position
    }
)

# Record videos
env = record_videos(env)
obs, info = env.reset()

# Run episode
done = False
while not done:
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()
show_videos()
```

## Differences from Base Environment

| Feature | `parking-v0` | `parking-reverse-dynObs-v0` |
|---------|--------------|----------------------------|
| Static obstacles | ✅ Yes | ✅ Yes |
| Moving obstacles | ❌ No | ✅ Yes (1 moving vehicle) |
| Planning complexity | Simple | More complex (must avoid moving obstacle) |
| Realism | Lower | Higher |

## Tips

1. **Position the obstacle**: Adjust `moving_obstacle_position` to place it where it makes sense for your scenario
2. **Speed**: Start with `1.0` m/s, adjust based on difficulty needed
3. **Heading**: 
   - `0.0` = moving right
   - `π/2` = moving up
   - `π` = moving left
   - `-π/2` = moving down
4. **Visual identification**: The moving obstacle is **orange** colored

## Troubleshooting

**Issue**: Environment not found
- **Solution**: Make sure you've run `gym.register_envs(highway_env)` before creating the environment

**Issue**: Moving obstacle not moving
- **Solution**: Check that `moving_obstacle=True` in config

**Issue**: Moving obstacle in wrong position
- **Solution**: Adjust `moving_obstacle_position` to place it correctly

## Next Steps

- Experiment with different obstacle speeds and positions
- Try multiple moving obstacles (requires code modification)
- Use with your RL agent to test dynamic obstacle avoidance
- Compare performance with/without moving obstacles

Happy parking! 🚗🅿️

