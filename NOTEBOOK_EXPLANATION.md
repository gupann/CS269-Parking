# Notebook Code Explanation - Cell by Cell

This document explains each cell in `parallelparking_dynObs_model_based.ipynb` in simple terms.

---

## Cell 1: Title
```markdown
# Model-Based Reinforcement Learning
## Parallel Parking
parking-parallel-v0
```
**What it is:** Just a title! This notebook implements model-based RL for parallel parking.

---

## Cell 2: Import Libraries
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
```

**What each does:**
- `torch`: PyTorch - for building and training neural networks
- `torch.nn`: Neural network layers (Linear, etc.)
- `torch.nn.functional`: Activation functions (ReLU, etc.)
- `numpy`: Numerical computing
- `namedtuple`: Create simple data structures (like a struct)

---

## Cell 3: Visualization Setup
```python
import matplotlib.pyplot as plt
%matplotlib inline
```
**What it does:** Sets up plotting so graphs appear in the notebook.

---

## Cell 4: Utility Functions
```python
import sys
from tqdm import trange
sys.path.insert(0, '/home/aayush_wsl/cs269_rl_parking/CS269-Parking/scripts')
from utils import record_videos, show_videos
```

**What it does:**
- `tqdm`: Progress bars (shows "Collecting data: 50%")
- `record_videos`: Wrapper to record environment videos
- `show_videos`: Function to display recorded videos

---

## Cell 6: Register Environment
```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)
```

**What it does:**
- `gymnasium`: Standard RL library (formerly OpenAI Gym)
- `highway_env`: Your custom parking environment
- `register_envs`: Makes "parking-parallel-v0" available to `gym.make()`

**Why needed?** Gym needs to know about your custom environment before you can use it.

---

## Cell 9: Test Environment with Random Actions
```python
env = gym.make("parking-parallel-v0", render_mode="rgb_array")
env = record_videos(env)
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
env.close()
show_videos()
```

**Line by line:**
1. `gym.make(...)`: Creates the parking environment
2. `record_videos(env)`: Wraps environment to record videos
3. `env.reset()`: Starts a new episode, returns initial observation
4. `action_space.sample()`: Gets a random valid action
   - Action = [acceleration, steering_angle]
   - Example: [0.5, -0.3] = accelerate forward, steer left
5. `env.step(action)`: Executes action, returns:
   - `obs`: New observation (car state)
   - `reward`: How good that action was
   - `done`: Episode finished? (True/False)
   - `truncated`: Time limit reached? (True/False)
   - `info`: Extra info (for debugging)
6. Loop continues until `done=True`
7. `show_videos()`: Displays the recorded episode

**What you'll see:** Car moving randomly (probably crashing or going nowhere) because actions are random!

---

## Cell 11: Check Observation Format
```python
print("Observation format:", obs)
```

**Output:**
```python
OrderedDict([
    ('observation', array([0.375, -0.070, 0.250, -0.111, 0.914, -0.405])),
    ('achieved_goal', array([0.375, -0.070, 0.250, -0.111, 0.914, -0.405])),
    ('desired_goal', array([0.268, -0.1, 0.0, 0.0, 1.0, 0.0]))
])
```

**What each number means:**
- `observation[0]`: x position (0.375)
- `observation[1]`: y position (-0.070)
- `observation[2]`: x velocity (0.250)
- `observation[3]`: y velocity (-0.111)
- `observation[4]`: cos(heading) (0.914)
- `observation[5]`: sin(heading) (-0.405)

**Why three parts?**
- `observation`: Current state
- `achieved_goal`: Where you are (same as observation here)
- `desired_goal`: Where you want to be (the parking spot)

This is a **GoalEnv** - the agent needs to know the goal!

---

## Cell 15: Collect Experience Data
```python
Transition = namedtuple('Transition', ['state', 'action', 'next_state'])

def collect_interaction_data(env, size=1000, action_repeat=2):
    data, done = [], True
    for _ in trange(size, desc="Collecting interaction data"):
        action = env.action_space.sample()
        for _ in range(action_repeat):
            if done:
              previous_obs, info = env.reset()
            obs, reward, done, truncated, info = env.step(action)
            data.append(Transition(
                torch.Tensor(previous_obs["observation"]),
                torch.Tensor(action),
                torch.Tensor(obs["observation"])
            ))
            previous_obs = obs
    return data
```

**What's happening:**

1. **`Transition`**: A simple data structure to store one experience
   ```python
   Transition(state=[...], action=[...], next_state=[...])
   ```

2. **Function purpose**: Collect 1000 random (state, action, next_state) transitions

3. **Why random?** We don't know what's good yet - just explore!

4. **`action_repeat=2`**: Apply same action for 2 steps (more stable)

5. **What we're collecting:**
   ```
   Example:
   State: [x=0.12, y=0.0, vx=0.0, vy=0.0, cos_h=1.0, sin_h=0.0]
   Action: [accel=-0.049, steer=-0.531]
   Next State: [x=0.12, y=0.000007, vx=-0.0098, vy=-0.000002, ...]
   ```
   This tells us: "If I'm here and do this, I end up there"

**Why needed?** We'll use this data to train a model that predicts: "What happens if I take this action?"

---

## Cell 17: Build Dynamics Model
```python
class DynamicsModel(nn.Module):
    STATE_X = 0
    STATE_Y = 1

    def __init__(self, state_size, action_size, hidden_size, dt):
        super().__init__()
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        self.A1 = nn.Linear(state_size + action_size, hidden_size)
        self.A2 = nn.Linear(hidden_size, A_size)
        self.B1 = nn.Linear(state_size + action_size, hidden_size)
        self.B2 = nn.Linear(hidden_size, B_size)
```

**What is this?** A neural network that predicts the next state.

**Architecture:**
```
Input: [state (6 numbers) + action (2 numbers)] = 8 numbers
  ↓
Hidden Layer 1 (64 neurons) → ReLU activation
  ↓
Output Layer A: 6×6 = 36 numbers (reshaped to 6×6 matrix)
Output Layer B: 6×2 = 12 numbers (reshaped to 6×2 matrix)
```

**The prediction formula:**
```
next_state = current_state + dt * (A @ current_state + B @ action)
```

Where:
- `A`: 6×6 matrix (how state affects next state)
- `B`: 6×2 matrix (how action affects next state)
- `@`: Matrix multiplication
- `dt`: Time step (1/policy_frequency)

**Why this structure?** Inspired by control theory - it's a "local linearization" of the true (nonlinear) dynamics.

```python
def forward(self, x, u):
    xu = torch.cat((x, u), -1)  # Combine state and action
    xu[:, self.STATE_X:self.STATE_Y+1] = 0  # Remove x,y dependency
    
    # Compute A matrix
    A = self.A2(F.relu(self.A1(xu)))
    A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
    
    # Compute B matrix
    B = self.B2(F.relu(self.B1(xu)))
    B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
    
    # Predict next state
    dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
    return x + dx.squeeze()*self.dt
```

**Step by step:**
1. Concatenate state `x` and action `u`: `[x, u]`
2. Zero out x,y positions (design choice - removes position dependency)
3. Pass through neural networks to get A and B matrices
4. Compute change: `dx = A*x + B*u`
5. Predict: `x_new = x + dx*dt`

**Example:**
- Input: state=[0, 0, 0, 0, 1, 0], action=[0.5, 0.0]
- Model computes: A and B matrices
- Model predicts: next_state ≈ [0.167, -0.067, 0.033, -0.032, 1.043, -0.0025]

---

## Cell 19: Train the Dynamics Model
```python
optimizer = torch.optim.Adam(dynamics.parameters(), lr=0.01)

train_ratio = 0.7
train_data, validation_data = data[:int(train_ratio * len(data))], data[int(train_ratio * len(data)):]

def compute_loss(model, data_t, loss_func = torch.nn.MSELoss()):
    states, actions, next_states = data_t
    predictions = model(states, actions)
    return loss_func(predictions, next_states)
```

**What's happening:**

1. **Optimizer**: Adam optimizer (updates model weights to reduce error)
   - `lr=0.01`: Learning rate (how big steps to take)

2. **Train/Validation Split**: 
   - 70% for training (learn from this)
   - 30% for validation (check if model generalizes)

3. **Loss Function**: Mean Squared Error
   ```
   loss = mean((predicted_next_state - actual_next_state)^2)
   ```
   - If prediction is perfect: loss = 0
   - If prediction is wrong: loss > 0 (bigger = worse)

```python
def train(model, train_data, validation_data, epochs=1500):
    train_data_t = transpose_batch(train_data)
    validation_data_t = transpose_batch(validation_data)
    losses = np.full((epochs, 2), np.nan)
    
    for epoch in trange(epochs, desc="Train dynamics"):
        # Forward pass: make predictions
        loss = compute_loss(model, train_data_t)
        validation_loss = compute_loss(model, validation_data_t)
        losses[epoch] = [loss.detach().numpy(), validation_loss.detach().numpy()]
        
        # Backward pass: compute gradients and update
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
```

**Training loop (1500 times):**
1. **Forward pass**: Model makes predictions
2. **Compute loss**: Compare predictions to reality
3. **Backward pass**: Compute gradients (how to improve)
4. **Update**: Change model weights to reduce error

**What you'll see:**
- Loss starts high (bad predictions)
- Loss decreases over time (model learning)
- Eventually plateaus (model as good as it can get)

**The plot shows:**
- Blue line: Training loss (should decrease)
- Orange line: Validation loss (should track training loss)
- If validation >> training: overfitting (model memorized, doesn't generalize)

---

## Cell 21: Visualize Trajectories
```python
def predict_trajectory(state, actions, model, action_repeat=1):
    states = []
    for action in actions:
        for _ in range(action_repeat):
            state = model(state, action)  # Predict next state
            states.append(state)
    return torch.stack(states, dim=0)
```

**What it does:** Simulates a sequence of actions using the dynamics model.

**Example:**
- Start: state = [0, 0, 0, 0, 1, 0]
- Actions: [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], ...]
- Model predicts: state1 → state2 → state3 → ...
- Returns: All predicted states (the trajectory)

```python
def visualize_trajectories(model, state, horizon=15):
    # Draw car at starting position
    plt.plot(state.numpy()[0]+2.5*np.array([-1, -1, 1, 1, -1]),
             state.numpy()[1]+1.0*np.array([-1, 1, 1, -1, -1]), 'k')
    
    # Try different action combinations
    for steering in np.linspace(-0.5, 0.5, 3):  # Left, center, right
        for acceleration in np.linspace(0.8, 0.4, 2):  # Fast, slow
            actions = torch.Tensor([acceleration, steering]).view(1,1,-1)
            states = predict_trajectory(state, actions, model, action_repeat=horizon)
            plot_trajectory(states, color=next(colors))
```

**What it does:**
- Starts from a given state
- Tries different action combinations:
  - Steering: -0.5 (left), 0.0 (center), 0.5 (right)
  - Acceleration: 0.8 (fast), 0.4 (slow)
- For each combination, predicts where the car will go
- Plots all trajectories

**What you'll see:** Multiple colored lines showing where the car would go with different actions. This helps verify the dynamics model is reasonable!

---

## Cell 23: Reward Model
```python
def reward_model(states, goal, gamma=None):
    goal = goal.expand(states.shape)
    reward_weigths = torch.Tensor(env.unwrapped.config["reward_weights"])
    rewards = -torch.pow(torch.norm((states-goal)*reward_weigths, p=1, dim=-1), 0.5)
    if gamma:
        time = torch.arange(rewards.shape[0], dtype=torch.float).unsqueeze(-1)
        rewards *= torch.pow(gamma, time)
    return rewards
```

**What it computes:** Reward for being in a given state (relative to goal).

**Formula:**
```
reward = -||(state - goal) * weights||^0.5
```

**Breaking it down:**
1. `states - goal`: Distance from goal in each dimension
2. `* reward_weights`: Weight each dimension (position matters more than velocity)
3. `norm(..., p=1)`: L1 norm (sum of absolute values)
4. `pow(..., 0.5)`: Square root (smoother curve)
5. `-`: Negative because we want to minimize distance

**Example:**
- At goal: reward ≈ 0 (perfect!)
- Far from goal: reward ≈ -10 (bad)
- Closer = better reward

**Why needed?** The planner uses this to evaluate which action sequences are good.

---

## Cell 25: CEM Planner
```python
def cem_planner(state, goal, action_size, horizon=5, population=100, selection=10, iterations=5):
    state = state.expand(population, -1)
    action_mean = torch.zeros(horizon, 1, action_size)
    action_std = torch.ones(horizon, 1, action_size)
    
    for _ in range(iterations):
        # 1. Sample action sequences
        actions = torch.normal(mean=action_mean.repeat(1, population, 1), 
                              std=action_std.repeat(1, population, 1))
        actions = torch.clamp(actions, min=env.action_space.low.min(), 
                              max=env.action_space.high.max())
        
        # 2. Simulate trajectories
        states = predict_trajectory(state, actions, dynamics, action_repeat=5)
        
        # 3. Evaluate sequences
        returns = reward_model(states, goal).sum(dim=0)
        
        # 4. Select top-k
        _, best = returns.topk(selection, largest=True, sorted=False)
        best_actions = actions[:, best, :]
        
        # 5. Update distribution
        action_mean = best_actions.mean(dim=1, keepdim=True)
        action_std = best_actions.std(dim=1, unbiased=False, keepdim=True)
    
    return action_mean[0].squeeze(dim=0)
```

**What it does:** Finds the best sequence of actions to reach the goal.

**Parameters:**
- `horizon=5`: Plan 5 steps ahead
- `population=100`: Try 100 different action sequences
- `selection=10`: Keep top 10 best
- `iterations=5`: Refine 5 times

**Algorithm (Cross-Entropy Method):**

**Iteration 1:**
1. Sample 100 random action sequences (each has 5 actions)
2. For each sequence, simulate using dynamics model
3. Compute total reward for each sequence
4. Keep top 10 sequences
5. Fit new distribution to these top 10

**Iteration 2:**
1. Sample 100 sequences from updated distribution (biased toward good actions)
2. Simulate and evaluate
3. Keep top 10
4. Update distribution again

**...repeat 5 times...**

**Final:** Return the best action sequence found.

**Why this works:** Like evolution - try many things, keep the best, mutate from there. Over iterations, converges to good actions.

---

## Cell 27: Test Complete System
```python
env = gym.make("parking-parallel-v0", render_mode='rgb_array')
env = record_videos(env)
obs, info = env.reset()

for step in trange(3 * env.unwrapped.config["duration"], desc="Testing 3 episodes..."):
    # Plan best action
    action = cem_planner(torch.Tensor(obs["observation"]),
                         torch.Tensor(obs["desired_goal"]),
                         env.action_space.shape[0])
    
    # Execute in real environment
    obs, reward, done, truncated, info = env.step(action.numpy())
    
    if done or truncated:
        obs, info = env.reset()  # New episode

env.close()
show_videos()
```

**What's happening:**

1. **Create environment** and start recording

2. **For each step:**
   - Get current observation (car state + goal)
   - Use CEM planner to find best action (using learned dynamics model)
   - Execute action in real environment
   - Get new observation

3. **When episode ends** (done or truncated):
   - Reset environment
   - Start new episode

4. **Show videos** of all episodes

**The complete flow:**
```
Real Environment
    ↓ (observation)
CEM Planner
    ↓ (uses dynamics model to simulate)
    ↓ (uses reward model to evaluate)
    ↓ (returns best action)
Real Environment
    ↓ (executes action)
    ↓ (new observation)
    ...repeat...
```

**What you'll see:** Videos of the car successfully parking! 🎉

---

## Summary: The Big Picture

1. **Collect data**: Random interactions with environment
2. **Learn dynamics**: Train neural network to predict state transitions
3. **Define reward**: How to evaluate if we're doing well
4. **Plan actions**: Use dynamics + reward to find good action sequences
5. **Execute**: Run planned actions in real environment
6. **Success**: Car parks!

This is **Model-Based RL** - we learn a model of the world, then use it to plan!

