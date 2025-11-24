# Complete Reinforcement Learning Tutorial
## Step-by-Step Guide for Beginners

---

## Part 1: What is Reinforcement Learning (RL)?

### The Basic Concept

Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by **interacting with an environment**. Think of it like training a dog:
- The **agent** (the dog) performs **actions** (sit, stay, roll over)
- The **environment** (the world) responds with **observations** (what the dog sees/hears)
- The **reward** (treat or scolding) tells the agent if the action was good or bad
- Over time, the agent learns which actions lead to better rewards

### Key Components

1. **Agent**: The learner/decision maker (in your case, a car trying to park)
2. **Environment**: The world the agent interacts with (the parking lot)
3. **State/Observation**: What the agent can see/measure (car position, speed, etc.)
4. **Action**: What the agent can do (steer left/right, accelerate/brake)
5. **Reward**: Feedback signal (positive for good parking, negative for crashes)
6. **Policy**: The strategy/rule the agent uses to choose actions

### The RL Loop

```
1. Agent observes the current state (s_t)
2. Agent chooses an action (a_t) based on its policy
3. Environment transitions to new state (s_{t+1}) and gives reward (r_t)
4. Agent learns from this experience
5. Repeat until the task is complete
```

---

## Part 2: What is Gym/Gymnasium?

### Overview

**Gymnasium** (formerly OpenAI Gym) is a **standardized library** for creating and testing RL algorithms. It provides:
- A consistent interface for environments
- Pre-built environments (games, robotics, etc.)
- Easy-to-use APIs for RL research

### Why Use Gym?

Instead of everyone creating their own environment differently, Gym provides a **standard interface**:

```python
env = gym.make("parking-parallel-v0")  # Create environment
obs, info = env.reset()                 # Start new episode
action = agent.choose_action(obs)       # Agent decides what to do
obs, reward, done, truncated, info = env.step(action)  # Execute action
```

### Key Gym Concepts

1. **Environment**: A class that implements the RL interface
2. **Observation Space**: What the agent can observe (e.g., 6 numbers for car state)
3. **Action Space**: What actions are possible (e.g., continuous steering + acceleration)
4. **Reset()**: Start a new episode, returns initial observation
5. **Step(action)**: Execute an action, returns:
   - `obs`: New observation
   - `reward`: Reward signal
   - `done`: Episode finished (success/failure)
   - `truncated`: Episode cut short (time limit)
   - `info`: Extra information (for debugging)

---

## Part 3: Understanding Your Parking Environment

### What is the Parking Task?

Your environment (`parking-parallel-v0`) is a **goal-conditioned** task:
- **Goal**: Park the car in a specific parking spot
- **State**: Car's position (x, y), velocity (vx, vy), heading (cos_h, sin_h)
- **Action**: Continuous control - acceleration and steering angle
- **Reward**: Based on how close you are to the goal position and orientation

### Observation Format

Your environment returns a **dictionary** with three parts:

```python
{
    'observation': [x, y, vx, vy, cos_h, sin_h],  # Current car state
    'achieved_goal': [x, y, vx, vy, cos_h, sin_h],  # Where car currently is
    'desired_goal': [x_goal, y_goal, vx_goal, vy_goal, cos_h_goal, sin_h_goal]  # Where car should be
}
```

**Why this format?** It's a "GoalEnv" - the agent needs to know both where it is AND where it wants to go!

---

## Part 4: Step-by-Step Code Explanation

### Step 1: Imports and Setup

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import highway_env
```

**What's happening:**
- `torch`: PyTorch library for neural networks
- `gymnasium`: RL environment library
- `highway_env`: Your custom parking environment

```python
gym.register_envs(highway_env)
```

**What's happening:** This registers your custom parking environment so you can use `gym.make("parking-parallel-v0")`

---

### Step 2: Create and Test the Environment

```python
env = gym.make("parking-parallel-v0", render_mode="rgb_array")
env = record_videos(env)
obs, info = env.reset()
```

**What's happening:**
- `gym.make()`: Creates the parking environment
- `render_mode="rgb_array"`: Allows video recording
- `env.reset()`: Starts a new episode, returns initial observation

```python
done = False
while not done:
    action = env.action_space.sample()  # Random action!
    obs, reward, done, truncated, info = env.step(action)
```

**What's happening:**
- `action_space.sample()`: Gets a random valid action (for testing)
- `env.step(action)`: Executes the action
- Loop continues until `done=True` (episode finished)

**Why random actions?** Just to test that the environment works! The car will move randomly.

---

### Step 3: Collect Experience Data

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
- **Purpose**: Collect random (state, action, next_state) transitions
- **Why?** We need data to train our dynamics model
- **Transition**: Stores one experience: "If I'm in state `s`, take action `a`, I end up in state `s'`"
- **action_repeat=2**: Apply the same action for 2 steps (more stable)

**Example transition:**
```
State: [x=0.12, y=0.0, vx=0.0, vy=0.0, cos_h=1.0, sin_h=0.0]
Action: [acceleration=-0.049, steering=-0.531]
Next State: [x=0.12, y=0.000007, vx=-0.0098, vy=-0.000002, cos_h=1.0, sin_h=0.00028]
```

This tells us: "If the car is at position (0.12, 0) with no velocity, and I accelerate backward while steering left, the car moves slightly and gains backward velocity."

---

### Step 4: Build a Dynamics Model

**What is a Dynamics Model?**

A dynamics model predicts: **"If I'm in state `s` and take action `a`, what will my next state `s'` be?"**

In control theory terms: `s_{t+1} = f(s_t, a_t)`

**Why do we need it?**

This is **Model-Based RL**:
1. Learn a model of how the world works (dynamics)
2. Use the model to plan ahead and find good actions
3. More sample-efficient than learning directly from rewards

```python
class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt):
        super().__init__()
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        # Two neural networks: A and B
        self.A1 = nn.Linear(state_size + action_size, hidden_size)
        self.A2 = nn.Linear(hidden_size, state_size * state_size)
        self.B1 = nn.Linear(state_size + action_size, hidden_size)
        self.B2 = nn.Linear(hidden_size, state_size * action_size)
```

**What's happening:**
- **Neural Network**: A function approximator that learns from data
- **A and B matrices**: Inspired by Linear Time-Invariant (LTI) systems
- **The model predicts**: `s_{t+1} = s_t + dt * (A @ s_t + B @ a_t)`
  - `A`: How current state affects next state
  - `B`: How action affects next state
  - `dt`: Time step size

**Why this structure?**
- It's a **local linearization**: At each point, we approximate the true (nonlinear) dynamics as linear
- More interpretable than a black-box neural network
- Works well for control tasks

```python
def forward(self, x, u):
    xu = torch.cat((x, u), -1)  # Concatenate state and action
    xu[:, self.STATE_X:self.STATE_Y+1] = 0  # Remove x,y dependency (design choice)
    
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

**Step-by-step:**
1. Combine state `x` and action `u` into one vector
2. Pass through neural networks to get A and B matrices
3. Compute change: `dx = A*x + B*u`
4. Predict next state: `x_new = x + dx*dt`

---

### Step 5: Train the Dynamics Model

```python
def compute_loss(model, data_t, loss_func = torch.nn.MSELoss()):
    states, actions, next_states = data_t
    predictions = model(states, actions)  # What model predicts
    return loss_func(predictions, next_states)  # Compare to reality
```

**What's happening:**
- **MSE Loss**: Mean Squared Error - measures how wrong our predictions are
- **Goal**: Make predictions match reality as closely as possible

```python
def train(model, train_data, validation_data, epochs=1500):
    for epoch in trange(epochs, desc="Train dynamics"):
        loss = compute_loss(model, train_data_t)
        validation_loss = compute_loss(model, validation_data_t)
        
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update model parameters
```

**What's happening:**
- **Training loop**: Repeat many times (1500 epochs)
- **Forward pass**: Make predictions
- **Backward pass**: Compute gradients (how to improve)
- **Optimizer step**: Update model weights to reduce error
- **Validation**: Check performance on unseen data (prevents overfitting)

**The learning process:**
1. Model makes bad predictions initially
2. Compare predictions to real data → compute error
3. Use error to update model (gradient descent)
4. Model gets better over time
5. Eventually, model accurately predicts: "If I'm here and do this, I'll end up there"

---

### Step 6: Reward Model

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

**What's happening:**
- **Purpose**: Compute reward for being in a given state
- **Formula**: `reward = -||(state - goal) * weights||^0.5`
  - Negative because we want to minimize distance
  - L1 norm (p=1): Sum of absolute differences
  - Square root: Makes reward curve smoother
  - Weights: Some dimensions matter more (e.g., position vs. velocity)

**Example:**
- If car is at goal: reward ≈ 0 (good!)
- If car is far from goal: reward ≈ -10 (bad!)
- Closer to goal = higher reward

**Why gamma?** Discount factor for future rewards (not used here, but common in RL)

---

### Step 7: Planning with Cross-Entropy Method (CEM)

**What is Planning?**

Now that we have a dynamics model, we can **simulate** what will happen if we take different actions, without actually doing them in the real environment!

**The Problem:** Find the best sequence of actions to reach the goal

**The Solution: Cross-Entropy Method (CEM)**

CEM is an optimization algorithm that:
1. Samples many random action sequences
2. Simulates them using the dynamics model
3. Evaluates which sequences are best (using reward model)
4. Updates the sampling distribution to favor good sequences
5. Repeats until convergence

```python
def cem_planner(state, goal, action_size, horizon=5, population=100, selection=10, iterations=5):
    state = state.expand(population, -1)
    action_mean = torch.zeros(horizon, 1, action_size)  # Start with zero mean
    action_std = torch.ones(horizon, 1, action_size)      # Start with unit variance
    
    for _ in range(iterations):
        # 1. Sample action sequences from normal distribution
        actions = torch.normal(mean=action_mean.repeat(1, population, 1), 
                              std=action_std.repeat(1, population, 1))
        actions = torch.clamp(actions, min=env.action_space.low.min(), 
                              max=env.action_space.high.max())
        
        # 2. Simulate trajectories using dynamics model
        states = predict_trajectory(state, actions, dynamics, action_repeat=5)
        
        # 3. Evaluate sequences (compute total reward)
        returns = reward_model(states, goal).sum(dim=0)
        
        # 4. Select top-k best sequences
        _, best = returns.topk(selection, largest=True, sorted=False)
        best_actions = actions[:, best, :]
        
        # 5. Update distribution to match best sequences
        action_mean = best_actions.mean(dim=1, keepdim=True)
        action_std = best_actions.std(dim=1, unbiased=False, keepdim=True)
    
    return action_mean[0].squeeze(dim=0)  # Return best action sequence
```

**Step-by-step explanation:**

1. **Initialize**: Start with a random distribution (mean=0, std=1)

2. **Sample**: Generate 100 random action sequences (each sequence has 5 actions)
   ```
   Sequence 1: [a1, a2, a3, a4, a5]
   Sequence 2: [a1, a2, a3, a4, a5]
   ...
   Sequence 100: [a1, a2, a3, a4, a5]
   ```

3. **Simulate**: For each sequence, use dynamics model to predict where we'll end up
   ```
   Start state → Action 1 → State 1 → Action 2 → State 2 → ... → Final State
   ```

4. **Evaluate**: Compute total reward for each sequence
   ```
   Sequence 1: Total reward = -5.2 (bad)
   Sequence 2: Total reward = -1.8 (good!)
   Sequence 3: Total reward = -8.1 (bad)
   ...
   ```

5. **Select**: Keep only the top 10 best sequences

6. **Update**: Fit a new distribution to these top sequences
   - New mean = average of top sequences
   - New std = spread of top sequences
   - This makes us more likely to sample good actions next time

7. **Repeat**: Do this 5 times, each time getting better action sequences

8. **Return**: The best action sequence found

**Why this works:**
- It's like evolution: try many things, keep the best, mutate from there
- Over iterations, the distribution converges to good actions
- No gradients needed (works for discrete/continuous/combinatorial problems)

---

### Step 8: Test the Complete System

```python
env = gym.make("parking-parallel-v0", render_mode='rgb_array')
env = record_videos(env)
obs, info = env.reset()

for step in trange(3 * env.unwrapped.config["duration"], desc="Testing 3 episodes..."):
    # Plan the best action using CEM
    action = cem_planner(torch.Tensor(obs["observation"]),
                         torch.Tensor(obs["desired_goal"]),
                         env.action_space.shape[0])
    
    # Execute the action in the real environment
    obs, reward, done, truncated, info = env.step(action.numpy())
    
    if done or truncated:
        obs, info = env.reset()  # Start new episode
env.close()
show_videos()
```

**What's happening:**
1. Create environment and start recording
2. For each step:
   - Use CEM planner to find best action (using learned dynamics model)
   - Execute action in real environment
   - Get new observation
3. When episode ends, reset and try again
4. Show videos of the car parking!

**The complete flow:**
```
Real Environment → Observation → CEM Planner → Action → Real Environment
                      ↑                                    ↓
                      └─────────── Dynamics Model ─────────┘
                      (used for planning/simulation)
```

---

## Part 5: Key Concepts Summary

### Model-Based vs Model-Free RL

**Model-Based (what you're doing):**
- Learn a model of the environment (dynamics)
- Use model to plan ahead
- More sample-efficient (needs less real experience)
- Can be more complex to implement

**Model-Free:**
- Learn directly from rewards (no model)
- Examples: Q-learning, Policy Gradient
- Simpler conceptually
- Usually needs more experience

### Why This Approach Works

1. **Sample Efficiency**: Learn dynamics from random data, then plan (don't need millions of episodes)
2. **Safety**: Can test actions in simulation before trying in real world
3. **Interpretability**: The dynamics model tells you how the system works
4. **Generalization**: Once you learn dynamics, you can plan for any goal

### Common RL Terminology

- **Episode**: One complete trial (e.g., one parking attempt)
- **Policy**: Strategy for choosing actions (here: CEM planner)
- **Value Function**: Expected future reward from a state
- **Q-Function**: Expected future reward from a state-action pair
- **Exploration**: Trying new actions to learn
- **Exploitation**: Using what you've learned to get rewards

---

## Part 6: Next Steps

### To Understand Better:

1. **Experiment with hyperparameters:**
   - Change `horizon` (how many steps to plan ahead)
   - Change `population` (how many sequences to try)
   - Change `hidden_size` (dynamics model complexity)

2. **Visualize:**
   - Look at the trajectory predictions
   - See how the dynamics model improves over training
   - Compare planned vs actual trajectories

3. **Try model-free methods:**
   - DQN (Deep Q-Network)
   - PPO (Proximal Policy Optimization)
   - Compare sample efficiency

### Common Issues:

1. **Model doesn't predict well:**
   - Need more training data
   - Model too simple (increase hidden_size)
   - Learning rate too high/low

2. **Planning doesn't work:**
   - Dynamics model inaccurate
   - Horizon too short
   - Reward function not well-tuned

3. **Car doesn't park:**
   - Check if dynamics model is accurate
   - Increase planning horizon
   - Adjust reward weights

---

## Part 7: Code Flow Diagram

```
START
  │
  ├─→ Create Environment (parking-parallel-v0)
  │
  ├─→ Collect Random Data (1000 transitions)
  │   └─→ (state, action, next_state) tuples
  │
  ├─→ Build Dynamics Model (neural network)
  │   └─→ Predicts: next_state = f(state, action)
  │
  ├─→ Train Dynamics Model (1500 epochs)
  │   └─→ Minimize prediction error
  │
  ├─→ Define Reward Model
  │   └─→ reward = -distance_to_goal
  │
  ├─→ CEM Planner (for each step)
  │   ├─→ Sample action sequences
  │   ├─→ Simulate with dynamics model
  │   ├─→ Evaluate with reward model
  │   └─→ Return best action
  │
  └─→ Execute in Real Environment
      └─→ Car parks! 🎉
```

---

## Conclusion

You've implemented a complete **Model-Based Reinforcement Learning** system:

1. ✅ Learned how the environment works (dynamics model)
2. ✅ Used that knowledge to plan good actions (CEM)
3. ✅ Successfully solved the parking task

This is a powerful approach that combines:
- **Supervised Learning** (training the dynamics model)
- **Optimization** (CEM planning)
- **Reinforcement Learning** (goal-oriented behavior)

Great job! 🚗🅿️

