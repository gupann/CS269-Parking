# Parallel Parking with Dynamic Obstacles - Complete Explanation

This document explains `parallel_parking_dynObs_env.py` - a parking environment with **moving obstacles** (dynamic obstacles).

---

## What is "DynObs"?

**DynObs** = **Dynamic Obstacles**

- **Static obstacles**: Don't move (parked cars, walls)
- **Dynamic obstacles**: Move around (other vehicles)

This environment adds a **moving vehicle** in the central lane that the parking agent must avoid!

---

## Class Structure

```python
class ParallelParkingDynObsEnv(ParkingEnv):
```

**Inheritance chain:**
```
ParkingEnv (base parking environment)
    ↑
ParallelParkingEnv (parallel parking setup)
    ↑
ParallelParkingDynObsEnv (adds moving obstacles)
```

**What this means:** `ParallelParkingDynObsEnv` inherits everything from `ParallelParkingEnv` and adds the moving obstacle feature.

---

## Part 1: Configuration (Lines 15-33)

```python
@classmethod
def default_config(cls) -> dict:
    config = super().default_config()  # Get parent config
    config.update({
        "street_length": 60.0,      # Horizontal size (left-right)
        "curb_offset": 10.0,        # Distance from center to parking rows
        "n_slots": 8,               # Number of parking slots per side
        "empty_slot_index": 3,      # Which slot is empty (goal)
        "wall_margin": 4.0,         # Wall thickness
        "add_walls": True,          # Add yellow border walls
    })
    return config
```

**What it does:**
- Gets default config from parent class (`ParallelParkingEnv`)
- Updates with specific values for this environment
- These values control the layout of the parking lot

**Visual layout:**
```
┌─────────────────────────────────────────┐ ← Top wall (yellow)
│  [car][car][car][car][car][car][car][car]  │ ← Top row (all filled)
│                                             │
│  ───────────────────────────────────────  │ ← Central lane (ego drives here)
│                                             │
│  [car][car][car][  ][car][car][car][car]  │ ← Bottom row (one empty = goal)
└─────────────────────────────────────────┘ ← Bottom wall (yellow)
```

---

## Part 2: Road Creation (Lines 38-115)

This method creates the physical layout: lanes, walls, and parking slot markers.

### Step 1: Create Road Network

```python
def _create_road(self) -> None:
    net = RoadNetwork()
    L = self.config["street_length"]  # 60.0 meters
    curb_offset = self.config["curb_offset"]  # 10.0 meters
    wall_margin = self.config["wall_margin"]  # 4.0 meters
```

**What it does:** Initializes a road network (like a graph of lanes).

### Step 2: Create Central Driving Lane

```python
lane_width = 10.0
lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)

net.add_lane(
    "drive_start",
    "drive_end",
    StraightLane(
        [0.0, 0.0],      # Start point (left)
        [L, 0.0],         # End point (right)
        width=lane_width, # 10 meters wide
        line_types=lt,    # Continuous lines on both sides
    ),
)
```

**What it creates:**
- A horizontal lane from x=0 to x=60, at y=0
- This is where the ego vehicle (your car) starts
- Width: 10 meters (wide enough for parking maneuvers)

**Visual:**
```
───────────────────────────────────────────  ← Central lane (y=0)
```

### Step 3: Create Parking Slot Markers

```python
n_slots = self.config["n_slots"]  # 8 slots
inner_margin_x = 5.0  # Empty space at edges
total_slot_span = L - 2 * inner_margin_x  # 60 - 10 = 50 meters
slot_length = total_slot_span / n_slots  # 50 / 8 = 6.25 meters per slot

for side, sign in (("bottom", -1.0), ("top", +1.0)):
    y_row = sign * curb_offset  # Top: y=+10, Bottom: y=-10
    
    for i in range(n_slots + 1):  # 9 markers (8 slots + 1 extra)
        x = inner_margin_x + i * slot_length
        # Create vertical white line marker
        net.add_lane(
            f"{side}_marker_{i}_in",
            f"{side}_marker_{i}_out",
            StraightLane(
                [x, y_row - 2.0],  # Bottom of marker
                [x, y_row + 2.0],  # Top of marker
                width=0.1,          # Very thin (just a line)
                line_types=lt,
            ),
        )
```

**What it creates:**
- Vertical white lines marking parking slot boundaries
- Top row: 9 markers at y=+10
- Bottom row: 9 markers at y=-10
- Each marker is 4 meters long (from y-2 to y+2)

**Visual:**
```
│  │  │  │  │  │  │  │  │  │  ← Top row markers
───────────────────────────────────────────  ← Central lane
│  │  │  │  │  │  │  │  │  │  ← Bottom row markers
```

### Step 4: Create Yellow Border Walls

```python
if self.config["add_walls"]:
    top_y = curb_offset + wall_margin      # 10 + 4 = 14
    bot_y = -curb_offset - wall_margin     # -10 - 4 = -14
    left_x = 0.0
    right_x = L                            # 60.0
    
    # Top & bottom walls (horizontal)
    for y in (top_y, bot_y):
        wall = Obstacle(
            self.road, 
            [(left_x + right_x) / 2.0, y],  # Center of wall
            heading=0.0                      # Horizontal
        )
        wall.LENGTH = (right_x - left_x) + 2.0  # 62 meters
        wall.WIDTH = 1.0                         # 1 meter thick
        wall.diagonal = np.sqrt(wall.LENGTH**2 + wall.WIDTH**2)
        wall.color = (1.0, 1.0, 0.0)  # Yellow
        self.road.objects.append(wall)
    
    # Left & right walls (vertical)
    height = top_y - bot_y  # 14 - (-14) = 28 meters
    for x in (left_x, right_x):
        wall = Obstacle(
            self.road,
            [x, (top_y + bot_y) / 2.0],  # Center of wall
            heading=np.pi / 2.0            # Vertical (90 degrees)
        )
        wall.LENGTH = height + 2.0  # 30 meters
        wall.WIDTH = 1.0
        wall.diagonal = np.sqrt(wall.LENGTH**2 + wall.WIDTH**2)
        wall.color = (1.0, 1.0, 0.0)  # Yellow
        self.road.objects.append(wall)
```

**What it creates:**
- Four yellow rectangular walls forming a border
- Top wall: horizontal, at y=14
- Bottom wall: horizontal, at y=-14
- Left wall: vertical, at x=0
- Right wall: vertical, at x=60

**Visual:**
```
┌─────────────────────────────────────────┐ ← Top wall (yellow)
│                                         │
│                                         │
│                                         │
└─────────────────────────────────────────┘ ← Bottom wall (yellow)
```

**Why walls?** Prevents the car from driving off the map and provides visual boundaries.

---

## Part 3: Vehicle Creation (Lines 120-202)

This is where the **key difference** from the base environment happens!

### Step 1: Create Ego Vehicle (Your Car)

```python
def _create_vehicles(self) -> None:
    n_slots = self.config["n_slots"]
    empty_idx = self.config["empty_slot_index"]
    curb_offset = self.config["curb_offset"]
    L = self.config["street_length"]
    
    self.controlled_vehicles = []
    
    # --- Ego in the central lane ---
    drive_lane = self.road.network.get_lane(
        ("drive_start", "drive_end", 0))
    ego_x = 0.2 * L  # 20% from left = 12 meters
    ego_y = drive_lane.position(ego_x, 0)[1]  # y = 0 (center of lane)
    
    ego = self.action_type.vehicle_class(
        self.road,
        [ego_x, ego_y],  # Position: (12, 0)
        heading=0.0,     # Pointing right (0 degrees)
        speed=0.0,       # Stationary
    )
    ego.color = VehicleGraphics.EGO_COLOR  # Special color (usually blue/green)
    self.road.vehicles.append(ego)
    self.controlled_vehicles.append(ego)
```

**What it creates:**
- **Ego vehicle**: The car you control (the agent)
- Position: (12, 0) - 20% from the left edge, in the center lane
- Heading: 0° (pointing right/east)
- Speed: 0 m/s (starts stationary)
- Color: Special color to distinguish it

**Visual:**
```
         🚗  ← Ego vehicle (at x=12, y=0)
───────────────────────────────────────────
```

### Step 2: Create Moving Obstacle (THE KEY DIFFERENCE!)

```python
# --- Moving obstacle vehicle in the central lane ---
# Position on the right half of the lane (60-80% of street length)
moving_obs_x = 0.7 * L  # 70% from left = 42 meters
moving_obs_y = 0.0      # Center of driving lane

moving_obstacle = Vehicle(
    self.road,
    [moving_obs_x, moving_obs_y],  # Position: (42, 0)
    heading=0.0,                    # Pointing right
    speed=1.0,                      # Moving at 1 m/s (constant speed)
)
# Set dimensions to match parked cars
moving_obstacle.LENGTH = 4.5  # 4.5 meters long
moving_obstacle.WIDTH = 2.0   # 2 meters wide
# Make it visually distinct (red or orange color)
moving_obstacle.color = (1.0, 0.5, 0.0)  # Orange color
self.road.vehicles.append(moving_obstacle)
```

**What it creates:**
- **Moving obstacle**: A vehicle that moves in the central lane
- Position: (42, 0) - 70% from left, in the center lane
- Heading: 0° (pointing right)
- Speed: 1.0 m/s (constant forward motion)
- Color: Orange (to distinguish from ego and parked cars)
- Size: Same as parked cars (4.5m × 2.0m)

**Visual:**
```
         🚗                    🚙  ← Moving obstacle (orange, moving right)
───────────────────────────────────────────
```

**Why is this important?**
- The ego vehicle must **avoid** this moving obstacle while parking
- Makes the task harder: can't just plan a static path
- Must account for the obstacle's movement
- Tests if the agent can handle dynamic environments

**How does it move?**
- The `Vehicle` class has a `step(dtmoves)` method that updates position based on speed
- Each environment step, the vehicle : `position += speed * direction * dt`
- Since `speed=1.0` and `heading=0.0`, it moves right at 1 m/s
- The environment automatically calls `step()` for all vehicles each frame
- **Important**: The moving obstacle is NOT in `controlled_vehicles`, so it doesn't receive actions from the agent
- It maintains constant speed because its action is `{"steering": 0, "acceleration": 0}` (default)
- With `acceleration=0`, speed stays constant at 1.0 m/s

### Step 3: Create Parked Cars (Top Row)

```python
# Geometry for slot centers
inner_margin_x = 5.0
total_slot_span = L - 2 * inner_margin_x  # 50 meters
slot_length = total_slot_span / n_slots   # 6.25 meters

car_len = 4.5  # Car length
car_wid = 2.0  # Car width

# --- Top row: all slots filled ---
for i in range(n_slots):  # 8 slots
    x_center = inner_margin_x + (i + 0.5) * slot_length
    y_center = +curb_offset  # y = +10 (top row)
    
    obstacle = Obstacle(self.road, [x_center, y_center], heading=0.0)
    obstacle.LENGTH = car_len
    obstacle.WIDTH = car_wid
    obstacle.diagonal = np.sqrt(car_len**2 + car_wid**2)
    self.road.objects.append(obstacle)
```

**What it creates:**
- 8 parked cars in the top row
- Each car is an `Obstacle` (static, doesn't move)
- Positioned at y=+10 (top row)
- Evenly spaced along the x-axis

**Visual:**
```
🚗🚗🚗🚗🚗🚗🚗🚗  ← Top row (all filled)
───────────────────────────────────────────
```

### Step 4: Create Parked Cars (Bottom Row) + Goal

```python
# --- Bottom row: all filled except one empty slot (goal) ---
goal_pos, goal_heading = None, None
for i in range(n_slots):  # 8 slots
    x_center = inner_margin_x + (i + 0.5) * slot_length
    y_center = -curb_offset  # y = -10 (bottom row)
    
    if i == empty_idx:  # Slot 3 (index 3)
        # empty slot -> goal here
        goal_pos = np.array([x_center, y_center])
        goal_heading = 0.0
        continue  # Skip creating obstacle for this slot
    
    # Create parked car for this slot
    obstacle = Obstacle(self.road, [x_center, y_center], heading=0.0)
    obstacle.LENGTH = car_len
    obstacle.WIDTH = car_wid
    obstacle.diagonal = np.sqrt(car_len**2 + car_wid**2)
    self.road.objects.append(obstacle)

# --- Goal landmark in the empty bottom slot ---
ego.goal = Landmark(self.road, goal_pos, heading=goal_heading)
self.road.objects.append(ego.goal)
```

**What it creates:**
- 7 parked cars in the bottom row (slot 3 is empty)
- One empty slot at index 3 (the goal)
- A `Landmark` object marking the goal position

**Visual:**
```
🚗🚗🚗🚗🚗🚗🚗🚗  ← Top row
───────────────────────────────────────────
🚗🚗🚗  🚗🚗🚗🚗  ← Bottom row (slot 3 empty = goal)
      ⭐  ← Goal landmark
```

---

## Complete Environment Layout

```
┌─────────────────────────────────────────────────────────┐
│                                                         │ ← Top wall (y=14)
│  🚗  🚗  🚗  🚗  🚗  🚗  🚗  🚗                        │ ← Top row (y=10)
│                                                         │
│  ───────────────────────────────────────────────────  │ ← Central lane (y=0)
│                                                         │
│         🚗                    🚙                        │ ← Ego (left) + Moving obstacle (right)
│                                                         │
│  🚗  🚗  🚗  ⭐  🚗  🚗  🚗  🚗                        │ ← Bottom row (y=-10)
│                                                         │
└─────────────────────────────────────────────────────────┘ ← Bottom wall (y=-14)
```

**Legend:**
- 🚗 = Parked car (static obstacle)
- 🚙 = Moving obstacle (orange, moves right at 1 m/s)
- 🚗 = Ego vehicle (your car, controlled by agent)
- ⭐ = Goal (empty parking slot)
- ─ = Central driving lane
- ┌┐└┘ = Yellow border walls

---

## How the Moving Obstacle Works

### Vehicle Dynamics

The `Vehicle` class implements a **bicycle model** for car dynamics:

```python
# From Vehicle.step(dt):
def step(self, dt: float) -> None:
    # Update position based on speed and heading
    v = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
    self.position += v * dt
    
    # Update heading based on steering (if steering was applied)
    # Update speed based on acceleration (if acceleration was applied)
```

**For the moving obstacle:**
- `speed = 1.0` (constant)
- `heading = 0.0` (pointing right)
- No steering or acceleration changes (moves straight)
- Each step: `position += [1.0, 0.0] * dt`

**Result:** The obstacle moves right at 1 meter per second.

### Environment Update Loop

Each environment step (`env.step(action)`):
1. **Agent chooses action** (steering, acceleration)
2. **Environment receives action** and forwards it to ego vehicle
3. **Environment calls `road.act()`**: 
   - All vehicles decide their actions
   - Ego vehicle: uses agent's action
   - Moving obstacle: uses default action `{"steering": 0, "acceleration": 0}`
4. **Environment calls `road.step(dt)`**:
   - Updates all vehicles' positions based on their actions
   - Ego vehicle: moves based on agent's action
   - Moving obstacle: moves at constant speed (1.0 m/s, heading 0°)
5. **Check for collisions** between all vehicles and obstacles
6. **Return** new observation, reward, done flag

**The moving obstacle is updated automatically** - you don't need to control it!

**Code flow:**
```python
# In AbstractEnv._simulate():
self.road.act()  # All vehicles decide actions
self.road.step(1 / simulation_frequency)  # All vehicles move

# In Road.act():
for vehicle in self.vehicles:
    vehicle.act()  # Each vehicle decides its action

# In Road.step(dt):
for vehicle in self.vehicles:
    vehicle.step(dt)  # Each vehicle updates position
```

**For the moving obstacle:**
- `vehicle.act()`: Sets action to `{"steering": 0, "acceleration": 0}` (default)
- `vehicle.step(dt)`: Updates position using current speed (1.0 m/s)
- Since `acceleration=0`, speed stays constant
- Since `steering=0`, heading stays constant (0°)
- Result: Moves right at constant 1.0 m/s

---

## Why This Makes the Task Harder

### Static Environment (base `ParallelParkingEnv`):
- All obstacles are fixed
- Agent can plan a path once
- Path remains valid throughout episode

### Dynamic Environment (`ParallelParkingDynObsEnv`):
- Moving obstacle changes position each step
- Agent must **replan** each step
- Must predict where obstacle will be
- Must avoid collision with moving obstacle
- More realistic (real parking lots have moving cars!)

### Example Scenario:

**Step 1:**
```
         🚗                    🚙
───────────────────────────────────────────
```

**Step 10:**
```
         🚗                         🚙  ← Moved right
───────────────────────────────────────────
```

**Step 20:**
```
         🚗                              🚙  ← Moved further right
───────────────────────────────────────────
```

The agent must account for this movement when planning!

---

## Key Differences from Base Environment

| Feature | `ParallelParkingEnv` | `ParallelParkingDynObsEnv` |
|---------|---------------------|---------------------------|
| Static obstacles | ✅ Yes (parked cars) | ✅ Yes (parked cars) |
| Moving obstacles | ❌ No | ✅ Yes (1 moving vehicle) |
| Planning complexity | Simple (static) | Complex (must predict movement) |
| Collision avoidance | Only with static objects | Static + dynamic objects |
| Realism | Lower | Higher |

---

## How to Use This Environment

```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

# Create the dynamic obstacles environment
# Note: The environment ID is "parking-parallel-dynObs-v0"
env = gym.make("parking-parallel-dynObs-v0", render_mode="rgb_array")

# Reset to start new episode
obs, info = env.reset()

# The observation includes information about the moving obstacle
# (if the observation type supports it)

# Run episode
done = False
while not done:
    action = your_agent.choose_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    # The moving obstacle updates automatically each step!
```

**Note:** The environment is registered as `"parking-parallel-dynObs-v0"` in `highway_env/__init__.py` (line 101-103).

---

## Challenges for the Agent

1. **Temporal Planning**: Must consider where obstacle will be, not just where it is
2. **Collision Avoidance**: Must avoid both static and dynamic obstacles
3. **Replanning**: Must replan each step as obstacle moves
4. **Timing**: Must time maneuvers to avoid the moving obstacle

**Example:** If the agent tries to park when the moving obstacle is blocking the path, it must wait or find an alternative route.

---

## Summary

`ParallelParkingDynObsEnv` extends the base parking environment by:

1. ✅ **Inheriting** all features from `ParallelParkingEnv`
2. ✅ **Adding** a moving obstacle vehicle in the central lane
3. ✅ **Making** the task more realistic and challenging
4. ✅ **Requiring** the agent to handle dynamic environments

The moving obstacle:
- Starts at x=42 (70% from left)
- Moves right at 1 m/s
- Is orange colored (visually distinct)
- Same size as parked cars
- Automatically updated each environment step

This creates a more challenging and realistic parking scenario! 🚗🅿️

