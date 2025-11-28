from __future__ import annotations

import numpy as np

from highway_env.envs.parking_env import ParkingEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env import utils


class ReverseParkingDynObsEnv(ParkingEnv):
    """
    Reverse parking environment with dynamic obstacles (moving vehicles).
    
    Extends the base ParkingEnv to add a moving obstacle vehicle that
    the agent must avoid while parking.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # Dynamic obstacle settings
                "moving_obstacle": False,
                # "moving_obstacle": True,
                # "moving_obstacle_speed": 1.0,  # m/s
                # "moving_obstacle_position": [-26.0, 14.0],  # Top rightmost parking spot [x, y]
                # "moving_obstacle_heading": np.pi / 2,  # Heading in radians (π/2 = pointing up, vertical like reverse parking)
                # "moving_obstacle_goal": [-26.0, 14.0],  # Goal: bottom lane, 1st (leftmost) parking spot [x, y]
                # "moving_obstacle_goal_tolerance": 2.0,  # Stop when within this distance of goal [m]
                # "moving_obstacle_exit_distance": 12.0,  # Distance to move straight before turning (to exit parking lane) [m]
            }
        )
        return config

    def _create_vehicles(self) -> None:
        """Create vehicles including ego, goal, parked cars, and moving obstacle."""
        # Get dynamic obstacle positions to exclude from static vehicle placement
        moving_obs_pos = self.config.get("moving_obstacle_position", [26.0, 14.0])
        moving_obs_goal = self.config.get("moving_obstacle_goal", [-26.0, 14.0])
        
        # Get all available parking spots (lanes)
        empty_spots = list(self.road.network.lanes_dict().keys())
        
        # Find and remove lanes that contain the dynamic obstacle's start and goal positions
        excluded_lanes = []
        
        for lane_index in empty_spots:
            lane = self.road.network.get_lane(lane_index)
            
            # Check if this lane contains the start position (with small margin only for vehicle size)
            # Use smaller margin to only exclude the exact lane, not adjacent ones
            start_on_lane = lane.on_lane(np.array(moving_obs_pos), margin=1.0)  # Small margin for vehicle width
            
            # Check if this lane contains the goal position
            goal_on_lane = lane.on_lane(np.array(moving_obs_goal), margin=1.0)
            
            # Exclude only lanes that actually contain the start or goal positions
            if start_on_lane or goal_on_lane:
                excluded_lanes.append(lane_index)
        
        # Remove excluded lanes from available spots
        for lane_index in excluded_lanes:
            if lane_index in empty_spots:
                empty_spots.remove(lane_index)
        
        # Create controlled vehicles (ego)
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = float(i - self.config["controlled_vehicles"] // 2) * 10.0
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0.0], 2.0 * np.pi * self.np_random.uniform(), 0.0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            if vehicle.lane_index in empty_spots:
                empty_spots.remove(vehicle.lane_index)
        
        # Create goal for ego vehicle
        for vehicle in self.controlled_vehicles:
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)
        
        # Create static parked vehicles (excluding dynamic obstacle positions)
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, longitudinal=4.0, speed=0.0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)
        
        # Create walls (if enabled)
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
        
        # Add moving obstacle if enabled
        if self.config.get("moving_obstacle", True):
            # Get moving obstacle configuration
            moving_obs_pos = self.config.get("moving_obstacle_position", [26.0, 14.0])
            moving_obs_speed = self.config.get("moving_obstacle_speed", 1.0)
            moving_obs_heading = self.config.get("moving_obstacle_heading", np.pi / 2)
            moving_obs_goal = self.config.get("moving_obstacle_goal", [-26.0, -6.0])
            goal_tolerance = self.config.get("moving_obstacle_goal_tolerance", 2.0)
            exit_distance = self.config.get("moving_obstacle_exit_distance", 12.0)
            
            # Create moving obstacle vehicle with goal-seeking behavior
            moving_obstacle = GoalSeekingVehicle(
                self.road,
                moving_obs_pos,
                heading=moving_obs_heading,
                speed=moving_obs_speed,
                goal_position=moving_obs_goal,
                goal_tolerance=goal_tolerance,
                exit_distance=exit_distance,
            )
            
            # # Set dimensions to match typical vehicle size
            # moving_obstacle.LENGTH = 4.5
            # moving_obstacle.WIDTH = 2.0
            
            # # Make it the same color as static cars (yellow)
            # moving_obstacle.color = (1.0, 1.0, 0.0)  # Yellow color (same as static cars)
            
            # Add to road (but NOT to controlled_vehicles - it moves autonomously)
            self.road.vehicles.append(moving_obstacle)


class GoalSeekingVehicle(Vehicle):
    """
    A vehicle that moves towards a goal position.
    First exits the parking lane vertically, then navigates to the goal.
    """
    
    def __init__(self, road, position, heading=0, speed=0, goal_position=None, goal_tolerance=2.0, exit_distance=12.0):
        super().__init__(road, position, heading, speed)
        self.goal_position = np.array(goal_position) if goal_position is not None else None
        self.goal_tolerance = goal_tolerance
        self.exit_distance = exit_distance
        self.reached_goal = False
        self.exited_parking_lane = False
        
        # Store initial position to track distance moved
        self.initial_position = np.array(position)
        
        # Controller parameters
        self.KP_HEADING = 2.0  # Proportional gain for heading control
        self.MAX_STEERING_ANGLE = np.pi / 3  # Maximum steering angle [rad]
        self.TARGET_SPEED = speed if speed > 0 else 1.0  # Target speed [m/s]
    
    def act(self, action=None):
        """Calculate action to move towards goal."""
        if self.reached_goal or self.goal_position is None:
            # Stop if goal reached or no goal set
            self.action = {"steering": 0.0, "acceleration": -self.speed * 2.0}  # Brake to stop
            return
        
        # Phase 1: Exit parking lane vertically (move straight in current heading direction)
        if not self.exited_parking_lane:
            # Calculate distance moved from initial position
            distance_moved = np.linalg.norm(np.array(self.position) - self.initial_position)
            
            # Check if we've moved far enough to exit the parking lane
            if distance_moved >= self.exit_distance:
                self.exited_parking_lane = True
            else:
                # Continue moving straight (no steering, maintain speed)
                self.action = {"steering": 0.0, "acceleration": 0.0}  # Keep current heading and speed
                return
        
        # Phase 2: Navigate towards goal (only after exiting parking lane)
        # Calculate direction to goal
        goal_vec = self.goal_position - np.array(self.position)
        distance_to_goal = np.linalg.norm(goal_vec)
        
        # Check if goal reached
        if distance_to_goal < self.goal_tolerance:
            self.reached_goal = True
            self.action = {"steering": 0.0, "acceleration": -self.speed * 2.0}  # Brake to stop
            return
        
        # Calculate desired heading towards goal
        desired_heading = np.arctan2(goal_vec[1], goal_vec[0])
        
        # Calculate heading error (wrap to [-π, π])
        heading_error = utils.wrap_to_pi(desired_heading - self.heading)
        
        # Calculate steering angle (proportional control, clamped to max)
        steering = np.clip(self.KP_HEADING * heading_error, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        
        # Calculate acceleration to maintain target speed
        speed_error = self.TARGET_SPEED - self.speed
        acceleration = np.clip(speed_error * 2.0, -2.0, 2.0)  # Simple P controller
        
        # Set action
        self.action = {"steering": steering, "acceleration": acceleration}

