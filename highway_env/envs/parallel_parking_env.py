from __future__ import annotations

import numpy as np

from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle, Landmark
from highway_env.vehicle.graphics import VehicleGraphics


class ParallelParkingEnv(ParkingEnv):
    """
    Parallel parking task with static parked cars on one side of the road
    and one empty spot that is the goal.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # number of parallel parking slots along the curb
                "n_slots": 6,
                # which slot index is empty and should be parked into
                "empty_slot_index": 2,
                # longitudinal size of the street (for scaling)
                "street_length": 60.0,
                # distance between driving lane center and parking lane center
                "curb_offset": 4.0,
                # we usually don't need the big rectangular walls here
                "add_walls": False,
            }
        )
        return config

    def _create_road(self) -> None:
        """
        Create a straight road with one driving lane and a row of
        short parking-lane segments that represent the parallel slots.
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)

        L = self.config["street_length"]
        curb_offset = self.config["curb_offset"]

        # Driving lane: along x, y = 0
        net.add_lane(
            "drive_start",
            "drive_end",
            StraightLane([0.0, 0.0], [L, 0.0], width=width, line_types=lt),
        )

        # Parking lane: short segments along x, below the driving lane (at y = -curb_offset)
        n_slots = self.config["n_slots"]
        slot_length = L / (n_slots + 2)  # leave margins at both ends

        for k in range(n_slots):
            x0 = (k + 1) * slot_length
            x1 = (k + 2) * slot_length
            net.add_lane(
                f"slot_{k}_in",
                f"slot_{k}_out",
                StraightLane(
                    [x0, -curb_offset],
                    [x1, -curb_offset],
                    width=width,
                    line_types=lt,
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """
        Ego starts in the driving lane, parked cars fill all slots except one,
        and the goal is the center of the empty slot.
        """
        n_slots = self.config["n_slots"]
        empty_idx = self.config["empty_slot_index"]
        curb_offset = self.config["curb_offset"]
        L = self.config["street_length"]

        self.controlled_vehicles = []

        # --- Ego vehicle on the driving lane, a bit before the first slot ---
        drive_lane = self.road.network.get_lane(
            ("drive_start", "drive_end", 0))
        ego_x = 0.1 * L
        ego_y = drive_lane.position(ego_x, 0)[1]

        ego = self.action_type.vehicle_class(
            self.road,
            [ego_x, ego_y],
            heading=0.0,  # facing along +x
            speed=0.0,
        )
        ego.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        # --- Parked "cars" as obstacles in every slot except the empty one ---
        for k in range(n_slots):
            lane = self.road.network.get_lane(
                (f"slot_{k}_in", f"slot_{k}_out", 0))
            center = lane.position((lane.length) / 2, 0)

            if k == empty_idx:
                # this is the empty slot, no parked car here
                # we will place the goal here instead
                goal_pos = center
                goal_heading = lane.heading
                continue

            # parked car as an obstacle rectangle
            obstacle = Obstacle(self.road, center, heading=lane.heading)
            # make the obstacle roughly car-sized
            obstacle.LENGTH = 4.5
            obstacle.WIDTH = 2.0
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)

        # --- Goal landmark in the empty slot ---
        ego.goal = Landmark(self.road, goal_pos, heading=goal_heading)
        self.road.objects.append(ego.goal)
