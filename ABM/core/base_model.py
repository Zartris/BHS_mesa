import abc
import gc
import time
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
from mesa import Model

from ABM.space_discrete.agents.agents import AChargingStation, AInfeedStation, AChute, AAGV, AObstacle
from ABM.utils import load_image_to_np


class AirportModelBase(Model, ABC):
    """
    The AirportModel class represents the model for an airport in the Agent-Based Model (ABM) simulation.
    It extends the Mesa Model class and defines the simulation's grid, schedules, data collection,
    and scenario setups.
    """

    def __init__(self, config, grid_width, grid_height, *args: Any, **kwargs: Any):
        """
        Initialize an AirportModel instance with the provided configuration, grid dimensions, and other parameters.

        Args:
            config (dict): The configuration dictionary containing parameters for the simulation.
            grid_width (int): The width of the simulation grid.
            grid_height (int): The height of the simulation grid.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.random.seed(config['seed'])
        np.random.seed(config['seed'])
        self.num_units = -1  # is being set later

        self.verbose = config['verbose']
        self._grid_width = grid_width
        self._grid_height = grid_height

        # Data stores
        self.obstacle_map = np.zeros((self._grid_width, self._grid_height), dtype=np.uint8)
        self.chutes_dict = {}
        self.charging_stations_dict = {}
        self.infeed_dict = {}
        self.agv_dict = {}
        n = gc.collect()
        self.print(f"Number of unreachable objects collected by GC: {n}")

    def print(self, s: str):
        """
        Print the given string if the model's verbosity is set to True.

        Args:
            s (str): The string to print.
        """
        if self.verbose:
            print(s)

    def get_config(self):
        """
        Get the current configuration of agent positions and goals.

        Returns:
            tuple: A tuple containing the start positions, goal positions, and priorities of agents.
        """
        start_pos = [agv.pos for agv in self.agv_dict.values()]
        goal_pos = [agv.goal for agv in self.agv_dict.values()]
        priority = [0 for _ in range(self.num_units)]
        return start_pos, goal_pos, priority

    def step(self):
        """
        Execute one step of the simulation, which includes server actions, computing the next step, and collecting data.
        """
        # Perform server actions
        server_start_time = time.perf_counter()
        self.server_action()
        self.server_step_time = time.perf_counter() - server_start_time

        # Compute the next step
        self.schedule.step()

        # Collect data at the end of the step
        self.datacollector.collect(self)  # (have to be the last line of the step function)

    @abc.abstractmethod
    def server_action(self):
        pass

    def run_performance_test(self, iterations):
        """
        Run a performance test by executing the specified number of simulation steps and measuring the time it takes.

        Args:
            iterations (int): The number of iterations (steps) for the performance test.
        """
        start_time = time.perf_counter()
        for i in range(iterations):
            print(f'Iteration {i}')
            self.step()
        end_time = time.perf_counter()
        print(f'Performance test: {iterations} iterations took {end_time - start_time} seconds')
