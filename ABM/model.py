from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mesa import Model, DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation, RandomActivationByType

from ABM.agents import AChargingStation, AInfeedStation, AChute, AAGV, AObstacle
from path_planning import AStarGenerator


class AirportModel(Model):
    def __init__(self, num_agv_agents, num_charging_stations, num_infeed_stations, num_chutes, grid_width, grid_height,
                 max_battery, grid_map=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.grid: MultiGrid = MultiGrid(grid_width,
                                         grid_height,
                                         torus=False)  # torus is false because we don't want the agents to wrap around the grid
        # Initialize the activation schedule by type as we want to have different activation schedules for different agents
        self.schedule = RandomActivationByType(self)
        # Initiate data collection
        self.datacollector = DataCollector(
            {
                # "NAME OF DATA": lambda m: m.schedule.get some kind of data (AAGV)
                "Max step time": lambda m: np.max(
                    [AAGV.step_time for AAGV in m.schedule.agents_by_type[AAGV].values()]),
            }
        )

        id_counter = 0
        # Create obstacles and obstacle map
        self.obstacle_map = np.zeros((grid_width, grid_height), dtype=np.uint8)
        if grid_map is not None:
            for x, y in grid_map:
                self.obstacle_map[y, x] = 1
                o = AObstacle(id_counter, self)
                self.grid.place_agent(o, (x, y))  # Sets the position of the agent
                id_counter += 1
        else:
            for _ in range(25):
                o = AObstacle(id_counter, self)
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                self.obstacle_map[y, x] = 1
                self.grid.place_agent(o, (x, y))
                id_counter += 1

        # Create charging stations
        for _ in range(num_charging_stations):
            cs = AChargingStation(id_counter, self)
            self.schedule.add(cs)
            x, y = self.grid.find_empty()
            self.grid.place_agent(cs, (x, y))
            id_counter += 1

        # Create infeed stations
        for _ in range(num_infeed_stations):
            infeed = AInfeedStation(id_counter, self)
            self.schedule.add(infeed)
            x, y = self.grid.find_empty()
            self.grid.place_agent(infeed, (x, y))
            id_counter += 1

        # Create chutes
        for _ in range(num_chutes):
            c = AChute(id_counter, self)
            self.schedule.add(c)
            x, y = self.grid.find_empty()

            self.grid.place_agent(c, (x, y))
            id_counter += 1

        # Create AGV agents
        for _ in range(num_agv_agents):
            agv = AAGV(id_counter, self, max_battery)
            self.schedule.add(agv)
            x, y = self.grid.find_empty()

            self.grid.place_agent(agv, (x, y))
            id_counter += 1

        # Create path planning generator
        self.a_star = AStarGenerator(self.obstacle_map)
        # found_count = 0
        # while found_count < 2:
        #     start_x, start_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
        #     end_x, end_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
        #     # test path planner
        #     found, path = self.a_star.findPath(start_x, start_y, end_x, end_y)
        #     if found:
        #         debug_map = self.obstacle_map.copy() * 255
        #         for y, x in path:
        #             debug_map[y, x] = 155
        #         plt.imshow(debug_map, origin='lower')
        #         plt.show()
        #         found_count += 1

    def find_path(self, start_x, start_y, goal_x, goal_y):
        found, path = self.a_star.findPath(start_y, start_x, goal_y, goal_x)
        if found:
            path = [(x, y) for y, x in path[::-1]]
        return found, path

    def step(self):
        # Compute the next step
        self.schedule.step()

        # Collect data at the end:
        self.datacollector.collect(self)  # (have to be the last line of the step function)
