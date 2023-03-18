import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mesa import Model, DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation, RandomActivationByType

from ABM.agents import AChargingStation, AInfeedStation, AChute, AAGV, AObstacle
from path_planning import AStarGenerator

from ABM.utils import load_image_to_np
import gc


class AirportModel(Model):
    def __init__(self, config, grid_width, grid_height, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.random.seed(config['seed'])
        np.random.seed(config['seed'])

        self._grid_width = grid_width
        self._grid_height = grid_height
        self.grid: MultiGrid = MultiGrid(self._grid_width,
                                         self._grid_height,
                                         torus=False)  # torus is false because we don't want the agents to wrap around the grid
        # Data stores
        self.obstacle_map = np.zeros((self._grid_width, self._grid_height), dtype=np.uint8)
        self.chutes_dict = {}
        self.charging_stations_dict = {}
        self.infeed_dict = {}

        # Initialize the activation schedule by type as we want to have different activation schedules for different agents
        self.schedule = RandomActivationByType(self)

        # Setting up the scenario
        if config['scenario_type'] == 'random_scenario':
            self.setup_random_scenario(config)
        elif config['scenario_type'] == 'image_scenario':
            self.setup_scenario_from_image(config)
        else:
            raise ValueError(f'Scenario type not supported: {config["scenario_type"]}')

        # Initiate data collection
        self.datacollector = DataCollector(
            {
                # "NAME OF DATA": lambda m: m.schedule.get some kind of data (AAGV)
                "Max step time": lambda m: np.max(
                    [agv.step_time for agv in m.schedule.agents_by_type[AAGV].values()]),
            }
        )

        # Create path planning generator
        self.a_star = AStarGenerator(self.obstacle_map)
        self.a_star.setDebugResult(True)
        # A warmup is performed in order to allocate memory for the path planning algorithm
        # This is done as the first path planning call is much slower than the following ones
        # for agv in self.schedule.agents_by_type[AAGV].values():
        #     agv.move()

        # found_count = 0
        # while found_count < 2:
        #     start_x, start_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
        #     end_x, end_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
        #     # test path planner
        #     found, path = self.a_star.findPath(start_x, start_y, end_x, end_y)

        # Here is a usecase where we don't perform well:
        found, path, debug_points = self.find_path(122, 180, 247, 110)
        if found:
            debug_map = self.obstacle_map.copy() * 255
            if len(debug_points) > 0:
                for y, x in debug_points:
                    debug_map[y, x] = 100
            for y, x in path:
                debug_map[y, x] = 170
            plt.imshow(debug_map.astype(np.uint8), origin='lower')
            plt.show()

    def find_path(self, start_x, start_y, goal_x, goal_y):
        found, path, debug_closed_set = self.a_star.findPath(start_x, start_y, goal_x, goal_y)
        if found:
            path = [(x, y) for x, y in path[::-1]]
            debug_closed_set = [(x, y) for x, y in debug_closed_set[::-1]]
        return found, path, debug_closed_set

    def step(self):
        # Compute the next step
        self.schedule.step()

        # Collect data at the end:
        self.datacollector.collect(self)  # (have to be the last line of the step function)
        n = gc.collect()
        print("Number of unreachable objects collected by GC:", n)

    def setup_random_scenario(self, config):
        scenario_config = config['random_scenario']

        id_counter = 0
        # we can cheat here as we know that the first agents are the obstacles so every spaces are free
        # Get unique random positions for the obstacles

        empty_spaces = self.grid.empties
        empty_spaces = self.random.shuffle(empty_spaces)
        for _ in range(scenario_config['num_obstacles']):
            o = AObstacle(id_counter, self)
            coord = empty_spaces[id_counter]
            self.grid.place_agent(o, coord)
            self.obstacle_map[coord.x, coord.y] = 1
            id_counter += 1

        # Create charging stations
        for _ in range(scenario_config['num_charging_stations']):
            cs = AChargingStation(id_counter, self, config['charging_station_params'])
            self.schedule.add(cs)
            coord = empty_spaces[id_counter]
            self.grid.place_agent(cs, coord)
            self.charging_stations_dict[(coord.x, coord.y)] = cs
            id_counter += 1

        # Create infeed stations
        for _ in range(scenario_config['num_infeed']):
            infeed = AInfeedStation(id_counter, self)
            self.schedule.add(infeed)
            coord = empty_spaces[id_counter]
            self.grid.place_agent(infeed, coord)
            self.infeed_dict[(coord.x, coord.y)] = infeed
            id_counter += 1

        # Create chutes
        for _ in range(scenario_config['num_chutes']):
            c = AChute(id_counter, self)
            self.schedule.add(c)
            coord = empty_spaces[id_counter]
            self.grid.place_agent(c, coord)
            self.chutes_dict[(coord.x, coord.y)] = c
            id_counter += 1

        # Create AGV agents
        for _ in range(scenario_config['num_agvs']):
            agv = AAGV(id_counter, self, config['agv_params'])
            self.schedule.add(agv)
            coord = empty_spaces[id_counter]
            self.grid.place_agent(agv, coord)
            id_counter += 1
        #
        # obstacle_pos = np.random.choice(self._grid_width * self._grid_height, size=scenario_config['num_obstacles'],
        #                                 replace=False)
        # # Convert number to x, y coordinates
        # xs = obstacle_pos % self._grid_width
        # ys = np.floor(obstacle_pos / self._grid_height).astype(np.uint)
        # obstacle_pos = np.transpose([xs, ys]).astype(np.uint)
        # # Create obstacles and obstacle map
        # for x, y in obstacle_pos:
        #     start = time.perf_counter()
        #     o = AObstacle(id_counter, self)
        #     # x, y = self.grid.find_empty()  # Slow because we need to sort each time
        #     self.obstacle_map[x, y] = 1
        #     self.grid.place_agent(o, (x, y))
        #     id_counter += 1
        #     print(f'Obstacle {id_counter} created in {time.perf_counter() - start} seconds')
        #
        # # Create charging stations
        # for _ in range(scenario_config['num_charging_stations']):
        #     cs = AChargingStation(id_counter, self, config['charging_station_params'])
        #     self.schedule.add(cs)
        #     x, y = self.grid.find_empty()
        #     self.grid.place_agent(cs, (x, y))
        #     id_counter += 1
        #
        # # Create infeed stations
        # for _ in range(scenario_config['num_infeed']):
        #     infeed = AInfeedStation(id_counter, self)
        #     self.schedule.add(infeed)
        #     x, y = self.grid.find_empty()
        #     self.grid.place_agent(infeed, (x, y))
        #     id_counter += 1
        #
        # # Create chutes
        # for _ in range(scenario_config['num_chutes']):
        #     c = AChute(id_counter, self)
        #     self.schedule.add(c)
        #     x, y = self.grid.find_empty()
        #     self.grid.place_agent(c, (x, y))
        #     id_counter += 1
        #
        # # Create AGV agents
        # for _ in range(scenario_config['num_agvs']):
        #     agv = AAGV(id_counter, self, config['agv_params'])
        #     self.schedule.add(agv)
        #     x, y = self.grid.find_empty()
        #     self.grid.place_agent(agv, (x, y))
        #     id_counter += 1

    def setup_scenario_from_image(self, config):
        id_counter = 0
        # Load image
        scenario_config = config['image_scenario']

        # 1. First check for images that exists and insert in map
        empty_space_map = np.ones((self._grid_width, self._grid_height), dtype=np.uint8)
        if Path(scenario_config['img_dir_path'], scenario_config['obstacle_img_name']).exists():
            print('Inserting obstacles into map')
            obstacle_img = load_image_to_np(Path(scenario_config['img_dir_path'], scenario_config['obstacle_img_name']),
                                            convert=None, remove_alpha=True, rotate=3)
            if obstacle_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'Obstacle map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the obstacle map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'obstacle map size: h:{obstacle_img.shape[0]} x w:{obstacle_img.shape[1]}')

            empty_space_map[np.where(
                np.all(obstacle_img[:, :] != scenario_config["floor_color"],
                       2))] = 0  # Remove every space that is not floor

            obstacles = np.where(np.all(obstacle_img[:, :] == scenario_config["obstacle_color"], 2))
            self.obstacle_map[obstacles] = 1  # Find all obstacles
            obstacles = np.transpose(obstacles)

            # Create obstacles
            for x, y in obstacles:
                o = AObstacle(id_counter, self)
                self.grid.place_agent(o, (x, y))  # Sets the position of the agent
                id_counter += 1

        if Path(scenario_config['img_dir_path'], scenario_config['charging_station_img_name']).exists():
            print('Inserting charging stations into map')
            charging_station_img = load_image_to_np(
                Path(scenario_config['img_dir_path'], scenario_config['charging_station_img_name']),
                convert=None, remove_alpha=True, rotate=3)
            if charging_station_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'Charging station map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the charging station map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'charging station map size: h:{charging_station_img.shape[0]} x w:{charging_station_img.shape[1]}')

            charging_stations = np.where(
                np.all(charging_station_img[:, :] == scenario_config["charging_station_color"], 2))
            charging_stations = np.transpose(charging_stations)
            for x, y in charging_stations:
                cs = AChargingStation(id_counter, self, config['charging_station_params'])
                self.schedule.add(cs)
                self.charging_stations_dict[(x, y)] = cs
                self.grid.place_agent(cs, (x, y))
                id_counter += 1

        if Path(scenario_config['img_dir_path'], scenario_config['infeed_img_name']).exists():
            print('Inserting infeeds into map')
            infeed_img = load_image_to_np(Path(scenario_config['img_dir_path'], scenario_config['infeed_img_name']),
                                          convert=None, remove_alpha=True, rotate=3)
            if infeed_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'Infeed map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the infeed map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'infeed map size: h:{infeed_img.shape[0]} x w:{infeed_img.shape[1]}')

            infeeds = np.where(np.all(infeed_img[:, :] == scenario_config["infeed_color"], 2))
            infeeds = np.transpose(infeeds)
            for x, y in infeeds:
                i = AInfeedStation(id_counter, self)
                self.infeed_dict[(x, y)] = i
                self.grid.place_agent(i, (x, y))
                id_counter += 1

        if Path(scenario_config['img_dir_path'], scenario_config['chute_img_name']).exists():
            print('Inserting chutes into map')
            chute_img = load_image_to_np(Path(scenario_config['img_dir_path'], scenario_config['chute_img_name']),
                                         convert=None, remove_alpha=True, rotate=3)
            if chute_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'Chute map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the chute map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'chute map size: h:{chute_img.shape[0]} x w:{chute_img.shape[1]}')

            chutes = np.where(np.all(chute_img[:, :] == scenario_config["chute_color"], 2))
            chutes = np.transpose(chutes)
            for x, y in chutes:
                c = AChute(id_counter, self)
                self.schedule.add(c)
                self.chutes_dict[(x, y)] = c
                self.grid.place_agent(c, (x, y))
                id_counter += 1

        if Path(scenario_config['img_dir_path'], scenario_config['agv_img_name']).exists():
            print('Inserting agvs into map')
            agv_img = load_image_to_np(Path(scenario_config['img_dir_path'], scenario_config['agv_img_name']),
                                       convert=None, remove_alpha=True, rotate=3)
            if agv_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'AGV map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the AGV map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'AGV map size: h:{agv_img.shape[0]} x w:{agv_img.shape[1]}')

            agvs = np.where(np.all(agv_img[:, :] == scenario_config["agv_color"], 2))
            agvs = np.transpose(agvs)
            for x, y in agvs:
                a = AAGV(id_counter, self, config['agv_params'])
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))
                id_counter += 1

        # 2. Then check for images that does not exists and insert random in map
        list_of_empty_coord = np.transpose(np.where(empty_space_map == 1))
        np.random.shuffle(list_of_empty_coord)
        self.empty_coords = list_of_empty_coord
        index = 0
        if not Path(scenario_config['img_dir_path'], scenario_config['obstacle_img_name']).exists():
            print(f'Obstacle map not found.\nInserting {scenario_config["num_obstacles"]} random obstacles into map')
            # Create obstacles
            for _ in range(scenario_config['num_obstacles']):
                x, y = list_of_empty_coord[index]
                o = AObstacle(id_counter, self)
                self.grid.place_agent(o, (x, y))
                self.obstacle_map[x, y] = 1
                id_counter += 1
                index += 1

        if not Path(scenario_config['img_dir_path'], scenario_config['charging_station_img_name']).exists():
            print(
                f'Charging station map not found.\nInserting {scenario_config["num_charging_stations"]} random charging stations into map')
            for _ in range(scenario_config['num_charging_stations']):
                x, y = list_of_empty_coord[index]
                cs = AChargingStation(id_counter, self, config['charging_station_params'])
                self.charging_stations_dict[(x, y)] = cs
                self.schedule.add(cs)
                self.grid.place_agent(cs, (x, y))
                id_counter += 1
                index += 1

        if not Path(scenario_config['img_dir_path'], scenario_config['infeed_img_name']).exists():
            print(f'Infeed map not found.\nInserting {scenario_config["num_infeeds"]} random infeeds into map')
            for _ in range(scenario_config['num_infeeds']):
                x, y = list_of_empty_coord[index]
                i = AInfeedStation(id_counter, self)
                self.infeed_dict[(x, y)] = i
                self.schedule.add(i)
                self.grid.place_agent(i, (x, y))
                id_counter += 1
                index += 1

        if not Path(scenario_config['img_dir_path'], scenario_config['chute_img_name']).exists():
            print(f'Chute map not found.\nInserting {scenario_config["num_chutes"]} random chutes into map')
            for _ in range(scenario_config['num_chutes']):
                x, y = list_of_empty_coord[index]
                c = AChute(id_counter, self)
                self.chutes_dict[(x, y)] = c
                self.schedule.add(c)
                self.grid.place_agent(c, (x, y))
                id_counter += 1
                index += 1

        if not Path(scenario_config['img_dir_path'], scenario_config['agv_img_name']).exists():
            print(f'AGV map not found.\nInserting {scenario_config["num_agvs"]} random AGVs into map')
            for _ in range(scenario_config['num_agvs']):
                x, y = list_of_empty_coord[index]
                a = AAGV(id_counter, self, config['agv_params'])
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))
                id_counter += 1
                index += 1
