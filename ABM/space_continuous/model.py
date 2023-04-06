from pathlib import Path
from pathlib import Path
from typing import Any

import numpy as np
from mesa import DataCollector
from mesa.time import RandomActivationByType
from path_planning_lib import MAPFProblemDefinition, PIBTSolver

from ABM.core.base_model import AirportModelBase
from ABM.space_continuous.agents.agv import AAGV
from ABM.space_continuous.agents.charging_station import AChargingStation
from ABM.space_continuous.agents.chute import AChute
from ABM.space_continuous.agents.infeed_station import AInfeedStation
from ABM.space_continuous.agents.obstacle import AObstacle
from ABM.space_continuous.space.my_continues_space import MyContinuousSpace

from ABM.utils import load_image_to_np


class AirportModelContinuous(AirportModelBase):
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
        super().__init__(config, grid_width, grid_height, *args, **kwargs)

        self.grid: MyContinuousSpace = MyContinuousSpace(x_max=self._grid_width,
                                                         y_max=self._grid_height,
                                                         torus=False)  # torus is false because we don't want the agents to wrap around the grid

        # Initialize the activation schedule by type as we want to have different activation schedules for different agents
        self.schedule = RandomActivationByType(self)
        self.collision_objs = {"dynamic": [], "static": []}
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
                "Max server step time": lambda m: m.server_step_time,
            }
        )

        # Setting up problem definition
        edge_cost = np.ones((self._grid_width, self._grid_height), dtype=int)
        self.problem_definition = MAPFProblemDefinition(_instance_name="Test",
                                                        _seed=config['seed'],
                                                        _max_comp_time=config['path_planner']['max_comp_time'],
                                                        _max_timestep=config['path_planner']['max_timestep'],
                                                        _num_agents=self.num_units,
                                                        grid_map=np.copy(self.obstacle_map).astype(int).tolist(),
                                                        edge_cost_moving_up=edge_cost.tolist(),
                                                        edge_cost_moving_down=edge_cost.tolist(),
                                                        edge_cost_moving_left=edge_cost.tolist(),
                                                        edge_cost_moving_right=edge_cost.tolist())
        # Initiate config
        self.get_and_set_config()

        s, g, p = self.problem_definition.getConfigs()

        self.pibt_solver = PIBTSolver(self.problem_definition)

        self.replan = False
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

        # TODO:: Here is a usecase where we don't perform well:
        # found, path, debug_points = self.find_path(122, 180, 247, 110)
        # if found:
        #     debug_map = self.obstacle_map.copy() * 255
        #     if len(debug_points) > 0:
        #         for y, x in debug_points:
        #             debug_map[y, x] = 100
        #     for y, x in path:
        #         debug_map[y, x] = 170
        #     plt.imshow(debug_map.astype(np.uint8), origin='lower')
        #     plt.show()

    def set_config(self, start_pos, goal_pos, priority):
        """
        Set the configuration for the problem definition with the given agent positions, goals, and priorities.

        Args:
            start_pos (list): A list of agent start positions.
            goal_pos (list): A list of agent goal positions.
            priority (list): A list of agent priorities.
        """
        # print("START POSITIONS:")
        #
        # for x, y in start_pos:
        #     print("{", end="")
        #     print(f"{x}, {y}", end="")
        #     print("},", end="\n")
        #
        # print("GOAL POSITIONS:")
        #
        # for x, y in goal_pos:
        #     print("{", end="")
        #     print(f"{x}, {y}", end="")
        #     print("},", end="\n")
        #
        # print("PRIORITIES POSITIONS:")
        # print("{", end="")
        # for p in priority:
        #     print(f"{p},", end="")
        # print("}", end="\n")
        self.problem_definition.setConfig(start_pos, goal_pos, priority)

    def get_and_set_config(self):
        """
        Retrieve and set the configuration for the problem definition using the current agent positions, goals, and priorities.
        """
        start_pos, goal_pos, priority = self.get_config()
        self.set_config(start_pos, goal_pos, priority)

    def find_paths(self):
        """
        Find paths for agents using the PIBT solver.
        """
        # Initiate config
        self.get_and_set_config()

        self.pibt_solver = PIBTSolver(self.problem_definition)

        self.pibt_solver.solve()
        if self.pibt_solver.succeed():
            solution = self.pibt_solver.getSolution()
            for agent_index, agv in enumerate(self.agv_dict.values()):
                path = solution.getPathToGoalXY(agent_index)
                # print(f"Agent {agent_index} path: {path}")
                # Path contains the start position, so we remove it
                agv.path = path[1:].tolist()
                debug = 0
        else:
            print("No solution found (in time)!")
            solution = self.pibt_solver.getSolution()
            for agent_index, agv in enumerate(self.agv_dict.values()):
                path = solution.getPathXY(agent_index)
                # print(f"Agent {agent_index} path: {path}")
                # Path contains the start position, so we remove it
                agv.path = path[1:].tolist()
                if list(agv.goal) != path[-1].tolist():
                    print(f"Agent {agent_index} goal {agv.goal} not at end of path!")
                    if list(agv.goal) not in path:
                        print(f"Agent {agent_index} goal {agv.goal} not in path!")
                debug = 0
        s, g, p = self.problem_definition.getConfigs()
        self.print(self.pibt_solver.printResult())

    def server_action(self):
        if self.replan:
            self.replan = False
            self.find_paths()

    def setup_random_scenario(self, config):
        """
        Set up a random scenario for the simulation based on the given configuration.

        Args:
            config (dict): The configuration dictionary containing parameters for the random scenario.
        """
        scenario_config = config['random_scenario']
        self.num_units = scenario_config['num_agvs']
        id_counter = 0

        # Create obstacles
        empty_spaces = [(x, y) for x in range(self._grid_width) for y in range(self._grid_height)]
        empty_spaces = self.random.shuffle(empty_spaces)
        for _ in range(scenario_config['num_obstacles']):
            o = AObstacle(id_counter, self, config["obstacle_params"])
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
            infeed = AInfeedStation(id_counter, self, config["infeed_params"])
            self.schedule.add(infeed)
            coord = empty_spaces[id_counter]
            self.grid.place_agent(infeed, coord)
            self.infeed_dict[(coord.x, coord.y)] = infeed
            id_counter += 1

        # Create chutes
        for _ in range(scenario_config['num_chutes']):
            c = AChute(id_counter, self, config["chute_params"])
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
            self.agv_dict[id_counter] = agv
            id_counter += 1

    def setup_scenario_from_image(self, config):
        """
        Set up a scenario for the simulation using images as input for the configuration.

        Args:
            config (dict): The configuration dictionary containing parameters for the scenario.
        """
        id_counter = 0
        # Load image
        scenario_config = config['image_scenario']
        self.num_units = scenario_config['num_agvs']

        # 1. First check for images that exists and insert in map
        ###################### OBSTACLES FROM IMAGE ###########################
        empty_space_map = np.ones((self._grid_width, self._grid_height), dtype=np.uint8)
        if Path(scenario_config['img_dir_path'], scenario_config['obstacle_img_name']).exists():
            # Code to process obstacles from image
            self.print('Inserting obstacles into map')
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
            outside = np.where(np.all(obstacle_img[:, :] == scenario_config["outside_color"], 2))
            self.obstacle_map[outside] = 1  # Find all obstacles
            obstacles = np.transpose(obstacles)

            # Code to place an obstacle in the grid
            for x, y in obstacles:
                o = AObstacle(id_counter, self, config["obstacle_params"])
                self.grid.place_agent(o, (x, y))  # Sets the position of the agent
                id_counter += 1

        ###################### CHARGING STATIONS FROM IMAGE ###########################
        if Path(scenario_config['img_dir_path'], scenario_config['charging_station_img_name']).exists():
            # Code to process charging stations from image
            self.print('Inserting charging stations into map')
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

            # Code to place a charging station in the grid
            for x, y in charging_stations:
                cs = AChargingStation(id_counter, self, config['charging_station_params'])
                self.schedule.add(cs)
                self.charging_stations_dict[(x, y)] = cs
                self.grid.place_agent(cs, (x, y))
                id_counter += 1

        ###################### INFEED FROM IMAGE ###########################
        if Path(scenario_config['img_dir_path'], scenario_config['infeed_img_name']).exists():
            # Code to process infeeds from image
            self.print('Inserting infeeds into map')
            infeed_img = load_image_to_np(Path(scenario_config['img_dir_path'], scenario_config['infeed_img_name']),
                                          convert=None, remove_alpha=True, rotate=3)
            if infeed_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'Infeed map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the infeed map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'infeed map size: h:{infeed_img.shape[0]} x w:{infeed_img.shape[1]}')

            infeeds = np.where(np.all(infeed_img[:, :] == scenario_config["infeed_color"], 2))
            infeeds = np.transpose(infeeds)
            # Code to place an infeed in the grid
            for x, y in infeeds:
                i = AInfeedStation(id_counter, self, config["infeed_params"])
                self.infeed_dict[(x, y)] = i
                self.grid.place_agent(i, (x, y))
                id_counter += 1

        ###################### CHUTE FROM IMAGE ###########################
        if Path(scenario_config['img_dir_path'], scenario_config['chute_img_name']).exists():
            # Code to process chutes from image
            self.print('Inserting chutes into map')
            chute_img = load_image_to_np(Path(scenario_config['img_dir_path'], scenario_config['chute_img_name']),
                                         convert=None, remove_alpha=True, rotate=3)
            if chute_img.shape[:2] != (self._grid_width, self._grid_height):
                raise ValueError(f'Chute map does not have the same size as the grid.\n'
                                 f'Please go into the config and change the grid size to match the chute map size.\n'
                                 f'current grid size: h:{self._grid_height} x w:{self._grid_width}\n'
                                 f'chute map size: h:{chute_img.shape[0]} x w:{chute_img.shape[1]}')

            chutes = np.where(np.all(chute_img[:, :] == scenario_config["chute_color"], 2))
            chutes = np.transpose(chutes)
            # Code to place a chute in the grid
            for x, y in chutes:
                c = AChute(id_counter, self, config["chute_params"])
                self.schedule.add(c)
                self.chutes_dict[(x, y)] = c
                self.grid.place_agent(c, (x, y))
                id_counter += 1

        ###################### AGVS FROM IMAGE ###########################
        if Path(scenario_config['img_dir_path'], scenario_config['agv_img_name']).exists():
            self.print('Inserting agvs into map')
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
                self.agv_dict[id_counter] = a
                id_counter += 1

        # 2. Then check for images that does not exists and insert random in map
        list_of_empty_coord = np.transpose(np.where(empty_space_map == 1))
        np.random.shuffle(list_of_empty_coord)
        self.empty_coords = list_of_empty_coord
        index = 0
        ###################### OBSTACLES RANDOM ###########################
        if not Path(scenario_config['img_dir_path'], scenario_config['obstacle_img_name']).exists():
            self.print(
                f'Obstacle map not found.\nInserting {scenario_config["num_obstacles"]} random obstacles into map')
            # Create obstacles
            for _ in range(scenario_config['num_obstacles']):
                x, y = list_of_empty_coord[index]
                o = AObstacle(id_counter, self, config["obstacle_params"])
                self.grid.place_agent(o, (x, y))
                self.obstacle_map[x, y] = 1
                id_counter += 1
                index += 1

        ###################### CHARGING STATION RANDOM ###########################
        if not Path(scenario_config['img_dir_path'], scenario_config['charging_station_img_name']).exists():
            self.print(
                f'Charging station map not found.\nInserting {scenario_config["num_charging_stations"]} random charging stations into map')
            for _ in range(scenario_config['num_charging_stations']):
                x, y = list_of_empty_coord[index]
                cs = AChargingStation(id_counter, self, config['charging_station_params'])
                self.charging_stations_dict[(x, y)] = cs
                self.schedule.add(cs)
                self.grid.place_agent(cs, (x, y))
                id_counter += 1
                index += 1

        ###################### INFEED RANDOM ###########################
        if not Path(scenario_config['img_dir_path'], scenario_config['infeed_img_name']).exists():
            self.print(f'Infeed map not found.\nInserting {scenario_config["num_infeeds"]} random infeeds into map')
            for _ in range(scenario_config['num_infeeds']):
                x, y = list_of_empty_coord[index]
                i = AInfeedStation(id_counter, self, config["infeed_params"])
                self.infeed_dict[(x, y)] = i
                self.schedule.add(i)
                self.grid.place_agent(i, (x, y))
                id_counter += 1
                index += 1

        ###################### CHUTE RANDOM ###########################
        if not Path(scenario_config['img_dir_path'], scenario_config['chute_img_name']).exists():
            self.print(f'Chute map not found.\nInserting {scenario_config["num_chutes"]} random chutes into map')
            for _ in range(scenario_config['num_chutes']):
                x, y = list_of_empty_coord[index]
                c = AChute(id_counter, self, config["chute_params"])
                self.chutes_dict[(x, y)] = c
                self.schedule.add(c)
                self.grid.place_agent(c, (x, y))
                id_counter += 1
                index += 1

        ###################### AGVS RANDOM ###########################
        if not Path(scenario_config['img_dir_path'], scenario_config['agv_img_name']).exists():
            self.print(f'AGV map not found.\nInserting {scenario_config["num_agvs"]} random AGVs into map')
            for _ in range(scenario_config['num_agvs']):
                x, y = list_of_empty_coord[index]
                a = AAGV(id_counter, self, config['agv_params'])
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))
                self.agv_dict[id_counter] = a
                id_counter += 1
                index += 1
