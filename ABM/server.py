import math
from typing import Dict

import numpy as np
from PIL import Image
from colour import Color
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import NumberInput
from mesa.visualization.modules import CanvasGrid, ChartModule

from ABM.agents import AAGV, AChargingStation, AInfeedStation, AChute, AObstacle
from ABM.model import AirportModel

RANDOM_SCENARIO = False
# load map file


if RANDOM_SCENARIO:
    grid_map = None

    NUMBER_OF_AGV_AGENTS = 10
    NUMBER_OF_CHARGING_STATIONS = 2
    NUMBER_OF_INFEED_STATIONS = 2
    NUMBER_OF_CHUTES = 2

    GRID_WIDTH = 50
    GRID_HEIGHT = 50
    canvas_scale = 6
    CANVAS_WIDTH = GRID_WIDTH * canvas_scale
    CANVAS_HEIGHT = GRID_HEIGHT * canvas_scale
else:
    path_to_map_file = "/home/zartris/Pictures/testmap.png"
    img = Image.open(path_to_map_file).convert('L')
    grid_map = np.array(img)
    grid_map = np.rot90(grid_map, 3)

    obstacles = np.transpose(np.where(grid_map == 0))

    NUMBER_OF_AGV_AGENTS = 10
    NUMBER_OF_CHARGING_STATIONS = 2
    NUMBER_OF_INFEED_STATIONS = 2
    NUMBER_OF_CHUTES = 2

    GRID_WIDTH = grid_map.shape[0]
    GRID_HEIGHT = grid_map.shape[1]
    canvas_scale = 6
    CANVAS_WIDTH = GRID_WIDTH * canvas_scale
    CANVAS_HEIGHT = GRID_HEIGHT * canvas_scale

BATTERY_COLOR_INTERVALS = 10
BATTERY_LEVEL_COLORS = list(Color("red").range_to(Color("green"), BATTERY_COLOR_INTERVALS))

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 1000

simulation_params = {
    "num_agv_agents": NumberInput(
        "Choose how many agents to include in the model", value=NUMBER_OF_AGV_AGENTS
    ),
    "num_charging_stations": NumberInput(
        "Choose how many charging stations to include in the model", value=NUMBER_OF_CHARGING_STATIONS
    ),
    "num_infeed_stations": NumberInput(
        "Choose how many infeed stations to include in the model", value=NUMBER_OF_INFEED_STATIONS
    ),
    "num_chutes": NumberInput(
        "Choose how many chutes to include in the model", value=NUMBER_OF_CHUTES
    ),
    "grid_width": NumberInput(
        "Choose the width of the grid", value=GRID_WIDTH
    ),
    "grid_height": NumberInput(
        "Choose the height of the grid", value=GRID_HEIGHT
    ),
    "max_battery": NumberInput(
        "Choose the max battery level", value=100
    ),
    "grid_map": None if grid_map is None else obstacles
    # "canvas_scale": NumberInput(
    #     "Choose the scale of the canvas", value=canvas_scale

    # "canvas_width": NumberInput(
    #     "Choose the width of the canvas", value=CANVAS_WIDTH
    # ),
    # "canvas_height": NumberInput(
    #     "Choose the height of the canvas", value=CANVAS_HEIGHT
    # )

}


def portrayal_method_for_agv(agent: AAGV, portrayal: Dict):
    portrayal["Shape"] = "circle"
    battery_percentage = agent.battery / agent.max_battery
    if battery_percentage == 0:
        color = BATTERY_LEVEL_COLORS[0]
    else:
        color = BATTERY_LEVEL_COLORS[math.ceil(battery_percentage * BATTERY_COLOR_INTERVALS) - 1]
    portrayal["Color"] = color.hex
    portrayal["r"] = 1
    portrayal["text"] = f"id: {agent.unique_id} battery: {battery_percentage:.2f}"
    portrayal["Layer"] = 2
    return portrayal


def portrayal_method_for_charging_station(agent: AChargingStation, portrayal: Dict):
    portrayal["Color"] = 'grey'
    return portrayal


def portrayal_method_for_infeed_station(agent: AInfeedStation, portrayal: Dict):
    portrayal["Color"] = 'blue'
    return portrayal


def portrayal_method_for_chute(agent: AChute, portrayal: Dict):
    portrayal["Color"] = 'red'
    return portrayal


def portrayal_method_for_obstacle(agent: AObstacle, portrayal: Dict):
    portrayal["Color"] = 'black'
    return portrayal


def agent_portrayal(agent):
    default_portrayal = {
        "Shape": "rect",  # type "circle", "rect", "image"
        "Filled": "true",
        "Color": "white",
        # "r": 0.01,
        "w": 1,
        "h": 1,
        "text": "",
        "Layer": 0,
        "text_color": "black",
        "scale": 1.0,
    }
    if isinstance(agent, AAGV):
        return portrayal_method_for_agv(agent, default_portrayal)
    elif isinstance(agent, AChargingStation):
        return portrayal_method_for_charging_station(agent, default_portrayal)
    elif isinstance(agent, AInfeedStation):
        return portrayal_method_for_infeed_station(agent, default_portrayal)
    elif isinstance(agent, AChute):
        return portrayal_method_for_chute(agent, default_portrayal)
    elif isinstance(agent, AObstacle):
        return portrayal_method_for_obstacle(agent, default_portrayal)
    else:
        raise ValueError("Unknown agent type")


grid = CanvasGrid(
    portrayal_method=agent_portrayal,
    grid_width=GRID_WIDTH,
    grid_height=GRID_HEIGHT,
    canvas_width=CANVAS_WIDTH,
    canvas_height=CANVAS_HEIGHT,
)

chart = ChartModule(
    [
        {"Label": "Max step time", "Color": "red"},
    ],
    canvas_height=300,
    data_collector_name="datacollector",
)

server = ModularServer(
    AirportModel, [grid, chart], "BHS Model", simulation_params
)
server.port = 8521  # The default
server.launch()
