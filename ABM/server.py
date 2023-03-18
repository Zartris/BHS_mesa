import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from colour import Color
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import NumberInput
from mesa.visualization.modules import CanvasGrid, ChartModule

from ABM.agents import AAGV, AChargingStation, AInfeedStation, AChute, AObstacle
from ABM.model import AirportModel
from ABM.utils import load_image_to_np


def run(config: Dict):
    GRID_WIDTH = config['grid_width']
    GRID_HEIGHT = config['grid_height']

    canvas_scale = config['canvas']['scale']
    CANVAS_WIDTH = GRID_WIDTH * canvas_scale
    CANVAS_HEIGHT = GRID_HEIGHT * canvas_scale

    BATTERY_COLOR_INTERVALS = config['canvas']['battery_color_interval']
    BATTERY_LEVEL_COLORS = list(Color("red").range_to(Color("green"), BATTERY_COLOR_INTERVALS))

    CANVAS_WIDTH = config['canvas']['width']
    CANVAS_HEIGHT = config['canvas']['height']

    # simulation_params = {
    #     "num_agv_agents": NumberInput(
    #         "Choose how many agents to include in the model", value=NUMBER_OF_AGV_AGENTS
    #     ),
    #     "num_charging_stations": NumberInput(
    #         "Choose how many charging stations to include in the model", value=NUMBER_OF_CHARGING_STATIONS
    #     ),
    #     "num_infeed_stations": NumberInput(
    #         "Choose how many infeed stations to include in the model", value=NUMBER_OF_INFEED_STATIONS
    #     ),
    #     "num_chutes": NumberInput(
    #         "Choose how many chutes to include in the model", value=NUMBER_OF_CHUTES
    #     ),
    #     "grid_width": NumberInput(
    #         "Choose the width of the grid", value=GRID_WIDTH
    #     ),
    #     "grid_height": NumberInput(
    #         "Choose the height of the grid", value=GRID_HEIGHT
    #     ),
    #     "max_battery": NumberInput(
    #         "Choose the max battery level", value=100
    #     ),
    #     "grid_map": None if grid_map is None else obstacles,
    #     "task_list": task_list,
    # }

    simulation_params = {
        "grid_width": GRID_WIDTH,
        "grid_height": GRID_HEIGHT,
        "config": config
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
        # portrayal["text"] = f"id: {agent.unique_id} battery: {battery_percentage:.2f}"
        portrayal["Layer"] = 2
        return portrayal

    def portrayal_method_for_charging_station(agent: AChargingStation, portrayal: Dict):
        portrayal["Color"] = '#00ff00'  # Green
        return portrayal

    def portrayal_method_for_infeed_station(agent: AInfeedStation, portrayal: Dict):
        portrayal["Color"] = '#0000ff'  # blue
        return portrayal

    def portrayal_method_for_chute(agent: AChute, portrayal: Dict):
        portrayal["Color"] = '#ff0000'  # red
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
            "Layer": 1,
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

    visualize = []
    if config['canvas']['live_visualisation']:
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
        visualize = [grid, chart]
        # visualize = [chart]

    server = ModularServer(
        AirportModel, visualize, "BHS Model", simulation_params
    )

    server.port = config['port']
    server.launch()
