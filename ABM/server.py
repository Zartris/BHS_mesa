import math
from typing import Dict

from colour import Color
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

from ABM.space_continuous.model import AirportModelContinuous
from ABM.space_continuous.viz.visualisation import SimpleCanvas
from ABM.space_discrete.agents.agents import AAGV, AChargingStation, AInfeedStation, AChute, AObstacle
from ABM.space_continuous.agents.agv import AAGV as AAGV_c
from ABM.space_continuous.agents.charging_station import AChargingStation as AChargingStation_c
from ABM.space_continuous.agents.infeed_station import AInfeedStation as AInfeedStation_c
from ABM.space_continuous.agents.chute import AChute as AChute_c
from ABM.space_continuous.agents.obstacle import AObstacle as AObstacle_c
from ABM.space_discrete.model import AirportModel


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
        portrayal["r"] = 0.4
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
            "r": 1.,
            "w": 1.,
            "h": 1.,
            "text": "",
            "Layer": 1,
            "text_color": "black",
            "scale": 1.0,
        }
        if isinstance(agent, AAGV) or isinstance(agent, AAGV_c):
            return portrayal_method_for_agv(agent, default_portrayal)
        elif isinstance(agent, AChargingStation) or isinstance(agent, AChargingStation_c):
            return portrayal_method_for_charging_station(agent, default_portrayal)
        elif isinstance(agent, AInfeedStation) or isinstance(agent, AInfeedStation_c):
            return portrayal_method_for_infeed_station(agent, default_portrayal)
        elif isinstance(agent, AChute) or isinstance(agent, AChute_c):
            return portrayal_method_for_chute(agent, default_portrayal)
        elif isinstance(agent, AObstacle) or isinstance(agent, AObstacle_c):
            return portrayal_method_for_obstacle(agent, default_portrayal)
        else:
            raise ValueError("Unknown agent type")

    visualize = []
    if config['canvas']['live_visualisation']:
        if config['continuous']:
            canvas = SimpleCanvas(agent_portrayal, CANVAS_WIDTH, CANVAS_HEIGHT)
        else:
            canvas = CanvasGrid(
                portrayal_method=agent_portrayal,
                grid_width=GRID_WIDTH,
                grid_height=GRID_HEIGHT,
                canvas_width=CANVAS_WIDTH,
                canvas_height=CANVAS_HEIGHT,
            )
        agent_step_time_chart = ChartModule(
            [
                {"Label": "Max step time", "Color": "red"},
            ],
            canvas_height=300,
            data_collector_name="datacollector",
        )
        server_step_time_chart = ChartModule(
            [
                {"Label": "Max server step time", "Color": "red"},
            ],
            canvas_height=300,
            data_collector_name="datacollector",
        )
        visualize = [canvas, server_step_time_chart, agent_step_time_chart]
        # visualize = [chart]

    if config['continuous']:
        model = AirportModelContinuous
    else:
        model = AirportModel
    server = ModularServer(
        model, visualize, "BHS Model", simulation_params
    )

    server.port = config['port']
    server.launch()
