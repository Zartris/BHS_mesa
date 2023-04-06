from mesa import Agent

from ABM.space_continuous.agents.base import ContinuousAgent


class AObstacle(ContinuousAgent):
    def __init__(self, unique_id, model, config):
        super().__init__(unique_id, model, config, is_static=True, has_collision=True)
        # Used for visualization
