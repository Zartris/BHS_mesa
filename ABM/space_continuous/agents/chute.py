from mesa import Agent

from ABM.space_continuous.agents.base import ContinuousAgent


class AChute(ContinuousAgent):
    def __init__(self, unique_id, model, config):
        super().__init__(unique_id, model, config, is_static=True, has_collision=False)

    def mark_luggage_received(self):
        # Define luggage received logic here
        pass

    def step(self):
        # Check if a AGV is in the same cell, if so, check if it has a task with goal to this chute
        # If so, mark the task as received
        pass
