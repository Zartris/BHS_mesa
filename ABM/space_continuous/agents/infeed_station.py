from mesa import Agent

from ABM.space_continuous.agents.base import ContinuousAgent


class AInfeedStation(ContinuousAgent):
    def __init__(self, unique_id, model, config):
        super().__init__(unique_id, model, config, is_static=True, has_collision=False)

    def link_agv(self, agv_agent):
        # Define linking logic here
        pass

    def step(self):
        # Two states, either random task spawn or read from file
        pass

    def get_task(self):
        # Two states, either random task spawn or read from file
        if self.model.random_task:
            # Random task generation
            pass
        else:
            # Read from file
            pass
