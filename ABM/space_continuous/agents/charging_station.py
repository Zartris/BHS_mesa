from mesa import Agent

from ABM.space_continuous.agents.agv import AAGV
from ABM.space_continuous.agents.base import ContinuousAgent


class AChargingStation(ContinuousAgent):
    def __init__(self, unique_id, model, config):
        super().__init__(unique_id, model, config, is_static=True, has_collision=False)
        self.is_free = True
        # self.model: AirportModel = model
        self.charge_amount = config['charge_amount']

    def step(self):
        self.is_free = True
        # cell_mates contains the agents in the same cell as the charging station
        cell_mates = self.model.grid.get_neighborhood_from_list(self.pos, self.radius, self.model.grid.dynamic_agents,
                                                                include_center=True)
        if len(cell_mates) <= 1:
            return

        # cell_mates contains also the charging station itself
        for agent in cell_mates:
            if isinstance(agent, AAGV):  # and agent.state == 'charging':
                # Should maximum be one AGV in the cell
                agent.charge_battery(self.charge_amount)
                self.is_free = False  # TODO:: For now, but should be changed to a queue or negotiated
