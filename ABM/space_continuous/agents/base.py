from ABM.core.base_agent import AgentPlus


class ContinuousAgent(AgentPlus):
    def __init__(self, unique_id, model, config, is_static, has_collision):
        super().__init__(unique_id, model)
        self.radius = config['radius']  # pixel size of the agent
        self.is_static = is_static  # no need to check for collisions with other static members
        self.has_collision = has_collision
