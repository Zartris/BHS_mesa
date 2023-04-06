from mesa import Agent, Model


class AgentPlus(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

    def print(self, s: str):
        if self.model.verbose:
            print(s)
