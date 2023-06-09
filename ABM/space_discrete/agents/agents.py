from __future__ import annotations

import time

from mesa import Agent, Model


# from ABM.model import AirportModel

class AgentPlus(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

    def print(self, s: str):
        if self.model.verbose:
            print(s)


class AAGV(AgentPlus):
    def __init__(self, unique_id, model, agv_config: dict):
        super().__init__(unique_id, model)
        self.state = 'idle'
        self.battery: float = agv_config['max_battery']
        self.max_battery: float = agv_config['max_battery']
        self.task = None
        self.path: list = []
        self._goal = None

        # Movement cost for each step
        self.movement_cost = agv_config['move_cost']
        self.wait_cost = agv_config['move_cost']
        self.idle_cost = agv_config['idle_cost']

    @property
    def goal(self):
        if self._goal is None:
            return self.pos
        return self._goal

    @goal.setter
    def goal(self, value: tuple):
        self._goal = value

    def random_move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,  # position of the agent
            moore=False,  # true for diagonals, false for only cardinal directions
            include_center=True,  # include the current position
            radius=1  # max move length
        )
        # Shuffle the possible steps so that the agent doesn't always move in the same direction
        self.random.shuffle(possible_steps)
        # Check if the new position is occupied by an obstacle or another AGV
        for new_position in possible_steps:
            if new_position == self.pos:
                break

            # Check if the new position is occupied
            occupied = False
            for agent in self.model.grid.iter_cell_list_contents(new_position):
                if isinstance(agent, AObstacle) or isinstance(agent, AAGV):
                    occupied = True
                    self.print("Agent {} is blocked by {}".format(self.unique_id, agent.unique_id))
                    break
            # If the new position is not occupied, move to it
            if not occupied:
                break

        if new_position == self.pos:
            # If the new position is the same as the current position, wait
            self.battery -= self.wait_cost
        else:
            # If the new position is different, move
            self.battery -= self.movement_cost
        self.battery = max(self.battery, 0)
        # Move the agent to the new position
        self.model.grid.move_agent(self, new_position)

    # @profile
    def move(self):
        if len(self.path) == 0 or self.pos == self.goal:
            # testing: set new goal and wait for next timestep
            self.goal = tuple(self.model.empty_coords[self.random.randint(0, len(self.model.empty_coords) - 1)])
            self.print(f"\nAgent {self.unique_id} looking for path to {self.goal}:")
            self.model.replan = True

            # If there is no path, move randomly
            # goal = self.model.grid.find_empty()
            if len(self.path) == 0:
                # Stay idle if there is no path and wait for path generation
                self.model.grid.move_agent(self, self.pos)
                return  # return early

        # Check if we should move:
        if self.pos == tuple(self.path[0]):
            # If we are already at the next position, remove it from the path
            self.path.pop(0)
            self.model.grid.move_agent(self, self.pos)
            return  # return early

        # If there is a path, move to the next position
        occupied = False
        for agent in self.model.grid.iter_cell_list_contents(tuple(self.path[0])):
            if isinstance(agent, AObstacle):  # or isinstance(agent, AAGV):
                occupied = True
                self.print("Agent {} is blocked by {}".format(self.unique_id, agent.unique_id))
                break
        if not occupied:
            next_position = self.path.pop(0)
        else:
            next_position = self.pos
        self.model.grid.move_agent(self, next_position)

    def look_for_charge_station(self):
        # Check if there is a charging station available
        charging_stations = self.model.schedule.charging_stations
        available_stations = [s for s in charging_stations if s.is_available()]
        # TODO:: if not urgent find a free charging station
        if len(available_stations) > 0:
            # Select a random charging station and move to it
            charging_station = self.random.choice(available_stations)
            self.model.grid.move_agent(self, charging_station.pos)
            charging_station.occupy(self)
            self.state = 'charging'
        else:
            # If there are no available charging stations, wait
            self.state = 'wait'
        # TODO:: if urgent, move to the nearest charging station and push the other agent out of the way

    def charge_battery(self, amount):
        """This is only called by the charging station"""
        self.battery += amount
        self.battery = min(self.battery, 100)

    # def pickup(self, task):
    #     # Move to the infeed station and pick up the task
    #     self.model.grid.move_agent(self, self.infeed_station.pos)
    #     self.task = task
    #     self.state = 'delivering'
    #
    # def deliver(self):
    #     # Move to the corresponding chute and deliver the task
    #     chute = self.task.chute
    #     self.model.grid.move_agent(self, chute.pos)
    #     chute.receive(self.task)
    #     self.task = None
    #     self.state = 'idle'

    def step(self):
        start = time.perf_counter()
        if self.battery == 0:
            return  # Agent is dead
        self.move()
        self.step_time = time.perf_counter() - start

        # check agent condition:
        # 1. if battery is low, look for charging station
        # 2. if there is a task, pick it up
        # 3. if there is no task and battery is high, move to idle position

        # if self.battery < 10:
        #     self.look_for_charge_station()
        #
        # if self.state == 'idle':
        #     # Check if there is a task available at the infeed station
        #     task = self.infeed_station.get_task()
        #     if task is not None:
        #         self.pickup(task)
        #     else:
        #         # If there are no tasks, wait or charge
        #         if self.battery < 20:
        #             self.charge()
        #         else:
        #             self.state = self.random.choice(['up', 'down', 'left', 'right', 'wait'])
        # elif self.state == 'charging':
        #     # Check if the battery is fully charged
        #     if self.battery >= 100:
        #         self.infeed_station.notify_charging_complete(self)
        #         self.state = 'idle'
        #     else:
        #         self.battery += 10
        # elif self.state == 'delivering':
        #     # Check if the agent has reached the infeed station
        #     if self.pos == self.task.infeed_station.pos:
        #         self.deliver()
        #     else:
        #         # Move towards the infeed station
        #         self.move()


class AChargingStation(Agent):
    def __init__(self, unique_id, model, config):
        super().__init__(unique_id, model)
        self.is_free = True
        # self.model: AirportModel = model
        self.charge_amount = config['charge_amount']

    def step(self):
        self.is_free = True
        # cell_mates contains the agents in the same cell as the charging station
        cell_mates = self.model.grid.get_cell_list_contents(self.pos)
        if len(cell_mates) <= 1:
            return

        # cell_mates contains also the charging station itself
        for agent in cell_mates:
            if isinstance(agent, AAGV):  # and agent.state == 'charging':
                # Should maximum be one AGV in the cell
                agent.charge_battery(self.charge_amount)
                self.is_free = False  # TODO:: For now, but should be changed to a queue or negotiated


class AInfeedStation(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

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


class AChute(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def mark_luggage_received(self):
        # Define luggage received logic here
        pass

    def step(self):
        # Check if a AGV is in the same cell, if so, check if it has a task with goal to this chute
        # If so, mark the task as received
        pass


class AObstacle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Used for visualization
