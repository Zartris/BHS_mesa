from __future__ import annotations

import time

from mesa import Agent


# from ABM.model import AirportModel


# TODO:: This is auto generated code. optimize the code
class AAGV(Agent):
    def __init__(self, unique_id, model, max_battery: float):
        super().__init__(unique_id, model)
        self.state = 'idle'
        self.battery: float = max_battery
        self.max_battery: float = max_battery
        self.task = None
        self.path = []
        # self.model = model
        # Movement cost for each step TODO:: Make these parameters
        self.movement_cost = 0.1
        self.wait_cost = 0.01

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
                    print("Agent {} is blocked by {}".format(self.unique_id, agent.unique_id))
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

    def move(self):
        if len(self.path) == 0:
            # If there is no path, move randomly
            print(f"\nAgent {self.unique_id} looking for path:")
            goal = self.model.grid.find_empty()
            found, path = self.model.find_path(start_x=self.pos[0], start_y=self.pos[1],
                                                     goal_x=goal[0], goal_y=goal[1])
            if not found:
                print("No path found")
                return

            self.path = path
            print("\n")

        # Check if we should move:
        if self.pos == self.path[0]:
            # If we are already at the next position, remove it from the path
            self.path.pop(0)
            self.model.grid.move_agent(self, self.pos)
            return  # return early

            # If there is a path, move to the next position
        occupied = False
        for agent in self.model.grid.iter_cell_list_contents(self.path[0]):
            if isinstance(agent, AObstacle) or isinstance(agent, AAGV):
                occupied = True
                print("Agent {} is blocked by {}".format(self.unique_id, agent.unique_id))
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
        if len(available_stations) > 0:
            # Select a random charging station and move to it
            charging_station = self.random.choice(available_stations)
            self.model.grid.move_agent(self, charging_station.pos)
            charging_station.occupy(self)
            self.state = 'charging'
        else:
            # If there are no available charging stations, wait
            self.state = 'wait'

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
        # 3. if there is no task, move randomly (change for path finding)

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
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_free = True
        # self.model: AirportModel = model
        self.charge_amount = 4

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
        # Define infeed logic here
        pass


class AChute(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def mark_luggage_received(self):
        # Define luggage received logic here
        pass

    def step(self):
        # Define chute logic here
        pass


class AObstacle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
