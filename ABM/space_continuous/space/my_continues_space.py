import numpy as np
from mesa.space import ContinuousSpace, FloatCoordinate
from numba import njit

from ABM.space_continuous.agents.base import ContinuousAgent


class MyContinuousSpace(ContinuousSpace):
    def __init__(self,
                 x_max: float,
                 y_max: float,
                 torus: bool,
                 x_min: float = 0,
                 y_min: float = 0):
        super().__init__(x_max, y_max, torus, x_min, y_min)
        self.static_agents = []
        self.dynamic_agents = []
        self.collision_agents = []

    @property
    def agents(self) -> list:
        return list(self._agent_to_index.keys())

    def place_agent(self, agent: ContinuousAgent, pos: FloatCoordinate) -> None:
        super().place_agent(agent, pos)
        if agent.is_static:
            self.static_agents.append(agent.unique_id)
        else:
            self.dynamic_agents.append(agent.unique_id)

        if agent.has_collision:
            self.collision_agents.append(agent.unique_id)

    def remove_agent(self, agent: ContinuousAgent) -> None:
        super().remove_agent(agent)
        if agent.is_static:
            self.static_agents.remove(agent.unique_id)
        else:
            self.dynamic_agents.remove(agent.unique_id)

        if agent.has_collision:
            self.collision_agents.remove(agent.unique_id)

    def move_agent(self, agent: ContinuousAgent, pos: FloatCoordinate) -> None:
        """Move an agent from its current position to a new position.

        Args:
            agent: The agent object to move.
            pos: Coordinate tuple to move the agent to.
        """
        pos = self.torus_adj(pos)
        agent.pos = pos

        if self._agent_points is not None:
            # instead of invalidating the full cache,
            # apply the move to the cached values
            idx = self._agent_to_index[agent]
            self._agent_points[idx] = pos

    def get_neighborhood_from_list(self, pos, radius, search_list, include_center=False):
        """Get all agents within a certain radius.

                Args:
                    pos: (x,y) coordinate tuple to center the search at.
                    radius: Get all the objects within this distance of the center.
                    search_list: List of indexes to search through.
                    include_center: If True, include an object at the *exact* provided
                                    coordinates. i.e. if you are searching for the
                                    neighbors of a given agent, True will include that
                                    agent in the results.
                """
        if self._agent_points is None:
            self._build_agent_cache()
        agents_pos = self._agent_points[search_list]
        (idxs,) = _get_neighborhood_from_list_numba(pos, agents_pos, radius, self.torus, self.size)
        # deltas = np.abs(agents_pos - np.array(pos))
        # if self.torus:
        #     deltas = np.minimum(deltas, self.size - deltas)
        # dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
        #
        # (idxs,) = np.where(dists <= radius ** 2)
        neighbors = [
            self._index_to_agent[search_list[x]] for x in idxs if include_center or dists[x] > 0
        ]
        return neighbors


@njit
def _get_neighborhood_from_list_numba(pos, agents_pos, radius, torus, size):
    """
    Incomplete if used alone
    """
    deltas = np.abs(agents_pos - np.array(pos))
    if torus:
        deltas = np.minimum(deltas, size - deltas)
    dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
    return np.where(dists <= radius ** 2)
