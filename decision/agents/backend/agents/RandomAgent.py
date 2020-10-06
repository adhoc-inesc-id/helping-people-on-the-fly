from agents.backend.agents import Agent
from agents.backend.policies import random_policy


class Random(Agent):

    def __init__(self, num_actions):
        super().__init__("Random Agent")
        self._num_actions = num_actions

    def policy(self, observation):
        return random_policy(self._num_actions)

    # ############## #
    # Unused methods #
    # ############## #

    def _reinforce(self, timestep): pass


# ##### #
# Alias #
# ##### #

RandomAgent = Random
Dummy = RandomAgent
