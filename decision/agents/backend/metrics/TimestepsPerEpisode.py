import numpy as np

from decision.agents.backend.metrics.Metric import Metric


class TimestepsPerEpisode(Metric):

    def __init__(self):
        super().__init__("Timesteps per Episode")
        self._timesteps = 0
        self._timesteps_per_episode = []

    def reset(self):
        self._timesteps = 0
        self._timesteps_per_episode.clear()

    def result(self):
        return np.array(self._timesteps_per_episode)

    def _process(self, timestep):

        self._timesteps += 1

        if timestep.is_terminal:
            self._timesteps_per_episode.append(self._timesteps)
            self._timesteps = 0
