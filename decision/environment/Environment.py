from abc import ABC, abstractmethod
from typing import Optional

from agents import Timestep


class Environment(ABC):
    """
    An abstract class representing an environment.
    """

    def __init__(self,
                 name: str,
                 observation_space: tuple,
                 action_space: tuple,
                 render: bool,
                 action_descriptions: Optional[tuple] = None):

        self._name = name

        self._observation_space = observation_space
        self._action_space = action_space
        self._num_actions = len(self.action_space)
        self._action_descriptions = action_descriptions or ["Action Description Unavailable."
                                                            for _ in range(self._num_actions)]

        self._episode_timesteps = 0
        self._observation = None
        self._is_terminal = True

        self._should_render = render

    @property
    def name(self):
        """
        The name of the environment.
        """
        return self._name

    @property
    def observation_space(self):
        """
        The environment_code's observation space.
        A tuple containing a tensor shape.
        """
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_actions(self):
        """
        The number of available actions in a discrete environment.
        """
        return self._num_actions

    def describe_action(self, action):
        return self._action_descriptions[action]

    @property
    def observation(self):
        """
        The current observation.
        """
        return self._observation if self._observation is not None else self.reset()

    def reset(self):
        """
        Resets the environment_code to a given initial state.
        """
        self._observation = self._reset()
        self._is_terminal = False
        self._episode_timesteps = 0
        if self._should_render: self._render()
        return self._observation

    def step(self, action):

        if self._is_terminal: self._observation = self.reset()

        next_observation, reward, is_terminal, info = self._step(action)
        timestep = Timestep(self._episode_timesteps, self.observation, action, reward, next_observation, is_terminal, info)

        self._observation = next_observation
        self._is_terminal = is_terminal

        if self._should_render: self._render()

        self._episode_timesteps += 1

        return timestep

    @abstractmethod
    def _reset(self):
        """
        Template Method for self.reset.
        Overwritten by sub-classes.
        :return: The initial observation o0
        """
        raise NotImplementedError()

    @abstractmethod
    def _step(self, action):
        """
        Template method for step.
        """
        raise NotImplementedError()

    @abstractmethod
    def _render(self):
        """
        Template method for render.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """
        Closes the environment.
        """
        raise NotImplementedError()
