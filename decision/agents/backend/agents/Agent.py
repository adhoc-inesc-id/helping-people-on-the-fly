import pathlib
from abc import ABC, abstractmethod

from decision.agents.backend.policies import sample_action
from decision.agents.backend import Timestep


class Agent(ABC):

    """
    Agent class.
    Represents the concept of an autonomous agent.
    """

    def __init__(self, name):

        """
        Constructor
        :param: name (str) - Name of the agent.
        """
        self.name = name
        self._trainable = True
        self._total_timesteps = 0
        self._total_episodes = 0

    # ############## #
    # Main Interface #
    # ############## #

    @abstractmethod
    def policy(self, observation):
        """
        Returns the agent's policy, i.e., distribution over possible actions,
        for a given observation of the environment.
        :param: observation (ndarray) - Observation of the environment_code to act upon.
        :return: policy (ndarray) - Distribution over possible actions.
        """
        raise NotImplementedError()

    def action(self, observation):
        """
        Returns the agent's action for a given observation of the environment.
        :param: observation (ndarray) - Observation of the environment_code to act upon.
        :return: action (int/float) - Action to execute upon the environment.
        """
        policy = self.policy(observation)
        action = sample_action(policy)
        return action

    def reinforcement(self, timestep: Timestep):

        """
        Provides reinforcement for the agent
        :param: timestep (namedtuple) - A tuple containing:
            t (int) - The index of the timestep in an episode.
            observation (ndarray) - Observation of the environment_code before transitioning.
            action (int/float) - Action executed upon the environment.
            reward (float) - Reward obtained by transitioning.
            next_observation (ndarray) - Observation of the environment_code after transitioning.
            is_terminal (bool) - True if episode ended after transitioning.
            info (dict) - Additional information from the environment.
        """

        if self.trainable:

            self._total_timesteps += 1

            if timestep.is_terminal:
                self._total_episodes += 1

            # Any relevant info regarding updates (such as model losses, etc...)
            agent_info = self._reinforce(timestep)

            if agent_info is not None:
                timestep.info["agent"] = agent_info

    def train(self):
        """
        Enables the agent's training mode.
        When in training mode, the agent can learn from new experience (given through the reinforcement method).
        """
        self._trainable = True

    def eval(self):
        """
        Enables the agent's evaluation mode.
        When in evaluation mode, the agent doesn't learn from new experience (even if given through the
        reinforcement method).
        """
        self._trainable = False

    def save(self, directory):
        """
        Saves the agent's state into to a given directory.
        :param: directory (str): Path of the directory to save the agent's state into.
        """
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    def load(self, directory):
        """
        Loads an agent's state from a given directory.
        :param: directory (str): Path of the directory to save the agent's state into.
        """
        pass

    ##############
    # Attributes #
    ##############

    @property
    def params(self):
        """
        Property (dict). Contains all relevant information regarding the agent's parameters (e.g. learning rate).
        """
        return dict(name=self.name, trainable=self.trainable)

    @property
    def trainable(self):
        """
        Property (bool). True if agent is in training mode, False if in evaluation mode.
        """
        return self._trainable

    @property
    def total_timesteps(self):
        """
        Property (int). The total number of timesteps given to the agent through the reinforcement method.
        """
        return self._total_timesteps

    @property
    def total_episodes(self):
        """
        Property (int). The total number of episodes given to the agent through the reinforcement method.
        An episode is counted whenever a timestep is terminal.
        """
        return self._total_episodes

    # ################ #
    # Abstract Methods #
    # ################ #

    @abstractmethod
    def _reinforce(self, timestep: Timestep):
        """
        Template Method for self.reinforce
        Overwritten in sub-classes.
        """
        raise NotImplementedError()
