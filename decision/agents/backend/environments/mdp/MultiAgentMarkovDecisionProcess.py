from abc import ABC

import numpy as np

from agents.backend.environments.mdp.MarkovDecisionProcess import MarkovDecisionProcess


class MultiAgentMarkovDecisionProcess(MarkovDecisionProcess, ABC):

    def __init__(self, name, num_teammates,
                 state_space, disjoint_action_space,
                 transition_probabilities, rewards,
                 action_descriptions, render):

        self._num_agents = num_teammates + 1
        self._num_disjoint_actions = len(disjoint_action_space)
        self._num_joint_actions = self._num_disjoint_actions ** self._num_agents

        joint_action_space = self._setup_joint_action_space(self._num_agents, disjoint_action_space)

        assert len(joint_action_space) == self._num_joint_actions

        super(MultiAgentMarkovDecisionProcess, self).__init__(name, state_space, joint_action_space,
                                                              transition_probabilities, rewards,
                                                              action_descriptions, render)

        self._teammates = []

    def step(self, action):
        timestep = super().step(action)
        [teammate.reinforcement(timestep) for teammate in self._teammates]
        return timestep

    def _step(self, action):
        state = self.state
        teammates_actions = [teammate.action(state) for teammate in self._teammates]
        joint_actions = tuple([action] + teammates_actions)
        joint_action = self.action_space.index(joint_actions)
        next_state, reward, is_terminal, _ = self.transition(state, joint_action)
        joint_actions = {f"agent {i}": int(action) for i, action in enumerate(joint_actions)}
        return next_state, reward, is_terminal, {"Joint actions": joint_actions}

    def disjoint_pi_star(self, agent_index):
        if not hasattr(self, "_disjoint_pi_star"):
            pi_star = self.pi_star
            self._disjoint_pi_star = np.zeros((self.S, self.num_disjoint_actions))
            for s in range(self.S):
                pi_s = pi_star[s]
                for a, action_probability in enumerate(pi_s):
                    optimal_joint_action = self.action_space[a]
                    optimal_action = optimal_joint_action[agent_index]
                    self._disjoint_pi_star[s, optimal_action] += action_probability
        return self._disjoint_pi_star

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def num_teammates(self):
        return self._num_agents - 1

    @property
    def num_joint_actions(self):
        """ Alias for super().num_actions """
        return self.num_actions

    @property
    def joint_action_space(self):
        """ Alias for super().action_space """
        return self.action_space

    @property
    def num_disjoint_actions(self):
        return self._num_disjoint_actions

    def add_teammate(self, teammate):
        assert len(self._teammates) < self._num_agents - 1, "Maximum number of agents reached"
        self._teammates.append(teammate)

    @staticmethod
    def _setup_joint_action_space(num_agents, disjoint_action_space):

        joint_action_space = []

        for _ in range(num_agents):

            auxiliary = []

            if len(joint_action_space) == 0: # First action

                for a0 in range(len(disjoint_action_space)):
                    auxiliary.append([a0])

            else:

                for a in joint_action_space:

                    for a0 in range(len(disjoint_action_space)):
                        new_a = a + [a0]
                        auxiliary.append(tuple(new_a))

            joint_action_space = auxiliary

        return tuple(joint_action_space)
