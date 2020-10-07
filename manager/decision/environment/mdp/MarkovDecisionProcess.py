import math
from abc import ABC, abstractmethod

import numpy as np

from environment.Environment import Environment


class MarkovDecisionProcess(Environment, ABC):

    def __init__(self, name, state_space, action_space, transition_probabilities, rewards, action_descriptions, render,
                 discount_factor=0.95, min_value_iteration_error=10e-8):

        self._state_space = state_space
        self._num_states = len(self._state_space)

        self._P = transition_probabilities
        self._R = rewards

        example_state = self._state_space[0]
        observation_space = example_state.shape

        self._discount_factor = discount_factor
        self._min_value_iteration_error = min_value_iteration_error

        super(MarkovDecisionProcess, self).__init__(name, observation_space, action_space, render, action_descriptions)

    @property
    def state(self):
        """Returns the current state st. Alias for super().observation"""
        return self.observation

    @property
    def state_space(self):
        """Returns the state space"""
        return self._state_space

    @property
    def num_states(self):
        """Returns the total number of states X"""
        return self._num_states

    @property
    def S(self):
        """Alias for self.num_states"""
        return self.num_states

    @property
    def A(self):
        """Alias for self.num_actions"""
        return self.num_actions

    @property
    def transition_probabilities(self):
        """Returns the Transition Probabilities P (array w/ shape X, X)"""
        return self._P

    @property
    def P(self):
        """Alias for self.transition_probabilities"""
        return self.transition_probabilities

    @property
    def rewards(self):
        """Returns the Rewards R (array w/ shape X, A)"""
        return self._R

    @property
    def R(self):
        """Alias for self.rewards"""
        return self.rewards

    def transition(self, state, action):
        """
        Default implementation for an MDP (using P and r directly).
            Given st and at, transitions st into a st+1,
            and returning a both st+1 and rt.
            Override if necessary.
        """
        s1 = self.state_index(state)
        transition_probabilities = self.P[action, s1]
        s2 = np.random.choice(self.S, p=transition_probabilities)
        next_state = self.state_space[s2]
        if self.R.shape == (self.S, self.A):
            reward = self.R[s1, action]
        elif self.R.shape == (self.S,):
            reward = self.R[s2]
        elif self.R.shape == (self.S, self.A, self.S):
            reward = self.R[s1, action, s2]
        else: raise ValueError("Invalid reward matrix R.")
        is_terminal = self.is_terminal_state(next_state)
        extra_info = None
        return next_state, reward, is_terminal, extra_info

    @property
    def pi_star(self):
        """
            Computes (or returns, if already computed) the optimal policy for the MDP using value iteration.
            :return pi_star - The optimal policy for the MDP
        """
        if not hasattr(self, "_pi_star"):
            self._pi_star = np.zeros((self.S, self.A))
            for s in range(self.S):
                optimal_actions = np.argwhere(self.q_star[s] == self.q_star[s].max()).reshape(-1)
                self._pi_star[s, optimal_actions] = 1.0 / len(optimal_actions)
        return self._pi_star

    def v_pi(self, pi):
        """
        Evaluates a given policy pi in the MDP.
        :param pi - The
        :return:
        """
        V_pi = np.zeros(self.S)
        q = self.q_star
        for s in range(self.S):
            V_pi[s] = pi[s].dot(q[s])
        return V_pi

    @property
    def q_star(self):
        if not hasattr(self, "_Qstar"): self._Qstar, self._V = self._value_iteration()
        return self._Qstar

    @property
    def v(self):
        if not hasattr(self, "_V"): self._Qstar, self._V = self._value_iteration()
        return self._V

    def _value_iteration(self):

        """
        Solves the MDP using value iteration
        Returns the Optimal Q function Q*
        """

        A = self.num_actions
        X = self.num_states
        P = self.P

        if self.R.shape == (X, A):
            R = self.R
        elif self.R.shape == (X,):
            # FIXME - Find clever way
            R = np.zeros((X, A))
            for state in self.state_space:
                s = self.state_index(state)
                R[s, :] = self.R[s]
        elif self.R.shape == (X, A, X):
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError("Invalid reward matrix.")

        V = np.zeros(X)
        Q = np.zeros((X, A))

        error = math.inf
        while error > self._min_value_iteration_error:
            for a in range(A):
                Q[:, a] = R[:, a] + self._discount_factor * P[a].dot(V)
            Qa = tuple([Q[:, a] for a in range(A)])
            V_new = np.max(Qa, axis=0)
            error = np.linalg.norm(V_new - V)
            V = V_new

        return Q, V

    def state_index(self, state=None):
        """
            Returns the index of a given state in the state space.
            If the state is unspecified (None), returns the index of the current state st.
        """
        return self.state_index(self.state) if state is None else state_index_from(self.state_space, state)

    # ######### #
    # Interface #
    # ######### #

    @abstractmethod
    def initial_state(self):
        """Returns a initial state"""
        raise NotImplementedError()

    @abstractmethod
    def is_terminal_state(self, state):
        """Returns True if state is terminal, False otherwise"""
        raise NotImplementedError()

    # ##################### #
    # Environment Interface #
    # ##################### #

    def _step(self, action):
        next_state, reward, is_terminal, extra_info = self.transition(self.state, action)
        return next_state, reward, is_terminal, extra_info

    def _reset(self):
        return self.initial_state()


def state_index_from(state_space, state):
    """Returns the index of a state (array) in a list of states"""
    return [np.array_equal(state, other_state) for other_state in state_space].index(True)
