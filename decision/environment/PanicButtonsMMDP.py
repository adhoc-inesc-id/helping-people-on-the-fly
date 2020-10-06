from itertools import product

import numpy as np
from yaaf.agents import RandomAgent

from agents.GreedyDuoAgent import GreedyDuoAgent
from agents.SubOptimalAgent import SubOptimalAgent
from agents.backend.environments.mdp.MarkovDecisionProcess import state_index_from
from agents.backend.environments.mdp.MultiAgentMarkovDecisionProcess import MultiAgentMarkovDecisionProcess


class PanicButtonsMMDP(MultiAgentMarkovDecisionProcess):

    def __init__(self, rows=3, columns=3, initial_position=(0, 0, 0, 1), goal=(0, 2, 2, 2), teammate="greedy", render=False, P=None, R=None):

        num_teammates = 1
        self.rows = rows
        self.columns = columns
        self.initial_position = initial_position
        self.goal = goal
        self.teammate = teammate

        action_descriptions = ("Stay", "Right", "Left", "Down", "Up")
        self.joint_action_meanings = tuple(product(action_descriptions, repeat=2))
        self.action_meanings = action_descriptions
        disjoint_action_space = tuple(range(len(action_descriptions)))
        joint_action_space = self._setup_joint_action_space(num_agents=2, disjoint_action_space=disjoint_action_space)

        state_space = self._generate_states()

        transition_probabilities = self._generate_transition_probabilities_matrix(state_space, joint_action_space) if P is None else P
        if R is None:
            rewards = self._generate_reward_matrix(state_space, goal)
        else:
            rewards = R
            self.goal_state = np.array(goal)

        super(PanicButtonsMMDP, self).__init__("panic buttons mmdp", num_teammates,
                                               state_space, disjoint_action_space,
                                               transition_probabilities, rewards,
                                               action_descriptions, render)

        if teammate == "greedy":
            self.add_teammate(GreedyDuoAgent(1, self))
        elif teammate == "suboptimal":
            self.add_teammate(SubOptimalAgent(1, self))
        elif teammate == "random":
            self.add_teammate(RandomAgent(len(disjoint_action_space)))
        else:
            raise ValueError(f"Invalid teammate type {teammate}. Available teammates: [greedy], [suboptimal] and [random]")

    @property
    def stay(self):
        return self.action_meanings.index("Stay")

    #################
    # MDP Interface #
    #################

    def initial_state(self):
        state = np.array(self.initial_position)
        return state

    def is_terminal_state(self, state):
        return self.R[self.state_index(state)] == 0.0

    #############
    # AUXILIARY #
    #############

    def _move_agents(self, state, joint_actions):

        a0 = joint_actions[0]
        a1 = joint_actions[1]

        delta_a0 = self.direction(a0)
        delta_a1 = self.direction(a1)

        x0 = (state[0] + delta_a0[0])
        x0 = min(x0, self.columns - 1)
        x0 = max(x0, 0)

        y0 = (state[1] + delta_a0[1])
        y0 = min(y0, self.rows - 1)
        y0 = max(y0, 0)

        x1 = (state[2] + delta_a1[0])
        x1 = min(x1, self.columns - 1)
        x1 = max(x1, 0)

        y1 = (state[3] + delta_a1[1])
        y1 = min(y1, self.rows - 1)
        y1 = max(y1, 0)

        next_state = np.array([x0, y0, x1, y1])

        return next_state

    @staticmethod
    def direction(a):
        actions = [
            (0, 0),     # Stay
            (1, 0),     # Right
            (-1, 0),    # Left
            (0, 1),     # Down
            (0, -1)     # Up
        ]
        return actions[a]

    def _generate_states(self):
        return [np.array([x0, y0, x1, y1])
                for x0 in range(self.columns)
                for x1 in range(self.columns)
                for y0 in range(self.rows)
                for y1 in range(self.rows)]

    def _generate_transition_probabilities_matrix(self, state_space, joint_action_space):
        S = len(state_space)
        A = len(joint_action_space)
        P = np.zeros((A, S, S))
        for a, joint_action in enumerate(joint_action_space):
            for x, state in enumerate(state_space):
                next_state = self._move_agents(state, joint_action)
                y = state_index_from(state_space, next_state)
                P[a, x, y] = 1.0
        return P

    def _generate_reward_matrix(self, state_space, goal):

        num_states = len(state_space)
        self.goal_state = np.array(goal)

        R = np.full(num_states, fill_value=-1.0)
        R[state_index_from(state_space, self.goal_state)] = 0.0

        return R

    #########################
    # Environment Interface #
    #########################

    def _render(self):
        print(f"{self._draw_state(self.state)}")

    def _draw_state(self, state):

        display = ""

        display += f" {' '.join([f' {i} ' for i in range(self.columns)])}\n"

        for row in range(self.rows):

            display += self._draw_row_border(self.columns)

            display += "|"
            for col in range(self.columns):
                cell = self._who_in(state, row, col)
                display += " " if col == 0 else "+ "
                display += f"{cell} "
            display += f"| {row}\n"

        display += self._draw_row_border(self.columns)

        return display

    def _who_in(self, state, column, row):

        robot_row, robot_column, human_row, human_column = state

        panic1_row, panic1_column, panic2_row, panic2_column = self.goal_state

        if robot_row == row and robot_column == column:
            return 'R'
        elif human_row == row and human_column == column:
            return 'H'
        elif row == panic1_row and column == panic1_column or row == panic2_row and column == panic2_column:
            return "*"
        else:
            return " "

    @staticmethod
    def _draw_row_border(columns):
        border = ""
        for y in range(columns):
            border += "+---"
        border += "+\n"
        return border

    def close(self):
        pass

    def display_goal(self):
        print(self._draw_state(self.initial_state()))
