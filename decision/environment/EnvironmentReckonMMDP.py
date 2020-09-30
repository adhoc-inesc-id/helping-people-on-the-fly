from itertools import product
from typing import Sequence

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from yaaf import ndarray_index_from
from yaaf.environments.markov import MarkovDecisionProcess


class EnvironmentReckonMMDP(MarkovDecisionProcess):

    def __init__(self,
                 adjacency_matrix: np.ndarray,
                 explorable_nodes: Sequence[int],
                 movement_failure_probability: float,
                 discount_factor: float = 0.90,
                 min_value_iteration_error: float = 10e-8,
                 initial_state: np.ndarray = np.array([0, 0, 1, 0, 0]),
                 node_meanings: Sequence[str] = ("door", "baxter", "single workbench", "double workbench", "table"),
                 id="environment-reckon-mmdp-v1"):

        self._adjacency_matrix = adjacency_matrix
        self._explorable_nodes = explorable_nodes
        self._movement_failure_probability = movement_failure_probability
        self._node_meanings = node_meanings
        self._initial_state = initial_state

        # States
        states = self.generate_states(adjacency_matrix, explorable_nodes)

        # Actions
        self.individual_action_meanings = self.generate_action_meanings()
        self._joint_actions = list(np.array(list(product(range(len(self.individual_action_meanings)), repeat=2))))
        actions = tuple(range(len(self._joint_actions)))
        action_meanings = tuple(product(self.individual_action_meanings, repeat=2))

        # Transitions
        transition_probabilities = self.generate_transition_probabilities(
            states, self._joint_actions, adjacency_matrix, explorable_nodes, movement_failure_probability, self.individual_action_meanings
        )

        # Reward
        reward_matrix = self.generate_reward_matrix(states, actions)

        # Miu
        initial_state_distribution = np.zeros(len(states))
        initial_state_distribution[self.state_index_from(states, initial_state)] = 1

        super(EnvironmentReckonMMDP, self).__init__(
            id, states, actions, transition_probabilities, reward_matrix,
            discount_factor, initial_state_distribution, min_value_iteration_error,
            action_meanings=action_meanings)

    def step(self, action):
        next_state, reward, _, info = super().step(action)
        is_terminal = reward == 1.0
        return next_state, reward, is_terminal, info

    def joint_step(self, action_robot, action_human):
        joint_action = self.joint_action(action_robot, action_human)
        return self.step(joint_action)

    def joint_action(self, action_robot, action_human):
        if isinstance(action_robot, int):
            action_robot = self.individual_action_meanings[action_robot]
        if isinstance(action_human, int):
            action_human = self.individual_action_meanings[action_human]
        return self.action_meanings.index((action_robot, action_human))

    # ################## #
    # Auxiliary for init #
    # ################## #

    def generate_states(self, adjacency_matrix, explorable_nodes):
        num_explorable_nodes = len(explorable_nodes)
        num_nodes = adjacency_matrix.shape[0]
        num_agents = 2
        possible_bit_combinations = list(product(range(num_agents), repeat=num_explorable_nodes))
        states = [
            np.array([x_robot, x_human, *explorable_node_bits])
            for x_robot in range(num_nodes)
            for x_human in range(num_nodes)
            for explorable_node_bits in possible_bit_combinations
        ]
        valid_states = []
        for x_robot in range(num_nodes):
            for x_human in range(num_nodes):
                for x, state in enumerate(states):
                    if state[0] == x_robot and state[1] == x_human:
                        explorable_bits = state[num_agents:]
                        robot_in_explorable = x_robot in explorable_nodes
                        human_in_explorable = x_human in explorable_nodes
                        invalid_robot_node_state = robot_in_explorable and explorable_bits[explorable_nodes.index(x_robot)] == 0
                        invalid_human_node_state = human_in_explorable and explorable_bits[explorable_nodes.index(x_human)] == 0
                        if not invalid_robot_node_state and not invalid_human_node_state:
                            valid_states.append(x)
        states = [states[x] for x in valid_states]
        return states

    def generate_action_meanings(self):
        individual_action_meanings = (
            "move to lower-index node",
            "move to second-lower-index node",
            "move to third-lower-index node",
            "stay",
        )
        return individual_action_meanings

    def generate_transition_probabilities(self, states, joint_actions, adjacency_matrix, explorable_nodes, movement_failure_probability, individual_action_meanings):
        num_joint_actions = len(joint_actions)
        num_explorable_nodes = len(explorable_nodes)
        num_states = len(states)
        P = np.zeros((num_joint_actions, num_states, num_states))
        for state in states:
            x = self.state_index_from(states, state)
            x_robot, x_human, already_explored_bits = state[0], state[1], state[2:]
            for a in range(num_joint_actions):
                a_robot, a_human = joint_actions[a]
                if "move" in individual_action_meanings[a_robot]:
                    adjacencies = np.where(adjacency_matrix[x_robot]==1)[0]
                    downgrade_to_lower_index = int(a_robot) >= len(adjacencies)
                    a_robot = 0 if downgrade_to_lower_index else a_robot
                    next_x_robot = adjacencies[a_robot]
                    robot_transitions = {x_robot: movement_failure_probability, next_x_robot: 1.0 - movement_failure_probability}
                else:
                    robot_transitions = {x_robot: 1.0}
                if "move" in individual_action_meanings[a_human]:
                    adjacencies = np.where(adjacency_matrix[x_human]==1)[0]
                    downgrade_to_lower_index = int(a_human) >= len(adjacencies)
                    a_human = 0 if downgrade_to_lower_index else a_human
                    next_x_human = adjacencies[a_human]
                    human_transitions = {next_x_human: 1.0}
                else:
                    human_transitions = {x_human: 1.0}
                for next_x_robot, robot_prob in robot_transitions.items():
                    for next_x_human, human_prob in human_transitions.items():
                        next_state = np.array([next_x_robot, next_x_human]+[0 for _ in range(num_explorable_nodes)])
                        # If any moved into explorable location, set to 1 the explored bit
                        for explorable_node in explorable_nodes:
                            if next_x_robot == explorable_node or next_x_human == explorable_node:
                                bit_to_turn_on = explorable_nodes.index(explorable_node) + 2
                                next_state[bit_to_turn_on] = 1.0
                        for b, bit in enumerate(already_explored_bits):
                            b = b + 2
                            if bit == 1:
                                next_state[b] = 1
                        y = self.state_index_from(states, next_state)
                        P[a, x, y] = robot_prob * human_prob  # Independent events
        return P

    def generate_reward_matrix(self, states, actions):
        num_states = len(states)
        num_actions = len(actions)
        R = np.zeros((num_states, num_actions))
        for x, state in enumerate(states):
            if (state[2:]==1).all():
                R[x, :] = 1.0
        return R

    # ########## #
    # Draw utils #
    # ########## #

    def show_topological_map(self):
        self.draw_state(self._initial_state, self.spec.id)

    def draw_state(self, state, title=None):
        graph = nx.DiGraph()
        labels = {}
        x_robot, x_human, explored_bits = state[0], state[1], state[2:]
        num_nodes = self._adjacency_matrix.shape[0]
        colors = []
        for n in range(num_nodes):
            graph.add_node(n)
            label = self._node_meanings[n].replace(' ', '\n')
            labels[n] = f"[{n}: {label}]"
            if x_robot == n:
                labels[n] += f"\nR"
            if x_human == n:
                label = "\nH" if 'R' not in labels[n] else ', H'
                labels[n] += f"{label}"

            if n in self._explorable_nodes:
                if explored_bits[self._explorable_nodes.index(n)] == 1:
                    colors.append("lightgreen")
                    labels[n] = labels[n].replace("]", "]\n(explored)")
                else:
                    colors.append("lightgray")
                    labels[n] = labels[n].replace("]", "]\n(unexplored)")
            else:
                colors.append("white")

        rows, cols = np.where(self._adjacency_matrix == 1)
        graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
        plt.figure()
        if not hasattr(self, "_node_draw_pos"):
            self._node_draw_pos = nx.spring_layout(graph)
        fig = nx.draw_networkx_nodes(graph, node_color=colors, pos=self._node_draw_pos, node_size=8000)
        fig.set_edgecolor('k')
        plt.gcf().set_size_inches(10, 10, forward=True)
        nx.draw_networkx_edges(graph, self._node_draw_pos, width=1.0, node_size=8000, arrowsize=10)
        nx.draw_networkx_labels(graph, self._node_draw_pos, labels, font_size=12)
        plt.axis('off')
        plt.title(title or f"State: {state}")
        plt.show()
        plt.close()

    def render(self, mode='human'):
        self.draw_state(self.state, title=f"{self.spec.id}\n{self.state}")

    def joint_action_index(self, joint_action):
        return ndarray_index_from(self._joint_actions, joint_action)