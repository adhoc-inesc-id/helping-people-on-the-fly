from itertools import product
from typing import Sequence

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from environment.EnvironmentReckonMMDP import EnvironmentReckonMMDP


class GarbageCollectionMMDP(EnvironmentReckonMMDP):

    def __init__(self, adjacency_matrix: np.ndarray, explorable_nodes: Sequence[int],
                 movement_failure_probability: float, discount_factor: float = 0.90,
                 min_value_iteration_error: float = 10e-8, initial_state: np.ndarray = np.array([0, 0, 0, 0]),
                 node_meanings: Sequence[str] = ("door", "baxter", "single workbench", "double workbench", "table")):

        super().__init__(adjacency_matrix, explorable_nodes, movement_failure_probability, discount_factor,
                         min_value_iteration_error, initial_state, node_meanings, "garbage-collection-mmdp-v1")

    # ################## #
    # Auxiliary for init #
    # ################## #

    def generate_states(self, adjacency_matrix, dirty_nodes):
        num_dirty_nodes = len(dirty_nodes)
        num_nodes = adjacency_matrix.shape[0]
        num_agents = 2
        possible_bit_combinations = list(product(range(num_agents), repeat=num_dirty_nodes))
        states = [
            np.array([x_robot, x_human, *dirty_bits])
            for x_robot in range(num_nodes)
            for x_human in range(num_nodes)
            for dirty_bits in possible_bit_combinations
        ]
        valid_states = []
        for x_robot in range(num_nodes):
            for x_human in range(num_nodes):
                for x, state in enumerate(states):
                    if state[0] == x_robot and state[1] == x_human:
                        dirty_bits = state[num_agents:]
                        robot_in_dirty_node = x_robot in dirty_nodes
                        human_in_same = x_human == x_robot
                        invalid_state = robot_in_dirty_node and human_in_same and dirty_bits[dirty_nodes.index(x_robot)] == 0
                        if not invalid_state :
                            valid_states.append(x)
        states = [states[x] for x in valid_states]
        return states

    def generate_transition_probabilities(self, states, joint_actions, adjacency_matrix, dirty_nodes, movement_failure_probability, individual_action_meanings):
        num_joint_actions = len(joint_actions)
        num_dirty_nodes = len(dirty_nodes)
        num_states = len(states)
        P = np.zeros((num_joint_actions, num_states, num_states))
        for state in states:
            x = self.state_index_from(states, state)
            x_robot, x_human, already_clean_bits = state[0], state[1], state[2:]
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
                        next_state = np.array([next_x_robot, next_x_human]+[0 for _ in range(num_dirty_nodes)])
                        # If any moved into explorable location, set to 1 the explored bit
                        for dirty_node in dirty_nodes:
                            if next_x_robot == dirty_node and next_x_human == dirty_node:
                                bit_to_turn_on = dirty_nodes.index(dirty_node) + 2
                                next_state[bit_to_turn_on] = 1.0
                        for b, bit in enumerate(already_clean_bits):
                            b = b + 2
                            if bit == 1:
                                next_state[b] = 1
                        y = self.state_index_from(states, next_state)
                        P[a, x, y] = robot_prob * human_prob  # Independent events
        return P

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
                    labels[n] = labels[n].replace("]", "]\n(clean)")
                else:
                    colors.append("red")
                    labels[n] = labels[n].replace("]", "]\n(dirty)")
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
