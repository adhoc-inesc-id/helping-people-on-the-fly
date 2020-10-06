from unittest import TestCase

import numpy as np
from yaaf import Timestep
from yaaf.agents import GreedyAgent
from yaaf.evaluation import TimestepsPerEpisodeMetric

from agents.BOPA import BOPA
from agents.GreedyDuoAgent import GreedyDuoAgent
from environment.EnvironmentReckonMMDP import EnvironmentReckonMMDP
from environment.GarbageCollectionMMDP import GarbageCollectionMMDP
from environment.PanicButtonsMMDP import PanicButtonsMMDP


def run_episode(agent, environment, observers):
    terminal = False
    environment.reset()
    while not terminal:
        action = agent.action(environment.state)
        timestep = environment.step(action)
        terminal = timestep.is_terminal
        agent.reinforcement(timestep)
        [observer(timestep) for observer in observers]

class GAIPS2N73Tests(TestCase):

    def __init__(self, *args, **kwargs):

        super(GAIPS2N73Tests, self).__init__(*args, **kwargs)

        self._adjacency_matrix = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ])
        self._nodes_to_explore = [0, 1, 4]
        self._dirty_nodes = [2, 4]
        self._movement_failure_probability = 0.0
        self._dead_reckoning_failure_probability = 0.0
        self._transmission_failure_probability = 0.0

    def test_environment_reckon_mmdp_value_iteration(self):
        mmdp = EnvironmentReckonMMDP(self._adjacency_matrix, self._nodes_to_explore,
                                     self._movement_failure_probability)
        self._test_mmdp_value_iteration(mmdp, 4)

    def test_garbage_collection_mmdp_value_iteration(self):
        mmdp = GarbageCollectionMMDP(self._adjacency_matrix, self._dirty_nodes, self._movement_failure_probability)
        self._test_mmdp_value_iteration(mmdp, 6)

    def _test_mmdp_value_iteration(self, mmdp, optimal_steps):
        agent = GreedyAgent(mmdp)
        metric = TimestepsPerEpisodeMetric()
        terminal = False
        step = 0
        while not terminal:
            state = mmdp.state
            action = agent.action(state)
            _, reward, terminal, info = mmdp.step(action)
            next_state = mmdp.state
            timestep = Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            metric(timestep)
            step += 1
        assert step <= optimal_steps and len(metric.result()) == 1 and metric.result()[
            0] == optimal_steps, f"Failed to solve task in at most {optimal_steps} steps (took {metric.result()[0]})"


class DuoGridWorldTests(TestCase):

    def test_greedy(self, rows=3, columns=3, goal=(0, 2, 2, 2)):
        env = PanicButtonsMMDP(rows, columns, goal=goal)
        agent = GreedyDuoAgent(0, env)
        metric = TimestepsPerEpisodeMetric()
        run_episode(agent, env, [metric])
        steps_to_solve = metric.result()[-1]
        return steps_to_solve

    def test_adhoc_known_task(self, rows=3, columns=3, goal=(0, 2, 2, 2)):
        greedy_steps = self.test_greedy(rows, columns, goal)
        env = PanicButtonsMMDP(rows, columns, goal=goal)
        agent = BOPA([env], 0)
        metric = TimestepsPerEpisodeMetric()
        run_episode(agent, env, [metric])
        steps_to_solve = metric.result()[-1]
        assert greedy_steps == steps_to_solve, f"When knowing the task, ad hoc should solve it in {greedy_steps} steps (got {steps_to_solve})."

    def test_top_right_bottom_left(self):
        self.test_adhoc_known_task(goal=(0, 2, 2, 0))

    def test_bottom_right_bottom_left(self):
        self.test_adhoc_known_task(goal=(0, 2, 2, 2))
