from unittest import TestCase

from yaaf.evaluation import TimestepsPerEpisodeMetric
from environment.PanicButtonsMMDP import PanicButtonsMMDP
from agents.BOPA import BOPA
from agents.GreedyDuoAgent import GreedyDuoAgent
from run_full_empirical_evaluation import task_factory, run

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
        self._possible_tasks = task_factory("gaips", "small", "greedy")

    def _test_known_task(self, task):

        greedy = GreedyDuoAgent(0, task)
        result_greedy = run(greedy, task)

        bopa = BOPA([task])
        result_bopa = run(bopa, task)

        assert result_bopa == result_greedy, f"Greedy solved in {result_greedy} steps while BOPA solved in {result_bopa} steps"

    def test_env_reckon_known_task1(self):
        self._test_known_task(self._possible_tasks[0])

    def test_env_reckon_known_task2(self):
        self._test_known_task(self._possible_tasks[1])

    def test_env_reckon_known_task3(self):
        self._test_known_task(self._possible_tasks[2])

    def _test_env_reckon_bopa_task(self, task_no):
        task = self._possible_tasks[task_no]
        bopa = BOPA(self._possible_tasks)
        result_bopa = run(bopa, task)
        assert result_bopa <= 10, f"BOPA took more than 10 steps to solve task #{task_no} (took {result_bopa})"

    def test_env_reckon_bopa_task1(self):
        self._test_env_reckon_bopa_task(0)

    def test_env_reckon_bopa_task2(self):
        self._test_env_reckon_bopa_task(1)

    def test_env_reckon_bopa_task3(self):
        self._test_env_reckon_bopa_task(2)

class DuoGridWorldTests(TestCase):

    def test_greedy(self, n=3, config=1):
        env = PanicButtonsMMDP(n, "greedy", config)
        agent = GreedyDuoAgent(0, env)
        metric = TimestepsPerEpisodeMetric()
        run_episode(agent, env, [metric])
        steps_to_solve = metric.result()[-1]
        return steps_to_solve

    def test_adhoc_known_task(self, n=3, config=1):
        greedy_steps = self.test_greedy(n, config)
        env = PanicButtonsMMDP(n, "greedy", config)
        agent = BOPA([env], 0)
        metric = TimestepsPerEpisodeMetric()
        run_episode(agent, env, [metric])
        steps_to_solve = metric.result()[-1]
        assert greedy_steps == steps_to_solve, f"When knowing the task, ad hoc should solve it in {greedy_steps} steps (got {steps_to_solve})."

    def test_top_right_bottom_left(self):
        self.test_adhoc_known_task(config=2)

    def test_bottom_right_bottom_left(self):
        self.test_adhoc_known_task(config=3)
