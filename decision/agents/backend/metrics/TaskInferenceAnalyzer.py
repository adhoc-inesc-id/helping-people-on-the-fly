import numpy as np

from decision.agents.backend.metrics.Metric import Metric


class TaskInferenceAnalyzer(Metric):

    def __init__(self):
        super().__init__("Task Inference Analyzer")

    def reset(self):
        del self._belief_history

    def result(self):
        return np.array(self._belief_history)

    def _process(self, timestep):
        agent_info = timestep.info["agent"]
        num_possible_tasks = len(agent_info.keys())
        if not hasattr(self, "_belief_history"):
            self._belief_history = [[1/num_possible_tasks for _ in range(num_possible_tasks)]]
        beliefs = [agent_info[f"MMDP #{i}'s Likelihood"] for i in range(num_possible_tasks)]
        self._belief_history.append(beliefs)
