from collections import namedtuple

Timestep = namedtuple("Timestep", "t observation action reward next_observation is_terminal info")
from agents.backend.TaskInferenceAnalyzer import TaskInferenceAnalyzer
