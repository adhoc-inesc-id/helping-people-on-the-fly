import time
from random import getrandbits

import numpy as np
import yaaf
from yaaf.agents import RandomAgent
from yaaf.evaluation import TimestepsPerEpisodeMetric

from agents.BOPA import BOPA
from agents.GreedyDuoAgent import GreedyDuoAgent
from agents.backend import TaskInferenceAnalyzer
from tasks import SmallPanicButtons, MediumPanicButtons, LargePanicButtons

# ######### #
# Utilities #
# ######### #

def run(agent, task, observers=None, max_steps=5000):
    observers = observers or [TimestepsPerEpisodeMetric()]
    terminal = False
    task.reset()
    step = 0
    while not terminal:
        action = agent.action(task.state)
        timestep = task.step(action)
        terminal = timestep.is_terminal
        agent.reinforcement(timestep)
        [observer(timestep) for observer in observers]
        step += 1
        if step >= max_steps:
            break
    return observers[0].result()[0] if step < max_steps else np.array(max_steps)

def fetch_needed_runs(domain, size, teammate, task, agent, N):
    dir = f"resources/results/{domain}/{teammate}/{size}/task_{task}/{agent}"
    try:
        files = yaaf.files(dir)
        num_files = len([file for file in files if ".npy" in file])
        return num_files, max(0, N - num_files)
    except FileNotFoundError:
        return 0, N

def agent_factory(name, env, tasks):
    if name == "greedy": return GreedyDuoAgent(0, env)
    elif name == "random": return RandomAgent(env.num_disjoint_actions)
    elif name == "bopa": return BOPA(tasks, 0)
    else: raise ValueError()

def task_factory(size, teammate):
    if size == "small": return [SmallPanicButtons(teammate, config) for config in (1, 2, 3)]
    elif size == "medium": return [MediumPanicButtons(teammate, config) for config in (1, 2, 3)]
    elif size == "large": return [LargePanicButtons(teammate, config) for config in (1, 2, 3)]
    else: raise ValueError()


# #### #
# Main #
# #### #

if __name__ == '__main__':


    N = 32
    domain = "panic buttons"
    agents = ("greedy", "bopa", "random")
    sizes = ("small", "medium", "large")
    teammates = ("greedy", "suboptimal", "random")
    """
    N = 32
    agents = ("bopa",)
    sizes = ("small", "medium", "large")
    teammates = ("greedy", "suboptimal", "random")
    """

    print(f"{domain}", flush=True)
    for size in sizes:
        start = time.time()
        print(f"{size} {domain}", flush=True)
        for teammate in teammates:
            print(f"{size} {domain} {teammate}", flush=True)
            can_skip = True
            for config in range(1, 4):
                for agent_name in agents:
                    yaaf.mkdir(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/")
                    found, needed = fetch_needed_runs(domain, size, teammate, config, agent_name, N)
                    if needed > 0:
                        can_skip = False
                        break
                if not can_skip:
                    break
            if can_skip:
                continue
            tasks = task_factory(size, teammate)
            yaaf.mkdir(f"resources/results/{domain}/{teammate}/{size}")
            with open(f"resources/results/{domain}/{teammate}/{size}/num_states.txt", "w") as file: file.write(str(tasks[0].num_states))
            for t, env in enumerate(tasks):
                config = t + 1
                for agent_name in agents:
                    yaaf.mkdir(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/")
                    found, needed = fetch_needed_runs(domain, size, teammate, config, agent_name, N)
                    print(f"{domain} {size} task {config} with {teammate} teammate -> {agent_name} agent (found {found}, needs {needed})", end="", flush=True)
                    for n in range(needed):
                        metric = TimestepsPerEpisodeMetric()
                        if agent_name == "bopa":
                            analyzer = TaskInferenceAnalyzer()
                            observers = [metric, analyzer]
                        else:
                            observers = [metric]
                        result = run(agent_factory(agent_name, env, tasks), env, observers)
                        np.save(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/{getrandbits(64)}", result)
                        if agent_name == "bopa":
                            yaaf.mkdir(f"resources/beliefs/{domain}/{teammate}/{size}/task_{config}/")
                            np.save(f"resources/beliefs/{domain}/{teammate}/{size}/task_{config}/{getrandbits(64)}", analyzer.result())
                    print(f" -> Done", flush=True)
            print(f"{size} {domain} {teammate} teammate done", flush=True)
        end = time.time()
        total = end - start
        print(f"{size} {domain} done ({total} seconds)\n\n", flush=True)

    print(f"{domain} done!\n\n\n", flush=True)
