import time
from random import getrandbits

import numpy as np
import yaaf
from yaaf.evaluation import TimestepsPerEpisodeMetric

from agents.BOPA import TaskInferenceAnalyzer
from run_full_empirical_evaluation import task_factory, fetch_needed_runs, run, agent_factory

if __name__ == '__main__':

    N = 64
    domains = ("panic-buttons", "environment-reckon", "garbage-collection")
    agents = ("bopa",)
    sizes = ("small", "medium", "large")
    teammates = ("greedy", "suboptimal", "random")

    for domain in domains:

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

                tasks = task_factory(domain, size, teammate)
                yaaf.mkdir(f"resources/results/{domain}/{teammate}/{size}")
                with open(f"resources/results/{domain}/{teammate}/{size}/num_states.txt", "w") as file:
                    file.write(str(tasks[0].num_states))

                for t, env in enumerate(tasks):

                    config = t + 1

                    for agent_name in agents:

                        yaaf.mkdir(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/")

                        found, needed = fetch_needed_runs(domain, size, teammate, config, agent_name, N)

                        print(
                            f"{domain} {size} task {config} with {teammate} teammate -> {agent_name} agent (found {found}, needs {needed})",
                            end="", flush=True)
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
                                beliefs = analyzer.result()
                                yaaf.mkdir(f"resources/beliefs/{domain}/{teammate}/{size}/task_{config}")
                                np.save(f"resources/beliefs/{domain}/{teammate}/{size}/task_{config}/{getrandbits(64)}", beliefs)

                        print(f" -> Done", flush=True)

                print(f"{size} {domain} {teammate} teammate done", flush=True)

            end = time.time()
            total = end - start
            print(f"{size} {domain} done ({total} seconds)\n\n", flush=True)

        print(f"{domain} done!\n\n\n", flush=True)