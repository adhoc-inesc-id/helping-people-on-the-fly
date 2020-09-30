import csv
import logging
import shutil

import numpy as np
import yaaf

domains = ("panic-buttons", "environment-reckon", "garbage-collection")
agents = ("greedy", "bopa", "random")
sizes = ("small", "medium", "large")
teammates = ("greedy", "suboptimal", "random")

yaaf.rmdir("resources/csv")
yaaf.mkdir(f"resources/csv")

for domain in domains:

    for teammate in teammates:

        for config in (1, 2, 3):

            with open(f"resources/csv/{domain}-{teammate}-teammate-task-{config}.csv", mode='w') as csv_file:

                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Run ID (.npy file)", "Number of States", "Agent", "Steps to Solve"])

                for agent in ("greedy", "bopa", "random"):

                    for size in sizes:

                        with open(f"resources/results/{domain}/{teammate}/{size}/num_states.txt", "r") as file:
                            lines = file.readlines()
                            num_states = int(lines[0].replace("\n", ""))

                        agent_files = yaaf.files(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent}")
                        result_files = [file for file in agent_files if ".npy" in file]

                        for file in result_files:
                            result = np.load(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent}/{file}")
                            if result.shape == (0,):
                                print(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent}/{file}")
                                logging.warning(f"Corrupt file {domain}/{teammate}/{size}/task_{config}/{agent}/{file}, moving to badfiles.")
                                yaaf.mkdir(f"resources/badfiles/{domain}/{teammate}/{size}/task_{config}/{agent}")
                                shutil.move(
                                    f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent}/{file}",
                                    f"resources/badfiles/{domain}/{teammate}/{size}/task_{config}/{agent}/{file}")
                            else:
                                try: steps = result[0]
                                except Exception as e: steps = result
                                writer.writerow([f'{file.split(".")[0]}', num_states, f"{agent}", steps])
