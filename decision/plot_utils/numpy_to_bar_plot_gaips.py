import logging
import shutil

import matplotlib.pyplot as plt
import numpy as np
import yaaf
from yaaf.visualization import confidence_interval

RESOURCES_ROOT = f"../../resources"

# ######### #
# Utilities #
# ######### #

def sizes_bar_plot(task_no, agent_names, alias, colors, width=0.25, confidence_level=0.95, show=False):

    X = np.arange(3)
    data = load_results(task_no, agent_names, confidence_level)

    for i, agent_name in enumerate(agent_names):
        mean_per_teammate = data[agent_name]["means"]
        error = data[agent_name]["error"]
        color = colors[agent_name]
        N = data[agent_name]["N"]
        plt.bar(X + i * width, mean_per_teammate, yerr=error, color=color, width = width, label=f"{alias[i]} (N={N})", ecolor='black', capsize=10)

    plt.legend(loc="upper left")
    plt.xticks([width, 1 + width, 2 + width], ["Greedy", "Sub-Optimal", "Random"])
    plt.ylabel("Average Steps to Solve")
    plt.xlabel("Teammate Type")
    plt.ylim(0)
    plt.tight_layout()

    if not show:
        yaaf.mkdir(f"{RESOURCES_ROOT}/plots")
        yaaf.mkdir(f"{RESOURCES_ROOT}/plots/pdf")
        plt.savefig(f"{RESOURCES_ROOT}/plots/gaips_{task_no}.png")
        plt.savefig(f"{RESOURCES_ROOT}/plots/pdf/gaips_{task_no}.pdf")
    else:
        plt.title(f"GAIPS Task Nº {task_no}")
        plt.show()

    plt.close()

def load_results(task_no, agent_names, confidence_level):

    domain = "gaips"
    data = {}

    for agent_name in agent_names:

        data[agent_name] = {}
        data[agent_name]["means"] = []
        data[agent_name]["stds"] = []
        data[agent_name]["error"] = []

        for teammate in ("greedy", "random", "suboptimal"):

            files = yaaf.files(f"{RESOURCES_ROOT}/results/{domain}/{teammate}/small/task_{task_no}/{agent_name}")
            result_files = [file for file in files if ".npy" in file]
            results = []
            for file in result_files:
                result = np.load(f"{RESOURCES_ROOT}/results/{domain}/{teammate}/small/task_{task_no}/{agent_name}/{file}")
                if result.shape == (0,):
                    print(f"{RESOURCES_ROOT}/results/{domain}/{teammate}/small/task_{task_no}/{agent_name}/{file}")
                    logging.warning(f"Corrupt file {domain}/{teammate}/small/task_small/{agent_name}/{file}, moving to badfiles.")
                    yaaf.mkdir(f"{RESOURCES_ROOT}/badfiles/{domain}/{teammate}/small/task_{task_no}/{agent_name}")
                    shutil.move(
                        f"{RESOURCES_ROOT}/results/{domain}/{teammate}/small/task_{task_no}/{agent_name}/{file}",
                        f"{RESOURCES_ROOT}/badfiles/{domain}/{teammate}/small/task_{task_no}/{agent_name}/{file}"
                    )
                else:
                    try:
                        results.append(result[0])
                    except Exception as e:
                        results.append(result)

            results = np.array(results)

            if len(result_files) != results.shape[0]:
                logging.warning(
                    f"Bad files were found, total N is not going to match. Proceeding without N={len(result_files)}")

            N = results.shape[0]
            mean = results.mean() if N > 0 else -1
            std = results.std() if N > 0 else -1
            error = confidence_interval(mean, N, confidence_level) if N > 0 else 0
            data[agent_name]["means"].append(mean)
            data[agent_name]["stds"].append(std)

            if "N" in data[agent_name]:
                old_N = data[agent_name]["N"]
                data[agent_name]["N"] = min(N, old_N)
            else:
                data[agent_name]["N"] = N

            data[agent_name]["error"].append(error)

    return data

# #### #
# Main #
# #### #

if __name__ == '__main__':

    confidence_level = 0.95
    colors = {"greedy": "g", "bopa": "b", "random": "orange"}
    show = True

    agents = ("greedy", "bopa", "random")
    agent_alias = ("Optimal Policy", "BOPA", "Random Policy")

    teammates = ("greedy", "suboptimal", "random")

    for task_no in range(3):

        try:
            sizes_bar_plot(task_no+1, agents, agent_alias, colors, confidence_level=confidence_level, show=show)
        except FileNotFoundError as e:
            print(e)
