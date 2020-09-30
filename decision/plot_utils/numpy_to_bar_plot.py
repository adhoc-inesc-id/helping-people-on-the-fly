import logging
import shutil

import matplotlib.pyplot as plt
import numpy as np
import yaaf
from yaaf.visualization import confidence_interval


# ######### #
# Utilities #
# ######### #

def sizes_bar_plot(domain, config, teammate, agent_names, alias, data, colors, width=0.25, show=False):

    X = np.arange(3)

    for i, agent_name in enumerate(agent_names):
        means_per_size = data[agent_name]["means"]
        error = data[agent_name]["error"]
        color = colors[agent_name]
        N = data[agent_name]["N"]
        plt.bar(X + i * width, means_per_size, yerr=error, color=color, width = width, label=f"{alias[i]} (N={N})", ecolor='black', capsize=10)

    plt.legend(loc="upper left")
    plt.xticks([width, 1 + width, 2 + width], [data["num_states"]["small"], data["num_states"]["medium"], data["num_states"]["large"]])
    plt.ylabel("Average Steps to Solve")
    plt.xlabel("Task Complexity")
    plt.ylim(0)
    plt.tight_layout()

    if not show:
        yaaf.mkdir("../resources/plots")
        yaaf.mkdir("../resources/plots/pdf")
        plt.savefig(f"resources/plots/{domain}-{teammate}-task{config}.png")
        plt.savefig(f"resources/plots/pdf/{domain}-{teammate}-task{config}.png")
    else:
        plt.title(f"{domain} (Config #{config})\n({teammate} Teammate)")
        plt.show()

    plt.close()

def load_results(domain, teammate, config, agent_names, confidence_level):

    data = {}
    data["num_states"] = {}

    for size in ("small", "medium", "large"):
        with open(f"resources/results/{domain}/{teammate}/{size}/num_states.txt", "r") as file:
            lines = file.readlines()
            num_states = int(lines[0].replace("\n", ""))
            data["num_states"][size] = f"{size.capitalize()}\n({num_states} states)"

    for agent_name in agent_names:

        data[agent_name] = {}
        data[agent_name]["means"] = []
        data[agent_name]["stds"] = []
        data[agent_name]["error"] = []

        for size in ("small", "medium", "large"):

            files = yaaf.files(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}")
            result_files = [file for file in files if ".npy" in file]
            results = []
            for file in result_files:
                result = np.load(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/{file}")
                if result.shape == (0,):
                    print(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/{file}")
                    logging.warning(f"Corrupt file {domain}/{teammate}/{size}/task_{config}/{agent_name}/{file}, moving to badfiles.")
                    yaaf.mkdir(f"resources/badfiles/{domain}/{teammate}/{size}/task_{config}/{agent_name}")
                    shutil.move(f"resources/results/{domain}/{teammate}/{size}/task_{config}/{agent_name}/{file}", f"resources/badfiles/{domain}/{teammate}/{size}/task_{config}/{agent_name}/{file}")
                else:
                    try:
                        results.append(result[0])
                    except Exception as e:
                        results.append(result)

            results = np.array(results)

            if len(result_files) != results.shape[0]:
                logging.warning(f"Bad files were found, total N is not going to match. Proceeding without N={len(result_files)}")

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
    show = False

    domains = ("panic-buttons", "environment-reckon", "garbage-collection")
    agents = ("greedy", "bopa", "random")
    agent_alias = ("Optimal Policy", "BOPA", "Random Policy")

    sizes = ("small", "medium", "large")
    teammates = ("greedy", "suboptimal", "random")

    if not show:
        yaaf.rmdir("../resources/plots")
        yaaf.mkdir("../resources/plots")

    for domain in domains:
        print(f"{domain}", flush=True)
        for size in sizes:
            print(f"{size} {domain}", flush=True)
            for teammate in teammates:
                for config in (1, 2, 3):
                    try:
                        sizes_bar_plot(domain, config, teammate, agents, agent_alias, load_results(domain, teammate, config, agents, confidence_level), colors, show=show)
                        print(f"{size} {domain} {teammate} teammate task #{config} done!", flush=True)
                    except FileNotFoundError as e:
                        print(e)
                        print(f"{size} {domain} {teammate} teammate task #{config} missing data, skipping", flush=True)

                print(f"{size} {domain} {teammate} teammate done\n", flush=True)
            print(f"{size} {domain} done\n\n", flush=True)
        print(f"{domain} done!\n\n\n", flush=True)
