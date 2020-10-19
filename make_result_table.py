import csv

import numpy as np
import matplotlib.pyplot as plt

import yaaf
from yaaf.visualization import standard_error


if __name__ == '__main__':

    # Parameters
    resources_root = "resources"
    sizes = ("small", "medium", "large")
    size_alias = ("3x3", "4x4", "5x5")
    agents = ("greedy", "bopa", "random")
    teammates = ("greedy", "suboptimal", "random")

    # Load data
    values = {}
    for size in sizes:
        values[size] = {}
        for teammate in teammates:
            values[size][teammate] = {}
            for agent in agents:
                values[size][teammate][agent] = []
                for task in range(3):
                    root = f"{resources_root}/results/panic-buttons/{teammate}/{size}/task_{task + 1}/{agent}"
                    for file in yaaf.files(root):
                        if ".npy" in file:
                            result = np.load(f"{root}/{file}")
                            values[size][teammate][agent].append(result)


    # Table
    with open(f"result-table.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["", "Optimal", "SubOptimal", "Random"])
        for s, size in enumerate(sizes):
            for agent in agents:
                row = [f"{size_alias[s]} {agent.capitalize()}"]
                for teammate in teammates:
                    result = np.array(values[size][teammate][agent])
                    mean = round(result.mean(), 2)
                    std = round(result.std(), 2)
                    row.append(f"{mean} (±{std})")
                writer.writerow(row)
