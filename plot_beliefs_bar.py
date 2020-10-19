import numpy as np
import matplotlib.pyplot as plt

import yaaf
from yaaf.visualization import standard_error


if __name__ == '__main__':

    # Parameters
    resources_root = "resources"
    sizes = ("small", "medium", "large")
    teammates = ("greedy", "suboptimal", "random")
    last_timestep_only = False

    # Plot Stuff
    show_only = True
    confidence_level = 0.95
    bar_width = 0.25
    colors = ("maroon", "forestgreen", "navy")
    alias = ("Optimal Policy", "Sub Optimal Policy", "Random Policy")

    # Load data
    values = {}
    for size in sizes:
        values[size] = {}
        for teammate in teammates:
            values[size][teammate] = []
            for task in range(3):
                root = f"{resources_root}/beliefs/panic-buttons/{teammate}/{size}/task_{task + 1}"
                for file in yaaf.files(root):
                    if ".npy" in file:
                        beliefs = np.load(f"{root}/{file}")
                        num_steps = beliefs.shape[0]
                        beliefs_last_step = beliefs[-1]
                        correct_task_belief_in_last_step = beliefs_last_step[task]
                        if last_timestep_only:
                            values[size][teammate].append(correct_task_belief_in_last_step)
                        else:
                            for belief in beliefs:
                                values[size][teammate].append(belief[task])

    # Plot

    X = np.arange(3)
    for t, teammate in enumerate(teammates):
        means_per_teammate = []
        errors_per_teammate = []
        for size in sizes:
            result = np.array(values[size][teammate])
            mean = result.mean()
            std = result.std()
            n = result.shape[0]
            err = standard_error(std, n, confidence_level)
            means_per_teammate.append(mean)
            errors_per_teammate.append(err)
        plt.bar(X + t * bar_width, means_per_teammate, yerr=errors_per_teammate, width=bar_width, ecolor='black', capsize=10, color=colors[t], label=alias[t])
    plt.legend(loc="upper right")
    plt.xticks([bar_width, 1 + bar_width, 2 + bar_width], ["3x3", "4x4", "5x5"])
    plt.ylabel("Likelihood of Correct Task")
    plt.xlabel("World Size")
    plt.ylim(0, 1.2)
    plt.tight_layout()

    if show_only:
        plt.show()
    else:
        name = "panic-world-bar-plot.pdf" if last_timestep_only else "panic-world-bar-plot-all-timesteps.pdf"
        plt.savefig(name)