import numpy as np

import yaaf
from matplotlib.pyplot import close
from yaaf.visualization import LinePlot

RESOURCES_ROOT = "../../resources"

def beliefs_plot(domain, config, teammate, confidence_level, colors, markers, show):

    directory = f"{RESOURCES_ROOT}/beliefs/{domain}/{teammate}/{size}/task_{config}/"
    timesteps = np.inf

    for file in yaaf.files(directory):
        if ".npy" in file:
            belief_set = np.load(f"{directory}/{file}")
            timesteps = min(timesteps, belief_set.shape[0])

    #title = None
    title = f"{domain.capitalize()} Task {config}"
    plot = LinePlot(title, "Timestep", "Likelihood", timesteps, confidence=confidence_level, ymax=1.2, ymin=-0.2)

    for file in yaaf.files(directory):

        if ".npy" in file:

            belief_set = np.load(f"{directory}/{file}")

            for column in range(belief_set.shape[1]):
                prog = belief_set[:timesteps, column]
                plot.add_run(f"Task {column+1}", prog, color=colors[column], marker=markers[column])

    if teammate == "greedy":
        teammate = "optimal"

    if show:
        plot.show()
    else:
        yaaf.mkdir(f"{RESOURCES_ROOT}/belief-plots/{teammate}")
        yaaf.mkdir(f"{RESOURCES_ROOT}/belief-plots/pdf/{teammate}")
        plot.savefig(f"{RESOURCES_ROOT}/belief-plots/{teammate}/{domain}-{teammate}-{size}-task_{config}.png")
        plot.savefig(f"{RESOURCES_ROOT}/belief-plots/pdf/{teammate}/{domain}-{teammate}-{size}-task_{config}.pdf")

    close("all")


if __name__ == '__main__':

    confidence_level = 0.95
    colors = ("r", "g", "b")
    markers = ["D", "o", "x"]
    show = False

    domains = ("panic-buttons", "environment-reckon", "garbage-collection")
    sizes = ("small", "medium", "large")
    teammates = ("greedy", "suboptimal", "random")

    if not show:
        yaaf.rmdir(f"{RESOURCES_ROOT}/belief-plots")
        yaaf.mkdir(f"{RESOURCES_ROOT}/belief-plots")
        yaaf.rmdir(f"{RESOURCES_ROOT}/belief-plots/pdf")
        yaaf.mkdir(f"{RESOURCES_ROOT}/belief-plots/pdf")

    for domain in domains:
        print(f"{domain}", flush=True)
        for size in sizes:
            print(f"{size} {domain}", flush=True)
            for teammate in teammates:
                for config in (1, 2, 3):
                    try:
                        beliefs_plot(domain, config, teammate, confidence_level, colors, markers, show=show)
                        print(f"{size} {domain} {teammate} teammate task #{config} done!", flush=True)
                    except FileNotFoundError as e:
                        print(f"{size} {domain} {teammate} teammate task #{config} missing data, skipping", flush=True)
                print(f"{size} {domain} {teammate} teammate done\n", flush=True)
            print(f"{size} {domain} done\n\n", flush=True)
        print(f"{domain} done!\n\n\n", flush=True)

