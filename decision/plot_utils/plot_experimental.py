import numpy as np
from yaaf.visualization import LinePlot

def plot_task(number, task_beliefs):
    plot = LinePlot(f"Environment Reckon Task {number}", "Timestep", "Likelihood", task_beliefs.shape[0], ymin=-0.2, ymax=1.2)
    colors = ["r", "g", "b"]
    markers = ["D", "o", "x"]
    for column in range(task_beliefs.shape[1]):
        prog = task_beliefs[:task_beliefs.shape[1], column]
        plot.add_run(f"Task {column + 1}", prog, color=colors[column], marker=markers[column])
    plot.show()
    plot.savefig(f"../../resources/belief-plots/env-reckon-task-{number}.pdf")
    plot.savefig(f"../../resources/belief-plots/env-reckon-task-{number}.png")


if __name__ == '__main__':

    t0 = np.array([0.333, 0.333, 0.333])

    m1_t1 = np.array([9.99250843e-01, 3.74578599e-04, 3.74578599e-04])
    m1_t2 = np.array([9.99425379e-01, 1.00007458e-04, 4.74613995e-04])
    m1_t3 = np.array([9.99311247e-01, 1.14262296e-04, 5.74490894e-04])
    m1 = np.array([t0, m1_t1, m1_t2, m1_t3])
    plot_task(1, m1)

    m2_t1 = np.array([1.66583375e-04, 5.55444500e-01, 4.44388917e-01])
    m2_t2 = np.array([3.59683443e-04, 9.99280633e-01, 3.59683443e-04])
    m2 = np.array([t0, m2_t1, m2_t2])
    plot_task(2, m2)

    m3_t1 = np.array([1.66583375e-04, 5.55444500e-01, 4.44388917e-01])
    m3_t2 = np.array([4.49449396e-04, 4.49449396e-04, 9.99101101e-01])
    m3_t3 = np.array([5.49531524e-04, 1.00014947e-04, 9.99350454e-01])
    m3 = np.array([t0, m3_t1, m3_t2, m3_t3])
    plot_task(3, m3)
