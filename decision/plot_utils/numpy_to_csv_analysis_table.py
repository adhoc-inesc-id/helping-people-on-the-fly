import csv

from decision.numpy_to_bar_plot import load_results

confidence_level = 0.99
colors = {"greedy": "g", "bopa": "b", "random": "orange"}
show = False

domains = ("panic-buttons", "environment-reckon", "garbage-collection")
agents = ("greedy", "bopa", "random")
sizes = ("small", "medium", "large")
teammates = ("greedy", "suboptimal", "random")

with open(f"resources/results.csv", mode='w') as csv_file:

    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "Domain",
        "Teammate Policy",
        "Task Size",
        "Task Configuration",
        "Ad Hoc Agent",
        "Average Steps to Solve",
        "Standard Deviation",
        "N"
    ])

    for domain in domains:

        for teammate in teammates:

            for s, size in enumerate(sizes):

                for config in (1, 2, 3):

                    try:
                        data = load_results(domain, teammate, config, agents, confidence_level)
                    except Exception as e:
                        print(f"Skipping resources/results/{domain}/{teammate}/{size}")
                        continue

                    for i, agent_name in enumerate(agents):
                        mean = data[agent_name]["means"][s]
                        std = data[agent_name]["stds"][s]
                        N = data[agent_name]["N"]
                        writer.writerow([
                            domain.capitalize(),
                            teammate.capitalize(),
                            size.capitalize(),
                            config,
                            agent_name.capitalize(),
                            mean,
                            std,
                            N
                        ])
