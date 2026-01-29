import pandas as pd
import matplotlib.pyplot as plt






files = [
    ("results/maddpg_navigation_2026-01-29_13-40-04/data/metrics.csv", ("maddpg")),
    ("results/ibmarl_navigation_2026-01-29_14-09-06/data/metrics.csv", ("ibmarl (strict)"))
]

plt.figure()

for file, label in files:
    df = pd.read_csv(file)

    # Optional: filter by group if needed
    # df = df[df["group"] == "agents"]

    plt.plot(
        df["iteration"],
        df["episode_reward_mean"],
        label=label
    )

plt.xlabel("Iteration")
plt.ylabel("Episode Reward Mean")
plt.legend()
plt.tight_layout()
plt.show()





files = [
        ("results/ibmarl_navigation_2026-01-29_14-09-06/data/metrics.csv", "ibmarl (0.01)"),
]


for file, label in files:
    df = pd.read_csv(file)

    # Optional: filter by group if needed
    # df = df[df["group"] == "agents"]

    plt.plot(
        df["iteration"],
        df["rl_action_fraction"],
        label=label
    )

plt.xlabel("Iteration")
plt.ylabel("Rl_action_fraction")
plt.legend()
plt.tight_layout()
plt.show()
