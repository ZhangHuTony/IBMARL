import pandas as pd
import matplotlib.pyplot as plt





# files = [
#     ("results/randomm/data/metrics.csv", "random"),
#     ("results/il_terrible_8/data/metrics.csv", "terrible (8)"),
#     ("results/il_ok_24/data/metrics.csv", "ok (24)"),
#     ("results/il_good_36/data/metrics.csv", "good (36)"),
#     ("results/il_perfect_240/data/metrics.csv", "perfect (240)")
# ]

files = [
    ("results/maddpg_navigation_2026-01-24_17-56-17/data/metrics.csv", "maddpg"),
    ("results/ibmarl_navigation_2026-01-24_18-33-38/data/metrics.csv", "ibmarl (soft temp 1)"),
    ("results/ibmarl_navigation_2026-01-24_19-52-37/data/metrics.csv", "ibmarl (soft temp 0.1)"),
    ("results/ibmarl_navigation_2026-01-24_20-38-11/data/metrics.csv", "ibmarl (soft temp 0.001)"),
    ("results/ibmarl_navigation_2026-01-24_19-01-22/data/metrics.csv", "ibmarl (greedy)")
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
    ("results/ibmarl_navigation_2026-01-24_18-33-38/data/metrics.csv", "ibmarl (soft temp 1)"),
    ("results/ibmarl_navigation_2026-01-24_19-52-37/data/metrics.csv", "ibmarl (soft temp 0.1)"),
    ("results/ibmarl_navigation_2026-01-24_20-38-11/data/metrics.csv", "ibmarl (soft temp 0.001)"),
    ("results/ibmarl_navigation_2026-01-24_19-01-22/data/metrics.csv", "ibmarl (greedy)")
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
plt.ylabel("Episode Reward Mean")
plt.legend()
plt.tight_layout()
plt.show()
