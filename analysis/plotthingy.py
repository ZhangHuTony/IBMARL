import pandas as pd
import matplotlib.pyplot as plt





files = [
    ("results/randomm/data/metrics.csv", "random"),
    ("results/il_terrible_8/data/metrics.csv", "terrible (8)"),
    ("results/il_ok_24/data/metrics.csv", "ok (24)"),
    ("results/il_good_36/data/metrics.csv", "good (36)"),
    ("results/il_perfect_240/data/metrics.csv", "perfect (240)")
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



# Paths to your metric files
files = [
    ("results/maddpg/data/metrics.csv", "maddpg"),
   # ("results/maddpg_dropout/data/metrics.csv", "maddpg (dropout)"),
    #("results/ibmarl_navigation_2026-01-21_21-49-41/data/metrics.csv", "ibmarl, ok (greedy)"),
    #("results/ibmarl_navigation_2026-01-21_22-17-29/data/metrics.csv", "ibmarl, ok (soft 3)"),
    ("results/ibmarl_navigation_2026-01-22_13-15-19/data/metrics.csv", "ibmarl, good (soft)"),
    ("results/ibmarl_navigation_2026-01-22_13-29-10/data/metrics.csv", "ibmarl, good (greedy)"),
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
   # ("results/maddpg_dropout/data/metrics.csv", "maddpg (dropout)"),
    ("results/ibmarl_navigation_2026-01-21_21-49-41/data/metrics.csv", "ibmarl, ok (greedy)"),
    ("results/ibmarl_navigation_2026-01-21_22-17-29/data/metrics.csv", "ibmarl, ok (soft 3)"),
    ("results/ibmarl_navigation_2026-01-22_13-15-19/data/metrics.csv", "ibmarl, good (soft)"),
    ("results/ibmarl_navigation_2026-01-22_13-29-10/data/metrics.csv", "ibmarl, good (greedy)"),
    ("results/ibmarl_navigation_2026-01-22_13-44-11/data/metrics.csv", "ibmarl, perfect (greedy)")
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
