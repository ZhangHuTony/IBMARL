import pandas as pd
import matplotlib.pyplot as plt

# Paths to your metric files
files = [
    ("results/maddpg/data/metrics.csv", "maddpg"),
   # ("results/maddpg_dropout/data/metrics.csv", "maddpg (dropout)"),
    ("results/ibmarl_navigation_2026-01-21_21-49-41/data/metrics.csv", "ibmarl (greedy)"),
    ("results/ibmarl_navigation_2026-01-21_22-17-29/data/metrics.csv", "ibmarl (soft 3)"),
    ("results/ibmarl_navigation_2026-01-21_22-34-12/data/metrics.csv", "ibmarl (soft 1)")
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


file = "results/ibmarl_navigation_2026-01-21_22-17-29/data/metrics.csv"

df = pd.read_csv(file)

headers = [
    "rl_action_fraction",
]

plt.figure()

for h in headers:
    if h not in df.columns:
        raise KeyError(f"Column '{h}' not found in CSV")
    plt.plot(df["iteration"], df[h], label=h)

plt.xlabel("iteration")
plt.ylabel("Q value")
plt.legend()
plt.show()
