import pandas as pd
import matplotlib.pyplot as plt






files = [
    ("/home/connor/Desktop/Projects/IBMARL/results/s42_maddpg_gt_reward/data/metrics.csv", "maddpg (old hypers)"),
    ("/home/connor/Desktop/Projects/IBMARL/results/s42_maddpg_gt_reward_pt_2/data/metrics.csv", "maddpg (new hypers)")
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
