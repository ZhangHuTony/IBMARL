import pandas as pd
import matplotlib.pyplot as plt

baselines = [
    {"value": -57.55476379394531, "label": "24 demonstration r2bc performance", "color": "orange", "style": "--"},
    # Add more as needed...    
]

# --- CONFIGURATION: Your experiment files ---
files = [
    ("/home/connor/Desktop/Projects/IBMARL/results/s42_maddpg_gt_reward/data/metrics.csv", "maddpg (old hypers)"),
    ("/home/connor/Desktop/Projects/IBMARL/results/s42_maddpg_gt_reward_pt_2/data/metrics.csv", "maddpg (new hypers)")
]

plt.figure(figsize=(10, 6))

# 1. Loop through and plot all reference lines
for base in baselines:
    plt.axhline(
        y=base["value"],
        color=base["color"],
        linestyle=base["style"],
        linewidth=2,
        label=f'{base["label"]} ({base["value"]})',
        alpha=0.7
    )


# 2. Plot the experiment files
for file, label in files:
    try:
        df = pd.read_csv(file)
        
        # Optional: filter by group if needed
        # df = df[df["group"] == "agents"]

        plt.plot(
            df["iteration"],
            df["episode_reward_mean"],
            label=label,
            alpha=0.8
        )
    except FileNotFoundError:
        print(f"Warning: File not found: {file}")

plt.xlabel("Interaction steps (x1000)")
plt.ylabel("Episode Reward Mean")
plt.title("Training Performance vs Baselines")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
