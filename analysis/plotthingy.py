import pandas as pd
import matplotlib.pyplot as plt






baselines = [
    {"value": -57.55476379394531, "label": "24 demonstration r2bc performance", "color": "orange", "style": "--"},
    # Add more as needed...
]

# --- CONFIGURATION: Your experiment files ---
files = [
    ("results/maddpg_navigation_2026-02-01_21-36-48/data/metrics.csv", "maddpg"),
    ("results/ibmarl_navigation_2026-02-01_20-30-50/data/metrics.csv", "ibmarl non-soft"),
    ("results/ibmarl_navigation_2026-02-01_19-21-56/data/metrics.csv", "ibmarl soft ")
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


files = [
   # ("results/maddpg_navigation_2026-01-30_11-21-27/data/metrics.csv", "maddpg"),
    ("results/ibmarl_navigation_2026-02-01_20-30-50/data/metrics.csv", "ibmarl non-soft"),
    ("results/ibmarl_navigation_2026-02-01_19-21-56/data/metrics.csv", "ibmarl soft ")
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

# # --- CONFIGURATION: Your experiment files ---
# files = [
#     ("results/maddpg_navigation_2026-01-29_13-40-04/data/metrics.csv", "maddpg"),
#     ("results/ibmarl_navigation_2026-01-29_17-54-16/data/metrics.csv", "ibmarl strict (1.0 temp)"),
#     ("results/ibmarl_navigation_2026-01-29_18-24-19/data/metrics.csv", "ibmarl strict, (0.1 temp)"),
#     ("results/ibmarl_navigation_2026-01-29_16-11-03/data/metrics.csv", "ibmarl strict, (0.05 temp)"),
#    ("results/ibmarl_navigation_2026-01-29_15-36-19/data/metrics.csv", "ibmarl strict (0.02 temp)"),
#    ("results/maddpg_navigation_2026-01-29_19-35-23/data/metrics.csv", "maddpg")
# ]

# plt.figure(figsize=(10, 6))

# # 1. Loop through and plot all reference lines
# for base in baselines:
#     plt.axhline(
#         y=base["value"],
#         color=base["color"],
#         linestyle=base["style"],
#         linewidth=2,
#         label=f'{base["label"]} ({base["value"]})',
#         alpha=0.7
#     )

# # 2. Plot the experiment files
# for file, label in files:
#     try:
#         df = pd.read_csv(file)
        
#         # Optional: filter by group if needed
#         # df = df[df["group"] == "agents"]

#         plt.plot(
#             df["iteration"],
#             df["episode_reward_mean"],
#             label=label,
#             alpha=0.8
#         )
#     except FileNotFoundError:
#         print(f"Warning: File not found: {file}")

# plt.xlabel("Iteration")
# plt.ylabel("Episode Reward Mean")
# plt.title("Training Performance vs Baselines")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # --- CONFIGURATION: Your experiment files ---
# files = [
#     ("results/maddpg_navigation_2026-01-29_13-40-04/data/metrics.csv", "maddpg"),
#     ("results/ibmarl_navigation_2026-01-29_19-02-06/data/metrics.csv", "ibmarl (temp 0.1)"),
#     ("results/ibmarl_navigation_2026-01-29_18-24-19/data/metrics.csv", "ibmarl strict, (0.1 temp)"),
#     ("results/maddpg_navigation_2026-01-30_11-21-27/data/metrics.csv", "maddpg 2")
# ]

# plt.figure(figsize=(10, 6))

# # 1. Loop through and plot all reference lines
# for base in baselines:
#     plt.axhline(
#         y=base["value"],
#         color=base["color"],
#         linestyle=base["style"],
#         linewidth=2,
#         label=f'{base["label"]} ({base["value"]})',
#         alpha=0.7
#     )

# # 2. Plot the experiment files
# for file, label in files:
#     try:
#         df = pd.read_csv(file)
        
#         # Optional: filter by group if needed
#         # df = df[df["group"] == "agents"]

#         plt.plot(
#             df["iteration"],
#             df["episode_reward_mean"],
#             label=label,
#             alpha=0.8
#         )
#     except FileNotFoundError:
#         print(f"Warning: File not found: {file}")

# plt.xlabel("Iteration")
# plt.ylabel("Episode Reward Mean")
# plt.title("Training Performance vs Baselines")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


# files = [
#                 ("results/ibmarl_navigation_2026-01-29_17-54-16/data/metrics.csv", "ibmarl strict (1.0 temp)"),
#     ("results/ibmarl_navigation_2026-01-29_18-24-19/data/metrics.csv", "ibmarl strict, (0.1 temp)"),
#     ("results/ibmarl_navigation_2026-01-29_16-11-03/data/metrics.csv", "ibmarl strict, (0.05 temp)"),
#    ("results/ibmarl_navigation_2026-01-29_15-36-19/data/metrics.csv", "ibmarl strict (0.02 temp)"),

        
# ]


# for file, label in files:
#     df = pd.read_csv(file)

#     # Optional: filter by group if needed
#     # df = df[df["group"] == "agents"]

#     plt.plot(
#         df["iteration"],
#         df["rl_action_fraction"],
#         label=label
#     )

# plt.xlabel("Iteration")
# plt.ylabel("Rl_action_fraction")
# plt.legend()
# plt.tight_layout()
# plt.show()
