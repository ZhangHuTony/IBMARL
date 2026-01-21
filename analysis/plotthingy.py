import pandas as pd
import matplotlib.pyplot as plt

# Paths to your metric files
files = [
    ("results/maddpg_navigation_2026-01-16_13-37-17/data/metrics.csv", "maddpg"),
    ("results/ibmarl_navigation_2026-01-20_19-34-52/data/metrics.csv", "IBMARL (no actor or bootstrap)"),
    ("results/ibmarl_navigation_2026-01-20_19-23-28/data/metrics.csv", "IBMARL (only bootstrap)"),
    ("results/ibmarl_navigation_2026-01-20_21-35-41/data/metrics.csv", "IBMARL (-2 reward)")
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


file = "results/ibmarl_navigation_2026-01-16_14-49-32/data/metrics.csv"

df = pd.read_csv(file)

headers = [
    "episode_reward_mean",
    "il_action_frac",
    "q_exec_mean",
    "q_rl_policy_mean",
    "q_il_mean",
    "q_rl_minus_il_mean"
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
