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
    ("/home/tony-zhang/Research/IBMARL/results/ibmarl_navigation_2026-01-27_18-55-41/data/metrics.csv", "ibmarl strict(0.01)"),
    ("/home/tony-zhang/Research/IBMARL/results/ibmarl_navigation_2026-01-28_10-34-21/data/metrics.csv", "ibmarl strict(0.1)"),
    ("/home/tony-zhang/Research/IBMARL/results/ibmarl_navigation_2026-01-28_10-34-21/data/metrics.csv", "ibmarl (0.1)"),
    ("/home/tony-zhang/Research/IBMARL/results/maddpg_navigation_2026-01-28_11-06-35/data/metrics.csv", "maddpg")
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
        ("/home/tony-zhang/Research/IBMARL/results/ibmarl_navigation_2026-01-27_18-55-41/data/metrics.csv", "ibmarl (0.01)"),
        ("/home/tony-zhang/Research/IBMARL/results/ibmarl_navigation_2026-01-28_10-34-21/data/metrics.csv", "ibmarl (0.1)"),
            ("/home/tony-zhang/Research/IBMARL/results/ibmarl_navigation_2026-01-28_10-34-21/data/metrics.csv", "ibmarl (0.1)"),
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
