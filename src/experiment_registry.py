"""
Registry of available experiments
"""

from src.experiments.maddpg import MaddpgExperiment

EXPERIMENT_REGISTRY = {
    "maddpg": MaddpgExperiment,
}