"""
Registry of available experiments
"""

from src.experiments.maddpg import MaddpgExperiment
from src.experiments.ibmarl import IbmarlExperiment

EXPERIMENT_REGISTRY = {
    "maddpg": MaddpgExperiment,
    "ibmarl": IbmarlExperiment,

}