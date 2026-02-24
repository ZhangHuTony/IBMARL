"""
Registry of available experiments
"""

from src.experiments.maddpg import MaddpgExperiment
from src.experiments.ibmarl_experiment import IbmarlExperiment
from src.experiments.rlfd_experiment import RlfdExperiment
from src.experiments.rft_experiment import RftExperiment

EXPERIMENT_REGISTRY = {
    "maddpg": MaddpgExperiment,
    "ibmarl": IbmarlExperiment,
    "rlfd": RlfdExperiment,
    "rft": RftExperiment,
}