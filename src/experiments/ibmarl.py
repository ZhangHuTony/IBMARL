
from src.experiments.base_marl_experiment import BaseMARLExperiment
from pathlib import Path

from src.r2bc.mabc import DecentralizedMiniBC

class IbmarlExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)


        bc_path = Path(config["r2bc_checkpoint_path"])
        self.il_policy = self._initialize_Il_Policy(bc_path)

    def train(self):
        raise NotImplementedError
    
    def save_checkpoint(self):
        raise NotImplementedError
    
    def render_policy(self):
        raise NotImplementedError
    
    def _initialize_Il_Policy(self, r2bc_path):
        il = DecentralizedMiniBC.load_checkpoint(str(r2bc_path), device = self.device)
        il.eval()

        print("[IBMarl] Loaded R2BC IL policy.")
        print(f"[IBMarl] IL policy n={il.n}, in_size(total)={il.in_size*il.n}, out_size(total)={il.out_size*il.n}")
        return il