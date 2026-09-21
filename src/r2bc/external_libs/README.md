# R2BC learned-supervisor dependencies

MAPPO and IPPO checkpoint loading uses the same two source dependencies as the
companion R2BC repository. They are intentionally optional: heuristic R2BC
demonstrators do not import them.

From this directory, populate the dependencies before loading a MAPPO or IPPO
checkpoint:

```bash
cd src/r2bc/external_libs
git clone https://github.com/proroklab/HetGPPO.git HetGPPO
git -C HetGPPO checkout 660dc25e46cdcdf217b2d7a192d0b3098940c806
git clone https://github.com/proroklab/rllib_differentiable_comms.git rllib_differentiable_comms
git -C rllib_differentiable_comms checkout f6d4e6085f9d73bbab3c3b785ac75ad011f390a7
```

These are the exact revisions referenced by R2BC. The directories are ignored
by this repository so users can initialize them locally without adding their
copies to IBMARL commits.
