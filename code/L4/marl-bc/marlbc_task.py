"""
BenchMARL Task wrappers for RBC simulators.

Exposes RBCParallelEnv as BenchMARL Task enum members so that the existing
Experiment / Benchmark pipeline can be used unchanged — just swap the task:

    task = RBCTask.KL.get_task(config={
        "n_agents":  2,
        "obs_space": ["wealth", "income"],
        "num_iters": 500,
    })

Variants
--------
KL  — RBCSimulator_KL  (homogeneous agents, endogenous labour)
KS  — RBCSimulator_KS  (Krusell-Smith, aggregate + idiosyncratic shocks)
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from marlbc_env import RBCParallelEnv
from marlbc_simulators import RBCSimulator_KL, RBCSimulator_KS


# ------------------------------------------------------------------ #
# TaskClass — concrete implementation
# ------------------------------------------------------------------ #

class RBCTaskClass(TaskClass):
    """
    TaskClass implementation for RBC environments.

    Config keys
    -----------
    n_agents  : int         — number of household agents
    delta     : float        — capital depreciation rate
    obs_space : list[str]   — observation variables (see RBCParallelEnv)
    num_iters : int         — episode length (steps before truncation)
    """

    # ---- environment factory ---------------------------------------- #

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        cfg = copy.deepcopy(self.config)
        task_name = self.name  # "KL" or "KS"

        def _make() -> EnvBase:
            # 1. build the economic simulator
            n = cfg["n_agents"]
            delta = cfg.get("delta", 0.025)  
            if task_name == "KL":
                sim = RBCSimulator_KL(n_agents=n, delta=delta)
            elif task_name == "KS":
                sim = RBCSimulator_KS(n_agents=n, delta=delta)
            else:
                raise ValueError(f"Unknown RBCTask variant: {task_name}")

            # 2. wrap in PettingZoo parallel env
            pz_env = RBCParallelEnv(
                simulator=sim,
                obs_space=cfg["obs_space"],
                num_iters=cfg["num_iters"],
            )

            # 3. wrap with TorchRL — accepts an already-instantiated PettingZoo env
            return PettingZooWrapper(
                env=pz_env,
                return_state=False,
                group_map=None,          # auto-detect: all agents in one group
                use_mask=False,
                categorical_actions=False,  # continuous actions
                seed=seed,
                device=device,
            )

        return _make

    # ---- action / observation support ------------------------------- #

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    # ---- episode bookkeeping ---------------------------------------- #

    def max_steps(self, env: EnvBase) -> int:
        return self.config["num_iters"]

    def has_render(self, env: EnvBase) -> bool:
        return False

    # ---- agent grouping --------------------------------------------- #

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # PettingZooWrapper populates env.group_map automatically
        return env.group_map

    # ---- specs ------------------------------------------------------ #

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "rbc"


# ------------------------------------------------------------------ #
# Task enum — one member per simulator variant
# ------------------------------------------------------------------ #

class RBCTask(Task):
    """BenchMARL Task enum for RBC environments.

    Usage::

        task = RBCTask.KL.get_task(config={
            "n_agents":  2,
            "obs_space": ["wealth", "income"],
            "num_iters": 500,
        })
    """

    KL = None   # RBCSimulator_KL  — homogeneous, endogenous labour
    KS = None   # RBCSimulator_KS  — Krusell-Smith, idiosyncratic shocks

    @staticmethod
    def associated_class():
        return RBCTaskClass
