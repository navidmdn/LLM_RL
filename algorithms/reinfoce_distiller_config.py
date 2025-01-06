


from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ReinforceDistillerConfig:
    buffer_size: int = 10000
    gamma: float = 0.99
    nll_coeff: float = 0.5
    update_batch_size: int = 1
    max_grad_norm: float = 1.0
    logger: Optional[str] = "wandb"
    log_steps: int = 8
    evaluation_batch_size: int = 10
