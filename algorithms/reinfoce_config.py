


from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ReinforceConfig:
    buffer_size: int = 1000
    gamma: float = 0.99
    update_batch_size: int = 1
    max_grad_norm: float = 1.0
    logger: Optional[str] = "wandb"
    log_steps: int = 8
