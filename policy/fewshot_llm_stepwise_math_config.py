
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FewshotLLMStepwiseMathPolicyConfig:

    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.9
    learning_rate: float = 5e-5
    evaluation_temperature: float = 0.6
    evaluation_top_p: float = 0.9
    evaluation_max_new_tokens: int = 512