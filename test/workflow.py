import numpy as np
from fire import Fire
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from typing import List, Dict, Tuple
import re
from envs.gsm8k import GSM8KExample
from policy.fewshot_llm_stepwise_math_policy import FewShotLLMStepwiseMathPolicy
from policy.fewshot_llm_stepwise_math_config import FewshotLLMStepwiseMathPolicyConfig
from envs.gsm8k import GSM8kEnv


class MockedRewardModel:
    def get_reward(self, current_state, action, next_state):
        return 1


def test_gsm8k_policy(gsm8k_path: str, sys_prompt_path: str, model_name_or_path: str, cache_dir: str,
                      n_few_shot_examples = 5, qlora = False, sym_calc = False):

    with open(sys_prompt_path, 'r') as f:
        sys_prompt = f.read().strip()

    reward_model = MockedRewardModel()
    env = GSM8kEnv(data_file=gsm8k_path, reward_model=reward_model, n_few_shot_examples=n_few_shot_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if qlora:
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = True
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
            cache_dir=cache_dir
        )
    else:
        # todo: set correct data type for the model you're testing
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    config = FewshotLLMStepwiseMathPolicyConfig(
        max_new_tokens=64
    )

    policy = FewShotLLMStepwiseMathPolicy(tokenizer, model, config=config, system_prompt=sys_prompt,
                                          few_shot_examples=env.few_shot_examples)
    concluded = False
    i = 0

    state = env.reset()
    while not concluded:

        action = policy.get_action(state, symbolic_calculation=sym_calc)
        print(f"Action {i}: {action}")

        state, r, concluded, _ = env.step(action)
        i += 1

    print("Concluded:")
    print(env.current_state)



if __name__ == '__main__':
    Fire(test_gsm8k_policy)



