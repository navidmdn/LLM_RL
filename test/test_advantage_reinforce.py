import numpy as np
from fire import Fire
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from typing import List, Dict, Tuple
import re
from envs.gsm8k import GSM8KExample
from envs.reward import RewardModel
from policy.fewshot_llm_stepwise_math_policy import FewShotLLMStepwiseMathPolicy
from policy.fewshot_llm_stepwise_math_config import FewshotLLMStepwiseMathPolicyConfig
from envs.gsm8k import GSM8kEnv
from algorithms.advantage_reinforce import AdvantageReinforce
from algorithms.advantage_reinfoce_config import AdvantageReinforceConfig
from utils.hf import load_test_llama3_model


class MockedRewardModel(RewardModel):
    def get_reward(self, current_state, action, next_state):
        return 1


def load_model(model_name_or_path: str, device, cache_dir: str = None, local_test: bool = False, qlora: bool = False):
    if not local_test:
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
    else:
        model = load_test_llama3_model(model_name_or_path)
        model = model.to(device)

    return model


def test_reinforce(gsm8k_path: str, sys_prompt_path: str, model_name_or_path: str, cache_dir: str = None,
                   n_few_shot_examples = 5, qlora = False, sym_calc = False, local_test: bool = False):

    with open(sys_prompt_path, 'r') as f:
        sys_prompt = f.read().strip()

    reward_model = "final_answer"
    env = GSM8kEnv(data_file=gsm8k_path, reward_model=reward_model, n_few_shot_examples=n_few_shot_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_policy_model = load_model(model_name_or_path, device, cache_dir=cache_dir, local_test=local_test, qlora=qlora)
    ref_policy_model = load_model(model_name_or_path, device, cache_dir=cache_dir, local_test=local_test, qlora=qlora)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    base_policy_config = FewshotLLMStepwiseMathPolicyConfig(
        max_new_tokens=64,
        temperature=1.1

    )

    ref_policy_config = FewshotLLMStepwiseMathPolicyConfig(
        max_new_tokens=64,
        temperature=0.1
    )


    base_policy = FewShotLLMStepwiseMathPolicy(tokenizer, base_policy_model, config=base_policy_config, system_prompt=sys_prompt,
                                          few_shot_examples=env.few_shot_examples, symbol_calculation=sym_calc,
                                          use_lora=True)

    ref_policy = FewShotLLMStepwiseMathPolicy(tokenizer, ref_policy_model, config=ref_policy_config, system_prompt=sys_prompt,
                                         few_shot_examples=env.few_shot_examples, symbol_calculation=sym_calc,
                                         use_lora=False)

    alg_config = AdvantageReinforceConfig()
    adv_reinforce = AdvantageReinforce(base_policy, ref_policy, env, reward_model, config=alg_config)

    adv_reinforce.train()


if __name__ == '__main__':
    Fire(test_reinforce)



