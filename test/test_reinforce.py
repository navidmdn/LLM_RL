from fire import Fire
import torch
from envs.reward import RewardModel
from policy.fewshot_llm_stepwise_math_policy import FewShotLLMStepwiseMathPolicy
from policy.fewshot_llm_stepwise_math_config import FewshotLLMStepwiseMathPolicyConfig
from envs.gsm8k import GSM8kEnv
from algorithms.reinforce import Reinforce
from algorithms.reinfoce_config import ReinforceConfig
from envs.gsm8k import StepwiseVerbalRewardModel
from utils.hf import load_model_and_tokenizer

class MockedRewardModel(RewardModel):
    def get_reward(self, current_state, action, next_state):
        return 1


def test_reinforce(gsm8k_path: str, sys_prompt_path: str, model_name_or_path: str, cache_dir: str = None,
                   n_few_shot_examples = 5, qlora = False, sym_calc = False, local_test: bool = False,
                   buffer_cache_path: str = None, reward_model_sys_prompt_path: str = None):

    with open(sys_prompt_path, 'r') as f:
        sys_prompt = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_llm, reward_tokenizer = load_model_and_tokenizer(model_name_or_path, device=device, cache_dir=cache_dir,
                                                            local_test=local_test, quantized=qlora)
    reward_model = StepwiseVerbalRewardModel(model=reward_llm, tokenizer=reward_tokenizer,
                                             system_prompt_path=reward_model_sys_prompt_path,
                                             intermediate_steps_reward_scale_factor=0.2)

    env = GSM8kEnv(data_file=gsm8k_path, reward_model=reward_model, n_few_shot_examples=n_few_shot_examples)

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device=device, cache_dir=cache_dir, local_test=local_test, quantized=qlora)

    policy_config = FewshotLLMStepwiseMathPolicyConfig(
        max_new_tokens=64
    )
    policy = FewShotLLMStepwiseMathPolicy(tokenizer, model, config=policy_config, system_prompt=sys_prompt,
                                          few_shot_examples=env.few_shot_examples, symbol_calculation=sym_calc,
                                          use_lora=True)

    alg_config = ReinforceConfig()
    reinforce = Reinforce(policy, env, reward_model, config=alg_config)

    # # for collection test
    num_iterations = 1
    num_collection_episodes = 300
    max_steps = 10
    num_update_steps = 0
    load_buffer_from_cache_at_start = False
    cache_buffer = True

    # for training test
    # num_iterations = 10
    # num_collection_episodes = 0
    # max_steps = 10
    # num_update_steps = 10
    # load_buffer_from_cache_at_start = True
    # cache_buffer = False

    reinforce.train(
        num_iterations=num_iterations,
        num_collection_episodes=num_collection_episodes,
        max_trajectory_steps=max_steps,
        num_update_steps=num_update_steps,
        load_buffer_from_cache_at_start=load_buffer_from_cache_at_start,
        cache_buffer=cache_buffer,
        buffer_cache_path=buffer_cache_path
    )


if __name__ == '__main__':
    Fire(test_reinforce)



