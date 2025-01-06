import torch
import wandb
from fire import Fire

from algorithms.reinfoce_distiller_config import ReinforceDistillerConfig
from algorithms.reinforce_distiller import ReinforceDistiller
from envs.gsm8k import GSM8kEnv
from envs.gsm8k import StepwiseVerbalRewardModel, FinalAnswerRewardModel
from policy.fewshot_llm_stepwise_math_config import FewshotLLMStepwiseMathPolicyConfig
from policy.fewshot_llm_stepwise_math_policy import FewShotLLMStepwiseMathPolicy
from utils.hf import load_model_and_tokenizer


def run(gsm8k_path: str, sys_prompt_path: str, model_name_or_path: str, ref_model_name: str, cache_dir: str = None,
        n_few_shot_examples = 5, qlora = False, sym_calc = False, local_test: bool = False,
        buffer_cache_path: str = None, reward_model_sys_prompt_path: str = None,
        num_iterations: int = 10, num_collection_episodes: int = 1, max_trajectory_steps: int = 10,
        num_update_steps: int = 10, load_buffer_from_cache_at_start: bool = False,
        cache_buffer: bool = False, test_gsm8k_path: str = None, gradient_accumulation_steps: int = 16,
        save_path: str = "model.pt", evaluation_at_start: bool = False, evaluation_iterations: int = 10,
        n_save_iterations: int = None, n_eval_steps: int = 20, use_lora: bool = False, nll_coeff: float = 0.25,
        buffer_size: int = 100000, stepwise_eval: bool = False, update_ref_iterations: int = 100):


    with open(sys_prompt_path, 'r') as f:
        sys_prompt = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    reward_model = FinalAnswerRewardModel()
    env = GSM8kEnv(data_file=gsm8k_path, reward_model=reward_model, n_few_shot_examples=n_few_shot_examples)

    evaluation_env = None
    if test_gsm8k_path is not None:
        evaluation_env = GSM8kEnv(data_file=test_gsm8k_path, reward_model=reward_model,
                                  n_few_shot_examples=n_few_shot_examples, train_data_file=gsm8k_path,
                                  deterministic=True)


    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device=device, cache_dir=cache_dir,
                                                local_test=local_test, quantized=qlora)

    policy_config = FewshotLLMStepwiseMathPolicyConfig(
        max_new_tokens=64,
        temperature=0.8,
        learning_rate=1e-5,
    )

    policy = FewShotLLMStepwiseMathPolicy(tokenizer, model, config=policy_config, system_prompt=sys_prompt,
                                          few_shot_examples=env.few_shot_examples, symbol_calculation=sym_calc,
                                          use_lora=use_lora)

    ref_policy_config = FewshotLLMStepwiseMathPolicyConfig()



    ref_model, _ = load_model_and_tokenizer(ref_model_name, device=device, cache_dir=cache_dir,
                                            local_test=local_test, quantized=qlora)

    ref_policy = FewShotLLMStepwiseMathPolicy(tokenizer, ref_model, config=ref_policy_config, system_prompt=sys_prompt,
                                          few_shot_examples=env.few_shot_examples, symbol_calculation=False,
                                          use_lora=False)

    alg_config = ReinforceDistillerConfig(
        buffer_size=buffer_size,
        nll_coeff=nll_coeff,
    )

    reinforce = ReinforceDistiller(policy, ref_policy, env, reward_model, config=alg_config, evaluation_env=evaluation_env)


    reinforce.train(
        num_iterations=num_iterations,
        num_collection_episodes=num_collection_episodes,
        max_trajectory_steps=max_trajectory_steps,
        num_update_steps=num_update_steps,
        load_buffer_from_cache_at_start=load_buffer_from_cache_at_start,
        cache_buffer=cache_buffer,
        buffer_cache_path=buffer_cache_path,
        evaluate_at_start=evaluation_at_start,
        evaluation_iterations=evaluation_iterations,
        n_save_iterations=n_save_iterations,
        n_eval_steps=n_eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_path=save_path,
        stepwise_eval=stepwise_eval,
        update_ref_iterations=update_ref_iterations,
    )


if __name__ == '__main__':
    Fire(run)



