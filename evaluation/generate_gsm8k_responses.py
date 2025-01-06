import torch
from fire import Fire
from envs.gsm8k import GSM8kEnv
from envs.gsm8k import StepwiseVerbalRewardModel, FinalAnswerRewardModel
from policy.fewshot_llm_stepwise_math_config import FewshotLLMStepwiseMathPolicyConfig
from policy.fewshot_llm_stepwise_math_policy import FewShotLLMStepwiseMathPolicy
from utils.hf import load_model_and_tokenizer
from utils.misc import save_jsonl, save_json
from tqdm import tqdm


def run(gsm8k_path: str, sys_prompt_path: str, model_name_or_path: str, cache_dir: str = None,
                   n_few_shot_examples = 5, qlora = False, sym_calc = False, local_test: bool = False,
                   max_steps: int = 10, output_path: str = "results.json"):


    with open(sys_prompt_path, 'r') as f:
        sys_prompt = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_model = FinalAnswerRewardModel()
    env = GSM8kEnv(data_file=gsm8k_path, reward_model=reward_model, n_few_shot_examples=n_few_shot_examples,
                   deterministic=True)

    policy_config = FewshotLLMStepwiseMathPolicyConfig(
        max_new_tokens=64,
        temperature=1.0,
    )
    policy = FewShotLLMStepwiseMathPolicy(config=policy_config, system_prompt=sys_prompt,
                                          few_shot_examples=env.few_shot_examples, symbol_calculation=sym_calc,
                                          use_lora=False, pretrained_model_path=model_name_or_path, inference_mode=True,
                                          device=device, cache_dir=cache_dir)

    # iteratively go over environment and generate responses
    results = []
    for iteration in tqdm(range(len(env.data))):
        rewards = []
        states, actions, returns = [], [], []

        state = env.reset()
        print("-" * 100)
        print("working on state: ", state)
        assert state.generated_solution == "", "restarted environment with non-empty generated solution"

        for step in range(max_steps):
            action = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store state, action, and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state  # Update the current state

            if done:
                break

        results.append(state.to_dict())

        # save results
        save_jsonl(results, output_path)




if __name__ == '__main__':
    Fire(run)



