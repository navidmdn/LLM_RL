from policy.fewshot_llm_stepwise_math_policy import FewShotLLMStepwiseMathPolicy
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
from envs.env import MathExample
from envs.gsm8k import GSM8KExample
from utils.misc import load_test_llama3_model
import torch


def test_fewshot_llm_stepwise_math_policy_calculate_action_probabilities(model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    model = load_test_llama3_model(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    fewshot_examples = [
        MathExample( "and 2+2", "=4", "1"),
    ]

    states = [
        GSM8KExample(id="1", problem="2+2", solution="=4"),
        GSM8KExample(id="2", problem="3+3", solution="=6"),
        GSM8KExample(id="3", problem="3+5", solution="=8"),
    ]

    #first one already has a solution
    states[0].generated_solution = "i think "
    #second one is at the beginning
    states[1].generated_solution = "so"
    states[2].generated_solution = "lets see step by step:"

    actions = [
        " it is equal to 4",
        "=6",
        "\nfirst we"
    ]

    policy = FewShotLLMStepwiseMathPolicy(tokenizer, model, system_prompt="", few_shot_examples=fewshot_examples)
    action_probabilities = policy.calculate_action_probabilities(states, actions)
    print(action_probabilities)

test_fewshot_llm_stepwise_math_policy_calculate_action_probabilities()