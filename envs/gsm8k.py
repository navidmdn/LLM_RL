from copy import deepcopy
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from envs.env import Environment
from utils.misc import load_json, load_prompt
import numpy as np
from envs.env import MathExample
from envs.reward import RewardModel
from typing import List, Union, Tuple
from algorithms.common import Buffer


def get_final_answer(full_answer):
    #remove eot token if present
    full_answer = full_answer.replace("<|eot_id|>", "")

    full_answer = full_answer.lower()
    idx = full_answer.rfind("the answer is")
    if idx == -1:
        return None
    else:
        answer = full_answer[idx + len("the answer is: "):]
        answer = answer.replace(":", "").replace("$", "").strip()
        if len(answer)> 0:
            if answer[-1] == ".":
                answer = answer[:-1]
            left = "\\boxed{"
            if answer[:len(left)] == left and answer[-1] == "}":
                answer = answer[len(left):-1]
        return answer.replace(",", "")

class GSM8KExample(MathExample):
    def __init__(self, id, problem: str, solution: str):
        super(GSM8KExample, self).__init__(problem, solution, id)

        self.final_solution = get_final_answer(solution)
        assert self.final_solution is not None, f"Final solution not found in {solution}"

        self.generated_solution = ""
        self.steps = ["",]
        self.scores = []

    def is_in_progress(self):
        return len(self.generated_solution) > 0

    @property
    def is_concluded(self):
        #todo: adding eot_id for llama3; make sure you configure it beforehand for different models when instantiating env
        if "the answer is:" in self.generated_solution.lower() or "<|eot_id|>" in self.generated_solution:
            return True
        return False

    def __repr__(self):
        pretty_steps = ""
        for i, step in enumerate(self.steps):
            pretty_steps += f"{i+1}. {step.strip()}\n"
        return (f"Problem:\n{self.problem}\n\nGenerated Solution:\n{self.generated_solution}\n\n"
                f"Solution Steps:\n{pretty_steps}\n\nGT:\n{self.solution}\n\n"
                f"Scores: {self.scores}")


    def update_solution(self, solution_step: str):
        # delegating the newline handling to the LLM
        self.generated_solution += f"{solution_step}"
        if '\n' not in solution_step:
            self.steps[-1] += f" {solution_step}"
        else:
            self.steps.append(solution_step.strip())


    def to_dict(self):
        return {
            "id": self.id,
            "problem": self.problem,
            "solution": self.solution,
            "generated_solution": self.generated_solution
        }

class FinalAnswerRewardModel(RewardModel):
    def get_reward(self, current_state: GSM8KExample, action: str, next_state: GSM8KExample):
        if "the answer is:" in action.lower():
            final_solution = get_final_answer(action)
            if final_solution is None:
                return 0.0

            if final_solution == current_state.final_solution:
                return 1.0
        return 0.0

    def batch_rollout_reward(self, states: List[GSM8KExample], full_action_rollouts: List[str]):
        rewards = []
        for state, action_rollout in zip(states, full_action_rollouts):
            last_action = action_rollout.split("\n")[-1]
            rewards.append(self.get_reward(state, last_action, None))
        return rewards


class StepwiseMultiDimensionVerbalRewardModel(RewardModel):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, system_prompt_paths: str,
                 final_answer_reward: float = None):
        super(StepwiseMultiDimensionVerbalRewardModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.tokenizer.padding_side = 'left'
        self.final_answer_reward = final_answer_reward

        self.eval_prompts = load_json(system_prompt_paths)


        self.numerical_score_tokens = self.tokenizer.convert_tokens_to_ids([
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10"
        ])

        self.yn_score_tokens = self.tokenizer.convert_tokens_to_ids([
            "No",
            "Yes"
        ])

        for token in self.numerical_score_tokens:
            assert token is not None, f"Token {token} not found in the tokenizer"

        for token in self.yn_score_tokens:
            assert token is not None, f"Token {token} not found in the tokenizer"

    def get_reward(self, current_state: GSM8KExample, action: str, next_state: GSM8KExample):

        if self.final_answer_reward is not None:
            if "the answer is:" in action.lower():
                final_solution = get_final_answer(action)
                if final_solution == current_state.final_solution:
                    return 1.0

        assert set(self.eval_prompts.keys()) == {'additional_value', 'progression', 'mathematical_soundness'}

        # build prompts one by one

        prompt_info = self.eval_prompts['additional_value']
        additional_value_message = [
            {"role": "system", "content": prompt_info['system_prompt']},
            {"role": "user", "content": prompt_info['eval_prompt_template'].format(
                problem=current_state.problem, action=action.strip(), current_solution=current_state.generated_solution.strip())},
        ]

        prompt_info = self.eval_prompts['progression']
        progression_message = [
            {"role": "system", "content": prompt_info['system_prompt']},
            {"role": "user", "content": prompt_info['eval_prompt_template'].format(
                problem=current_state.problem, action=action.strip(), current_solution=current_state.generated_solution.strip())},
        ]

        prompt_info = self.eval_prompts['mathematical_soundness']
        mathematical_soundness_message = [
            {"role": "system", "content": prompt_info['system_prompt']},
            {"role": "user", "content": prompt_info['eval_prompt_template'].format(
                problem=current_state.problem, action=action.strip(), current_solution=current_state.generated_solution.strip())},
        ]

        prompts = [additional_value_message, progression_message, mathematical_soundness_message]

        prompts = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=False
        )

        # print("PROMPTS:", prompts)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            logits = output.logits

            # for debugging
            # for l in logits:
            #     token_id = torch.argmax(l[-1])
            #     print("argmax logit:", self.tokenizer.convert_ids_to_tokens(token_id.item()))

            # Extracting the logits for the score tokens
            score_logits = logits[:, -1, self.yn_score_tokens]
            score_probs = torch.softmax(score_logits, dim=-1)[:, 1]

            for n, s in zip(['add_r', 'prog_r', 'math_r'], score_probs):
                print(f"{n}: {s}")

            final_score = score_probs.mean().item()
        return final_score



class StepwiseVerbalRewardModel(RewardModel):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, system_prompt_path: str,
                 intermediate_steps_reward_scale_factor: float = 1.0):
        super(StepwiseVerbalRewardModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.intermediate_steps_reward_scale_factor = intermediate_steps_reward_scale_factor

        with open(system_prompt_path, 'r') as f:
            self.system_prompt = f.read().strip()

        self.score_tokens = self.tokenizer.convert_tokens_to_ids([
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10"
        ])

        for token in self.score_tokens:
            assert token is not None, f"Token {token} not found in the tokenizer"

    @staticmethod
    def build_eval_prompt(current_state: GSM8KExample, action: str):
        return f"Problem:\n{current_state.problem}\n\nProposed next step:\n{action.strip()}"

    def get_reward(self, current_state: GSM8KExample, action: str, next_state: GSM8KExample):

        eval_prompt = self.build_eval_prompt(current_state, action)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": eval_prompt},
            {"role": "assistant", "content": "Score (from 1 to 10):"}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            continue_final_message=True,
            tokenize=False
        )

        #todo: make sure scores are not tokenized in your tokenization with a tail space
        prompt += " "

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                return_dict=True,
            )

            logits = output.logits

            # Extracting the logits for the score tokens
            score_logits = logits[0, -1, self.score_tokens]
            score_probs = torch.softmax(score_logits, dim=-1)
        weighted_score = (torch.sum(score_probs * torch.arange(1, 11).to(score_probs.device)) / 10.0).item()
        return weighted_score * self.intermediate_steps_reward_scale_factor


class GSM8kEnv(Environment):
    def __init__(self, data_file: str, reward_model: Union[RewardModel, str] = "final_answer", n_few_shot_examples=5,
                 deterministic: bool = False, max_samples: int = None, train_data_file: str = None,
                 **kwargs):
        data = load_json(data_file)

        print("Total env data:", len(data))

        assert len(data) > 0, "No training data found"
        assert set(data[0].keys()) == {"id", "problem", "solution"}, "Invalid data format"

        if n_few_shot_examples > 0:
            if train_data_file is not None:
                print("taking few shot examples from train data")
                train_data = load_json(train_data_file)
                self.few_shot_examples = train_data[:n_few_shot_examples]
            else:
                print("taking few shot examples from main data source")
                self.few_shot_examples = data[:n_few_shot_examples]
                data = data[n_few_shot_examples:]
            self.few_shot_examples = [GSM8KExample(**data) for data in self.few_shot_examples]
        else:
            self.few_shot_examples = []

        if max_samples is not None:
            data = data[:max_samples]

        self.data = [GSM8KExample(**data) for data in data]
        self.deterministic = deterministic
        self.deterministic_idx = 0


        self.current_state = None

        if isinstance(reward_model, str):
            if reward_model == "final_answer":
                self.reward_model = FinalAnswerRewardModel()
            else:
                raise NotImplementedError(f"Reward model {reward_model} not implemented")
        else:
            self.reward_model = reward_model

        self.reset()

    def reset(self):
        if not self.deterministic:
            self.current_state = deepcopy(self.data[np.random.randint(0, len(self.data))])
        else:
            self.current_state = deepcopy(self.data[self.deterministic_idx])
            self.deterministic_idx = (self.deterministic_idx + 1) % len(self.data)

        return self.current_state

    def is_done(self):
        return self.current_state.is_concluded

    def step(self, action: str):
        print("Taking action:", action)
        current_state = deepcopy(self.current_state)
        self.current_state.update_solution(action)
        next_state = deepcopy(self.current_state)
        reward = self.reward_model.get_reward(current_state, action, next_state)
        print("Reward:", reward)
        print("Done: ", self.is_done())

        return next_state, reward, self.is_done(), None


    def render(self):
        pass


    def close(self):
        pass

class GSM8kBuffer(Buffer):
    def __init__(self, buffer_size: int = 1000):
        super(GSM8kBuffer, self).__init__(buffer_size)

    def random_sample(self, batch_size: int, ref_action: bool = False, **kwargs) -> Tuple:
        indices = np.random.choice(len(self.states), batch_size)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        if ref_action:
            ref_actions = [self.ref_actions[i] for i in indices]
            return states, actions, rewards, ref_actions
        return states, actions, rewards

    def sample_balanced(self, batch_size: int, ref_action: bool = False, **kwargs) -> Tuple:
        pass

    def sample(self, batch_size: int, ref_action: bool = False, **kwargs) -> Tuple:
       sampling_type = kwargs.get('sampling_type', 'random')
       if sampling_type == 'random':
           return self.random_sample(batch_size, ref_action)
       elif sampling_type == 'balanced':
           return self.sample_balanced(batch_size, ref_action)

