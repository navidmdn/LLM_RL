from copy import deepcopy
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.llava.convert_llava_weights_to_hf import convert_llava_llama_to_hf

from envs.env import Environment
from utils.misc import load_json
import numpy as np
import re
from envs.env import MathExample
from envs.reward import RewardModel
from typing import List, Union, Tuple, Dict
from algorithms.common import Buffer
from datasets import load_dataset, Dataset


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

class MATHExample(MathExample):
    def __init__(self, id, problem: str, solution: str, level: int, type: str):
        super(MATHExample, self).__init__(problem, solution, id)

        def convert_to_list_of_sentences(text):
            sents = text.split(". ")
            sents = [s + '.' for s in sents[:-1]] + [sents[-1]]
            return sents

        self.solution = "\n".join(convert_to_list_of_sentences(solution))

        self.type = type
        self.level = level
        self.final_solution = last_boxed_only_string(solution)

        self.generated_solution = ""
        self.steps = ["",]
        self.scores = []


    def is_in_progress(self):
        return len(self.generated_solution) > 0

    @property
    def is_concluded(self):
        #todo: adding eot_id for llama3; make sure you configure it beforehand for different models when instantiating env
        if last_boxed_only_string(self.generated_solution.lower()) is not None or "<|eot_id|>" in self.generated_solution:
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
    def get_reward(self, current_state: MATHExample, action: str, next_state: MATHExample):
        possible_ans = last_boxed_only_string(action.lower())
        if possible_ans is not None and possible_ans == current_state.final_solution:
            return 1.0
        return 0.0


class MATHEnv(Environment):
    def __init__(self, dataset_name: str = "lighteval/MATH", split: str = 'train',
                 reward_model: Union[RewardModel, str] = "final_answer", n_few_shot_examples=5,
                 deterministic: bool = False, cache_dir: str = None,
                 **kwargs):
        dataset = load_dataset(dataset_name, 'all', cache_dir=cache_dir)
        data = dataset[split]

        print("Total env data:", len(data))

        assert len(data) > 0, "No training data found"
        assert set(data[0].keys()) == {"problem", "solution", "level", "type"}, "Invalid data format"

        def convert_to_list_of_dict(ds: Dataset) -> List[Dict]:
            list_of_dict = []
            for i, d in enumerate(ds):
                ex = {'id': i, **d}
                list_of_dict.append(ex)
            return list_of_dict

        data = convert_to_list_of_dict(data)
        if n_few_shot_examples > 0:
            self.few_shot_examples = data[:n_few_shot_examples]
            data = data[n_few_shot_examples:]

        self.data = [MATHExample(**d) for d in data]
        self.few_shot_examples = [MATHExample(**d) for d in self.few_shot_examples]
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
        print("restarting environment...")
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
