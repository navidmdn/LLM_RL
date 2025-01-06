from accelerate.commands.config.config_args import cache_dir
from torch import inference_mode

from policy.llm_policy import LLMPolicy
from policy.fewshot_llm_stepwise_math_config import FewshotLLMStepwiseMathPolicyConfig
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from envs.env import MathExample
import re
from copy import deepcopy
import torch.optim as optim
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from contextlib import nullcontext



def truncate_response_with_stop_sequence(response: str, stop_sequence: List[str]) -> str:
    # print("raw response: ",)
    # print("-"*100)
    # print(response)
    # print("-"*100)

    # this is to handle new line by the llm itself

    # init_ws = ''
    # if response.startswith(" ") or response.startswith("\n"):
    #     if re.match(r'^\s*\n', response):
    #         init_ws = "\n"
    #     else:
    #         init_ws = " "


    # response = response.strip()
    smallest_truncation = response

    for seq in stop_sequence:

        if seq in response:
            cur_trunc = response.split(seq)[0] + seq
            if len(cur_trunc) < len(smallest_truncation):
                smallest_truncation = cur_trunc

    # print("current step adds:")
    # print("-" * 100)
    # print(f"{init_ws}{smallest_truncation}")
    # print("-" * 100)
    # return f"{init_ws}{smallest_truncation}"
    return smallest_truncation

def symb_calc_and_truncate(response: str) -> Tuple[str, bool]:
    """
    Extracts the calculation from the response, evaluates it and replaces the calculation with the result
    returns the new response and a boolean indicating if a calculation was found
    """
    # print("calling symb_calc_and_truncate on response: ", response)
    match = re.search(r"<<(.*?)>>", response)
    if match:
        calculation = match.group(1)
        calculation_expr = calculation.split('=')[0]
        # print("PERFORMING CALCULATION: ", calculation_expr)
        result = round(eval(calculation_expr), 2)
        response = re.sub(r"<<.*?>>", f"<<{calculation_expr}={result}>>", response)
        response = response.split(">>")[0] + f">>{result}"

        return response, True

    return response, False



class FewShotLLMStepwiseMathPolicy(LLMPolicy):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer = None,
            model: PreTrainedModel = None,
            config: FewshotLLMStepwiseMathPolicyConfig = FewshotLLMStepwiseMathPolicyConfig(),
            system_prompt: str=None,
            few_shot_examples: List[MathExample]=None,
            CoT_prompting: bool = False,
            use_lora: bool = False,
            lora_rank: int = 128,
            lora_alpha: int = 256,
            symbol_calculation: bool = False,
            inference_mode: bool = False,
            pretrained_model_path: str = None,
            **kwargs):
        super(FewShotLLMStepwiseMathPolicy, self).__init__(tokenizer, model)

        self.CoT_prompting = CoT_prompting

        if not inference_mode:
            self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

            if use_lora:
                peft_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules="all-linear",
                    lora_dropout=0.05,
                )
                self.model = get_peft_model(self.model, peft_config)

        else:
            self.optimizer = None
            device = kwargs.get("device", None)
            cache_dir = kwargs.get("cache_dir", None)
            self.model, self.tokenizer = self.load_model_and_tokenizer(pretrained_model_path,
                                                                       device=device, cache_dir=cache_dir)
            self.model.eval()


        self.config = config
        if few_shot_examples is None:
            few_shot_examples = []
        self.few_shot_examples = few_shot_examples
        self.system_prompt = system_prompt
        self.device = self.model.device
        print("instantiating model on device: ", self.device)
        self.symbol_calculation = symbol_calculation
        print("Using symbol calculation: ", self.symbol_calculation)


    def calculate_average_gradient_norm(self):
        total_norm = 0.0
        count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item()
                count += 1
        return total_norm / count if count > 0 else 0.0

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


    def batch_action_rollout(self, states):
        # assert not self.symbol_calculation, "Symbol calculation is not supported in batch action rollout"
        self.tokenizer.padding_side = 'left'

        few_shot_messages = []
        for ex in self.few_shot_examples:
            few_shot_messages.append({"role": "user", "content": ex.problem})
            few_shot_messages.append({"role": "assistant", "content": ex.solution})

        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    *few_shot_messages,
                    {"role": "user", "content": state.problem}
                ],
                add_generation_prompt=True,
                tokenize=False,
            ) for state in states
        ]

        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=False)

        input_ids = inputs['input_ids'].to(self.device)
        attention_masks = inputs['attention_mask'].to(self.device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_masks,
            max_new_tokens=self.config.evaluation_max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.evaluation_temperature,
            top_p=self.config.evaluation_top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        resps = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=True)
        return resps


    def generate_with_input(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=False)

        terminators = [
            self.tokenizer.eos_token_id,
            # todo: add whatever token is necessary
            # tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        input_ids = inputs['input_ids'].to(self.device)
        attention_masks = inputs['attention_mask'].to(self.device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_masks,
            max_new_tokens=self.config.max_new_tokens,
            eos_token_id=terminators,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        resp = self.tokenizer.batch_decode(output[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=False)[0]
        return resp


    def get_action(self, state: MathExample, n_recursion: int = 0) -> str:
        assert not state.is_concluded, f"Concluded examples should not be passed to the model for generation:\n{state}"

        few_shot_messages = []
        for ex in self.few_shot_examples:
            few_shot_messages.append({"role": "user", "content": ex.problem})
            few_shot_messages.append({"role": "assistant", "content": ex.solution})

        prompt = [
            {"role": "system", "content": self.system_prompt},
            *few_shot_messages,
            {"role": "user", "content": state.problem}
        ]

        trail_newline_or_space = ""
        if state.is_in_progress():

            # todo: https://github.com/vllm-project/vllm/issues/9547 a bug in the tokenizer
            # doesnt allow continuation for sequences ending in \n so we strip here
            if len(state.generated_solution.rstrip()) != len(state.generated_solution):
                trail_newline_or_space = state.generated_solution[-1]
                assert trail_newline_or_space in [' ',
                                                  '\n'], f"Expected a space or newline at the end of the generated solution, got: {trail_newline_or_space}"

            prompt.append({"role": "assistant", "content": state.generated_solution.rstrip()})
            updated_prompt = self.tokenizer.apply_chat_template(
                prompt,
                continue_final_message=True,
                tokenize=False,
            )
            updated_prompt += trail_newline_or_space

        else:
            updated_prompt = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )


        resp = self.generate_with_input(updated_prompt)
        step_response = truncate_response_with_stop_sequence(resp, ['\n'])

        if not self.symbol_calculation:
            return step_response
        else:
            try:
                if "<<" in step_response and ">>" in step_response:
                    constructed_response = ""
                    for _ in range(4):

                        step_calculation, calc_exists = symb_calc_and_truncate(step_response)
                        constructed_response += step_calculation
                        if not calc_exists:
                            break

                        if prompt[-1]['role'] == "assistant":
                            prompt[-1]['content'] += step_calculation
                        else:
                            prompt.append({"role": "assistant", "content": constructed_response})

                        prompt_enc = self.tokenizer.apply_chat_template(
                            prompt,
                            continue_final_message=True,
                            tokenize=False,
                        )

                        resp = self.generate_with_input(prompt_enc)
                        step_response = truncate_response_with_stop_sequence(resp, ['\n'])

                    return constructed_response
                else:
                    return step_response
            except Exception as e:
                print("Error in symbol calculation: ", e)
                return step_response\


    def calculate_action_probabilities(self, states: List[MathExample], actions: List[str], inference_mode=False) -> Tuple[torch.Tensor, ...]:
        self.tokenizer.padding_side = 'right'

        context = torch.no_grad() if inference_mode else nullcontext()

        few_shot_messages = []
        for ex in self.few_shot_examples:
            few_shot_messages.append({"role": "user", "content": ex.problem})
            few_shot_messages.append({"role": "assistant", "content": ex.solution})

        full_prompts = []
        state_input_chat_prompts = []

        for state, action in zip(states, actions):
            prompt = [
                {"role": "system", "content": self.system_prompt},
                *few_shot_messages,
                {"role": "user", "content": state.problem},
            ]

            trail_newline_or_space_in_state = ""

            # todo: https://github.com/vllm-project/vllm/issues/9547 a bug in the tokenizer
            # doesnt allow continuation for sequences ending in \n so we strip here
            if len(state.generated_solution.rstrip()) != len(state.generated_solution):
                trail_newline_or_space_in_state = state.generated_solution[-1]
                assert trail_newline_or_space_in_state in [' ', '\n'], f"Expected a space or newline at the end of the generated solution, got: {trail_newline_or_space_in_state}"


            full_prompt = prompt + [{"role": "assistant", "content": (state.generated_solution + action).rstrip()}]
            state_prompt = prompt + [{"role": "assistant", "content": state.generated_solution.rstrip()}]


            state_prompt = self.tokenizer.apply_chat_template(
                state_prompt,
                continue_final_message=True,
                tokenize=False,
            )

            full_prompt = self.tokenizer.apply_chat_template(
                full_prompt,
                continue_final_message=True,
                tokenize=False,
            )

            state_prompt += trail_newline_or_space_in_state

            full_prompts.append(full_prompt)
            state_input_chat_prompts.append(state_prompt)

        state_inputs = self.tokenizer(state_input_chat_prompts, return_tensors='pt', padding=True, truncation=False)
        # only encoding action without any special tokens
        full_inputs = self.tokenizer(full_prompts, return_tensors='pt', padding=True, truncation=False,)

        full_input_ids = full_inputs['input_ids'].to(self.device)
        full_attention_masks = full_inputs['attention_mask'].to(self.device)
        position_ids = full_attention_masks.cumsum(1) - full_attention_masks.long().to(self.device)

        with context:
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_masks,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True
            )

        logits = outputs.logits

        if inference_mode:
            logits = logits.detach()

        assert logits.shape[1] == full_input_ids.shape[1], f"Expected output length to be equal to input length, got {outputs.shape[1]} and {full_input_ids.shape[1]}"

        #todo: this sometimes misses the first token as we are tokenizing state and action separately (newlines, spaces at the end)
        state_lens = state_inputs['attention_mask'].to(self.device).sum(1)

        # set state tokens of full prompt to pad token
        state_pad_mask = torch.arange(full_input_ids.shape[1], device=self.device).unsqueeze(0) < state_lens.unsqueeze(1)
        action_input_ids = torch.masked_fill(full_input_ids, state_pad_mask, self.tokenizer.pad_token_id)
        # we also need to keep the last token of the state
        action_pad_mask = torch.arange(full_input_ids.shape[1], device=self.device).unsqueeze(0) >= state_lens.unsqueeze(1)
        action_pad_mask = torch.masked_fill(action_pad_mask, ~full_attention_masks.bool(), False)

        action_probs_dist = F.softmax(logits, dim=-1)

        #shift probs to the right by 1 -> (i2 label for p1) (i3 label for p2) ...
        action_probs_dist = torch.roll(action_probs_dist, shifts=1, dims=1)

        action_probs = torch.gather(action_probs_dist, dim=-1, index=action_input_ids.unsqueeze(-1)).squeeze(-1)
        # ignoring last action probability as it doesn't correspond to an action token
        action_probs_masked: Tuple = tuple(p[m][:-1] for p, m in zip(action_probs, action_pad_mask))


        #for debugging
        # print("full action_probs: ", action_probs.tolist())

        # action_prob_dist_masked = tuple(p[m][:-1] for p, m in zip(action_probs_dist, action_pad_mask))
        # for action, state, full_ids, state_ids, probs, prob_dists in zip(actions, state_input_chat_prompts, full_input_ids, state_inputs['input_ids'], action_probs_masked, action_prob_dist_masked):
        #     action_token_ids = torch.argmax(prob_dists, dim=-1)
        #     print("action token ids:", action_token_ids)
        #     decoded_action = self.tokenizer.decode(action_token_ids, skip_special_tokens=False)
        #
        #     print("state: ")
        #     print("==" * 100)
        #     print(state[-100:])
        #     print("==" * 100)
        #     print("state ids: ")
        #     print("==" * 100)
        #     print(state_ids[-200:])
        #     print("==" * 100)
        #     print("full ids: ")
        #     print("==" * 100)
        #     print(full_ids[-200:])
        #     print("==" * 100)
        #     print("actions: ")
        #     print("==" * 100)
        #     print(action)
        #     print("==" * 100)
        #     print("decoded action: ")
        #     print("==" * 100)
        #     print(decoded_action)
        #     print("==" * 100)
        #     print(probs)
        #     print("==" * 100)

        return action_probs_masked

    def update(self, max_grad_norm: float = 1.0):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()





