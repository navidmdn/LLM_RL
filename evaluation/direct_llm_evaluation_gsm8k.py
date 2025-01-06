"""
This script is used to generate responses from a model using the Hugging Face library. It is used to generate responses
from either a model on HF or a fine-tuned model. The script takes in a test file, model path, and other parameters to
generate responses from the model. The responses are then saved to a file.

Note that you need to set model specific configs like the tokenizer, model, and other parameters in the script.
"""

from fire import Fire
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from utils.misc import load_json, load_prompt
import os


SYSTEM_MSG = "Your task is to solve a mathematical problem step by step. mark your final answer with 'the answer is:` phrase."

def generate_resp(model, tokenizer, test_examples, few_shot_examples, sys_msg, use_CoT = False,
                  device = torch.device('cpu'), temperature=0.6, top_p=0.9,
                  max_new_tokens=1024, do_sample=True):

    few_shot_messages = []
    for ex in few_shot_examples:
        few_shot_messages.append({"role": "user", "content": ex['problem']})
        few_shot_messages.append({"role": "assistant", "content": ex['solution']})

    prompts = []

    for ex in test_examples:
        cur_prompts = [
            {"role": "system", "content": sys_msg},
            *few_shot_messages,
            {"role": "user", "content": ex['problem']},
        ]
        if use_CoT:
            cur_prompts.append({"role": "assistant", "content": "Let's think step by step:\n"})
        prompts.append(cur_prompts)


    input_chat_prompts = [tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt= not use_CoT,
        continue_final_message=use_CoT,
        tokenize=False,
    ) for prompt in prompts]

    # print(input_chat_prompts[0])

    inputs = tokenizer(input_chat_prompts, return_tensors='pt', padding=True, truncation=False)

    terminators = [
        tokenizer.eos_token_id,
    ]

    input_ids = inputs['input_ids'].to(device)
    attention_masks = inputs['attention_mask'].to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_masks,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    resps = tokenizer.batch_decode(output[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=True)
    return resps



def generate_responses(train_path: str = 'data/train.json', test_path: str = 'data/test.json',
                       model_path: str = 'outputs/', save_batch_size: int = 10,
                       save_path: str = "data/results.json", model_response_col: str = "generated_solution",
                       n_few_shot_examples: int = 8, use_CoT: bool = False, qlora: bool = False,
                       cache_dir: str = None):

    few_shot_examples = load_json(train_path)[:n_few_shot_examples]
    test_data = load_json(test_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("total validation data:", len(test_data))

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
            model_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
            cache_dir=cache_dir,
        )
    else:
        #todo: set correct data type for the model you're testing
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    result = []
    i = 0
    def checkpoint(data, save_path):

        # check if directory exists otherwise create it
        save_dir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    pbar = tqdm(total=len(test_data))

    while i < len(test_data):
        examples = test_data[i:i+save_batch_size]
        resps = generate_resp(
            model=model,
            tokenizer=tokenizer,
            test_examples=examples,
            few_shot_examples=few_shot_examples,
            sys_msg=SYSTEM_MSG,
            device=device,
            use_CoT=use_CoT,
        )

        for ex, resp in zip(examples, resps):
            # print("____ gt sol _____")
            # print(ex['solution'])
            # print("____ pred sol _____")
            # print(resp)
            res_json = ex.copy()
            res_json[model_response_col] = resp
            result.append(res_json)

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)

        i += save_batch_size
        pbar.update(save_batch_size)

    checkpoint(result, save_path)


if __name__ == '__main__':
    Fire(generate_responses)