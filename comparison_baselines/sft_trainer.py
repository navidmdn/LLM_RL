from cProfile import label
from dataclasses import dataclass, field
import os
import torch
from datasets import load_dataset
import random
from transformers import LlamaForCausalLM, AutoConfig, EarlyStoppingCallback
import numpy as np
from trl.cli import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
import nltk
import glob
from peft import LoraConfig

from trl import (
    SFTConfig,
    SFTTrainer)

from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorMixin
from transformers import PreTrainedTokenizerBase
from typing import List, Union, Dict, Any, Optional
from utils.misc import load_prompt


class Llama3ChatCompletionDataCollator(DataCollatorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, return_tensors='pt'):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raw_input_ids: List[np.ndarray] = [e['input_ids'] for e in examples]
        raw_labels: List[np.ndarray] = [e['labels'] for e in examples]

        for inp, lab in zip(raw_input_ids, raw_labels):
            assert len(inp) == len(lab), f"Input and label lengths don't match: {len(inp)} != {len(lab)}"

        max_input_len = max([len(p) for p in raw_input_ids])

        labels = []
        input_ids = []
        attention_masks = []
        for full_input_ids, label_ids in zip(raw_input_ids, raw_labels):
            pad_len = max_input_len - len(full_input_ids)
            pad_list = [self.tokenizer.pad_token_id] * pad_len
            ignore_loss_list = [-100] * pad_len

            label_ids = np.concatenate([label_ids, ignore_loss_list])
            attention_mask = [1] * len(full_input_ids) + [0] * pad_len
            full_input_ids = np.concatenate([full_input_ids, pad_list])

            labels.append(label_ids)
            input_ids.append(full_input_ids)
            attention_masks.append(attention_mask)

        # ids = ",".join([str(x) for x in input_ids[0]])
        # labs = ",".join([str(x) for x in labels[0]])
        # print(
        #     f"input_ids ({len(input_ids[0])}): [{ids}]\n"
        #     f"labels ({len(labels[0])}): [{labs}]\n"
        #     f"attention_mask: {attention_masks[0]}\n"
        # )

        return {
            'input_ids': torch.LongTensor(np.array(input_ids)),
            'labels': torch.LongTensor(np.array(labels)),
            'attention_mask': torch.LongTensor(np.array(attention_masks))
        }


def print_batch(batch: Dict, tokenizer: PreTrainedTokenizerBase):

    input_ids_list = batch['input_ids'].numpy()
    attention_mask_list = batch['attention_mask'].numpy()
    label_list = batch['labels'].numpy()

    print("_" * 50)
    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, label_list):
        non_padded_input_ids = input_ids[attention_mask == 1]
        non_padded_label = label[attention_mask == 1]
        print("Input:")
        print(tokenizer.decode(non_padded_input_ids))

        completion = non_padded_label[non_padded_label != -100]
        print("Label:")
        print(tokenizer.decode(completion))


@dataclass
class ScriptArguments:
    train_file: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    dev_dir: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "Model ID to use for SFT training"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load the model in 4bit"}
    )
    use_peft: bool = field(
        default=False, metadata={"help": "Whether to use PEFT"}
    )
    flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use fast attention"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Path to the cache directory"}
    )
    num_return_sequences: int = field(
        default=1, metadata={"help": "Number of sequences to generate"}
    )
    local_test: bool = field(
        default=False, metadata={"help": "Whether to run a local test"}
    )
    max_new_tokens: int = field(
        default=100, metadata={"help": "Max number of tokens to generate"}
    )
    eval_do_sample: bool = field(
        default=True, metadata={"help": "Whether to sample during evaluation"}
    )
    eval_temperature: float = field(
        default=0.3, metadata={"help": "Temperature for sampling during evaluation"}
    )
    eval_top_p: float = field(
        default=0.93, metadata={"help": "Top p for sampling during evaluation"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "Alpha for LoRA"}
    )
    lora_r: int = field(
        default=64, metadata={"help": "R for LoRA"}
    )
    system_prompt_path: str = field(
        default=None, metadata={"help": "Path to the system prompt file"}
    )


def load_test_llama3_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_size = 256
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    model = LlamaForCausalLM(config=config)
    return model


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "json",
        data_files=script_args.train_file,
        split="train",
        cache_dir=script_args.cache_dir,
    )

    dev_datasets = {}
    dev_files = glob.glob(os.path.join(script_args.dev_dir, "*val.json"))
    print(f"list of validation files: {dev_files}")

    if len(dev_files) > 0:
        for dev_file_path in dev_files:
            print('loading ', dev_file_path)
            ds = load_dataset(
                "json",
                data_files=dev_file_path,
                split="train",
                cache_dir=script_args.cache_dir,
            )
            dev_split_name = dev_file_path.split("/")[-1].split(".")[0]
            dev_datasets[dev_split_name] = ds
    else:
        raise NotImplementedError("No dev files provided")

    ################
    # Model & Tokenizer
    ###############
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True, cache_dir=script_args.cache_dir)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    assert script_args.system_prompt_path is not None, "System prompt path is not provided"
    sys_prompt = load_prompt(script_args.system_prompt_path)

    def preprocess_dataset(example):
        """
        Preprocess the dataset to convert it to chat format
        :param example:
        :return:
        """
        full_chat_formatted = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": example['problem']},
            {"role": "assistant", "content": example['solution']}
        ]

        user_prompt_chat_formatted = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": example['problem']},
        ]

        full_chat_input_ids = tokenizer.apply_chat_template(
                full_chat_formatted,
                add_generation_prompt=False,
                tokenize=True
        )

        input_chat_input_ids = tokenizer.apply_chat_template(
                user_prompt_chat_formatted,
                add_generation_prompt=True,
                tokenize=True
        )

        labels = np.array(full_chat_input_ids)
        labels[:len(input_chat_input_ids)] = -100

        return {
            'input_ids': np.array(full_chat_input_ids),
            'labels': labels,
        }

    init_cols = train_dataset.column_names
    train_dataset = train_dataset.map(preprocess_dataset).remove_columns(init_cols)
    dev_datasets = {k: v.map(preprocess_dataset).remove_columns(init_cols) for k, v in dev_datasets.items()}

    data_collator = Llama3ChatCompletionDataCollator(tokenizer=tokenizer)


    device_map = None
    if 'llama-3' in script_args.model_id.lower() and not script_args.local_test:
        print("Using llama 3 recommended data type")
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = True
    elif 'llama-2' in script_args.model_id.lower() and not script_args.local_test:
        print("Using llama 2 recommended data type")
        torch_dtype = torch.float16
        quant_storage_dtype = torch.float16
        bnb_4bit_use_double_quant = False
        device_map = {"": 0}
    else:
        torch_dtype = None
        quant_storage_dtype = None
        bnb_4bit_use_double_quant = None

    quantization_config = None
    if script_args.load_in_4bit:
        print("Using 4bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    if not script_args.local_test:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if script_args.flash_attention else None,
            torch_dtype=quant_storage_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            cache_dir=script_args.cache_dir,
            device_map=device_map)
    else:
        model = load_test_llama3_model(script_args.model_id)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    with training_args.main_process_first(
            desc="Log a few random samples from the collated training set"
    ):
        # check the format of collated dataset
        dl = DataLoader(train_dataset.select(range(2)), batch_size=2, collate_fn=data_collator)
        batch = next(iter(dl))
        print_batch(batch, tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = model.to(device)
        # with torch.no_grad():
        #     batch = {k: v.to(model.device) for k, v in batch.items()}
        #     sample_loss = model(**batch).loss
        #     print("Sample loss: ", sample_loss.item())

    ################
    # PEFT
    ################

    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.05,
            r=script_args.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
        )

    def count_corrects(preds, labels):
        def get_final_answer(full_answer):
            # remove eot token if present
            full_answer = full_answer.replace("<|eot_id|>", "")

            full_answer = full_answer.lower()
            idx = full_answer.rfind("the answer is")
            if idx == -1:
                return None
            else:
                answer = full_answer[idx + len("the answer is: "):]
                answer = answer.replace(":", "").replace("$", "").strip()
                if len(answer) > 0:
                    if answer[-1] == ".":
                        answer = answer[:-1]
                    left = "\\boxed{"
                    if answer[:len(left)] == left and answer[-1] == "}":
                        answer = answer[len(left):-1]
                return answer.replace(",", "")

        corrects = 0
        for pred, label in zip(preds, labels):
            pred_answer = get_final_answer(pred)
            label_answer = get_final_answer(label)
            if pred_answer == label_answer:
                corrects += 1

        return corrects


    class EvalResult:
        def __init__(self):
            self.corrects = 0
            self.total = 0

        def reset(self):
            self.corrects = 0
            self.total = 0

        def compute_metrics(self, eval_predictions, compute_result: bool):
            #todo: do i need to revert it at the end? probably not because the training data is already processed and
            # data collator doesn't tokenize new data

            tokenizer.padding_side = 'left'
            input_ids = eval_predictions.inputs['input_ids'].cpu().numpy()
            labels = eval_predictions.label_ids.cpu().numpy()

            labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
            inputs = np.where(input_ids == -100, tokenizer.pad_token_id, input_ids)

            # separate the instruction and response
            input_txts = tokenizer.batch_decode(inputs, skip_special_tokens=False)

            # manually handle the prompt split for llama3
            prompt_split_text = "<|start_header_id|>assistant<|end_header_id|>"
            instructions = [txt.split(prompt_split_text)[0].strip() for txt in input_txts]
            instructions = [i+f"{prompt_split_text}\n\n" for i in instructions]

            inputs = tokenizer(instructions, return_tensors="pt", padding='longest', truncation=False,
                               add_special_tokens=False)
            input_lens = inputs["input_ids"].shape[1]

            decoded_preds = []
            i = 0
            batch_size = training_args.per_device_eval_batch_size
            while i < len(inputs["input_ids"]):
                input_id_list = inputs["input_ids"][i:i + batch_size]
                attention_mask_list = inputs["attention_mask"][i:i + batch_size]

                input_id_list = input_id_list.to(model.device)
                attention_mask_list = attention_mask_list.to(model.device)


                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                output = model.generate(
                    input_id_list,
                    attention_mask=attention_mask_list,
                    max_new_tokens=script_args.max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=script_args.eval_do_sample,
                    temperature=script_args.eval_temperature,
                    top_p=script_args.eval_top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )

                decoded_preds.extend(tokenizer.batch_decode(output[:, input_lens:].cpu().numpy(), skip_special_tokens=True))
                i += batch_size

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                if random.random() < 0.05:
                    print(f"Pred: {pred}")
                    print("*"*100)
                    print(f"Label: {label}")
                    print("-"*100)


            self.corrects += count_corrects(decoded_preds, decoded_labels)
            self.total += len(decoded_preds)

            if compute_result:
                result = {"accuracy": self.corrects/self.total}
                self.reset()
                return result


    eval_results = EvalResult()
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets,
        peft_config=peft_config,
        tokenizer=tokenizer,
        compute_metrics=eval_results.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    if trainer.accelerator.is_main_process and script_args.use_peft:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)