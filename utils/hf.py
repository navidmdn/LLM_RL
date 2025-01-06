from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
import torch



def load_test_llama3_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_size = 256
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    model = LlamaForCausalLM(config=config)
    return model


def load_model_and_tokenizer(model_name_or_path: str, local_test=False, quantized=False, device='cuda', cache_dir=None):
    if not local_test:
        if quantized:
            print("Loading quantized model")
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

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer