from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import torch


class LLMPolicy:
    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        self.tokenizer = tokenizer
        self.model = model

    def save_model_and_tokenizer(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model_and_tokenizer(self, path: str, device: torch.device = torch.device('cpu'), cache_dir: str = None):
        model = AutoModelForCausalLM.from_pretrained(path, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
        model = model.to(device)

        return model, tokenizer

    def batch_action_rollout(self, states):
        NotImplementedError()