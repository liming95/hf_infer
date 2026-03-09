
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

class HFEngine:
    def __init__(self, model_name_or_path: str, device="cuda"):
        self.model_name_or_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        self.device = device

        # Initialize the model and tokenizer here

    def generate(self, prompt: str) -> str:
        # Generate text based on the prompt using the model
        response
        return "Generated text based on the prompt"