
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch
from threading import Thread
from .tool import check_output, token_match_rate, debug_token_diff, compare_tokens

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids shape: (batch_size, seq_len)
        return input_ids[0, -1].item() in self.stop_token_ids


def my_decode(
    model,
    input_ids,
    generation_config=None,
    **kwargs
):

    strategy = generation_config.my_strategy
    print(f"Using decoding strategy: {strategy}")

class HFEngine:
    def __init__(self, model_name_or_path: str, device="cuda"):
        self.model_name_or_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        self.device = device
        self.max_new_tokens = 2048
        self.do_sample = False
        self.temperature = 0
        self.top_p = 1
        self.top_k = 100000000
        self.stop_criteria = StoppingCriteriaList([
            StopOnToken([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
        ])

        # Initialize the model and tokenizer here

    def generate(self, prompt: str) -> str:
        # Generate text based on the prompt using the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stopping_criteria=self.stop_criteria
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def stream_generate(self, prompt: str) -> str:
        """
        streaming style generation (similar to sglang)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            streamer=streamer,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stopping_criteria=self.stop_criteria
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        output_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            output_text += new_text
        thread.join()
        return output_text
    
    def sd_generate_test(self, prompt: str) -> str:
        # Placeholder for future implementation of a custom decoding strategy
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # gen_config = self.model.generation_config.clone()
        import copy
        gen_config = copy.deepcopy(self.model.generation_config)

        gen_config.my_strategy = "tree"

        self.model.generate(
            **inputs,
            generation_config=gen_config,
            custom_generate=my_decode,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stopping_criteria=self.stop_criteria
        )

def test_sd_generate():
    model_path = "/workspace/models/Qwen2.5-1.5B-Instruct"
    engine = HFEngine(model_path)
    prompt = "What is the capital of France?"
    response = engine.sd_generate_test(prompt)
    print(f"SD Generated Response: {response}")

# Example usage:
def test_engine():
    model_path = "/workspace/models/Qwen2.5-1.5B-Instruct"
    engine = HFEngine(model_path)
    prompt = "What is the capital of France?"
    response = engine.generate(prompt)
    streamed_response = engine.stream_generate(prompt)
    print('\n')
    print('-----------------------------')
    print(f'Response: {response}')
    print('\n')
    print(f'Streamed Response: {streamed_response}')
    print('-----------------------------')
    # Compare the two responses
    check_result = check_output(response, streamed_response)
    print(f'Comparison Result: {check_result}')
    token_match = token_match_rate(engine.tokenizer, response, streamed_response)
    print(f'Token Match Rate: {token_match:.2%}')
    first_diff = compare_tokens(engine.tokenizer, response, streamed_response)
    if not first_diff["match"]:
        print(f'First difference at token position {first_diff["diff_position"]}:')
        print(f'Token in response: {first_diff.get("token1", "N/A")}')
        print(f'Token in streamed response: {first_diff.get("token2", "N/A")}')
    else:
        print("The two responses are identical in tokens.")
    debug_token_diff(engine.tokenizer, response, streamed_response)

if __name__ == "__main__":
    # test_engine()
    test_sd_generate()