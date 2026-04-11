from .hf_tools import history_speculative_decoding
from .candidate_generator import HistoryDB

class SDEngine(HFEngine):
    def __init__(self, model_name_or_path, device="cuda"):
        super().__init__(model_name_or_path, device=device)

    def generate(self, prompt: str) -> str:
        # prepare input_ids
        # decode
        return super().generate(prompt)
    
    def sd_generate(self, prompt: str, history_db: HistoryDB) -> str:
        # Placeholder for future implementation of a custom decoding strategy
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # gen_config = self.model.generation_config.clone()
        import copy
        # gen_config = copy.deepcopy(self.model.generation_config)

        # gen_config.historydb = history_db
        gen_config = copy.deepcopy(self.model.generation_config)

        gen_config.historydb = history_db
        gen_config.prompt_lookup_num_tokens = 10
        gen_config.max_matching_ngram_size = 4

        sd_generated_ids = self.model.generate(
            **inputs,
            generation_config=gen_config,
            custom_generate=history_speculative_decoding,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stopping_criteria=self.stop_criteria
        )

        prompt_len = inputs["input_ids"].shape[1]
        sd_generated_ids = sd_generated_ids[0][prompt_len:]
        return self.tokenizer.decode(sd_generated_ids, skip_special_tokens=True)

def test_sd_engine():
    historydb = HistoryDB()
    model_path = "/workspace/models/Qwen2.5-1.5B-Instruct"
    engine = SDEngine(model_path)

    # history data generation
    prompt = "What is the capital of France?"
    history_data = prompt + engine.generate(prompt)
    
    print("History data:", history_data)
    history_ids = engine.tokenizer(history_data, return_tensors="pt")["input_ids"][0].to(engine.device)
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt").to(engine.device)
    prompt_ids = prompt_ids['input_ids'][0].tolist()
    historydb.add(prompt_ids, history_ids)

    # sd generation
    sd_generated_text = engine.sd_generate(prompt, historydb)
    print("Generated output:", sd_generated_text)

if __name__ == "__main__":
    test_sd_engine()

  
# @torch.no_grad()
# def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
#     '''
#         Finds candidate prediction tokens based on n-gram matching in the input_ids.
#         input_ids: Tensor of shape (batch_size, sequence_length) containing token IDs of the input sequence.
#         max_ngram_size: Maximum size of n-grams to consider for matching.
#         num_pred_tokens: Number of prediction tokens to return after a match is found.
#     '''
#     input_length = input_ids.size(1)

#     if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
#         raise ValueError("Invalid max_ngram_size or num_pred_tokens")

#     for ngram_size in range(max_ngram_size, 0, -1):
#         ngram = input_ids[0, -ngram_size:].tolist()

#         # Create sliding windows of size ngram_size
#         windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

#         # Convert ngram to a tensor for comparison
#         ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

#         # Find where the windows match the ngram
#         matches = (windows == ngram_tensor).all(dim=2)

#         # Get the indices of matches
#         match_indices = matches.nonzero(as_tuple=True)[1]

#         # Iterate through match indices to find a valid continuation
#         for idx in match_indices:
#             start_idx = idx + ngram_size
#             end_idx = start_idx + num_pred_tokens
#             # Ensure we don't go beyond the length of input_ids and avoid self-match
#             if end_idx <= input_length and start_idx < input_length - ngram_size:
#                 return input_ids[0, start_idx:end_idx]

#     # If no match is found, return an empty tensor
#     return torch.tensor([], dtype=torch.long, device=input_ids.device)

