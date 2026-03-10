class SDEngine(HFEngine):
    def __init__(self, model_name_or_path, device):
        super().__init__(model_name_or_path, device)

    def generate(self, prompt: str) -> str:
        # prepare input_ids
        # decode
        return super().generate(prompt)
    
    def spec_decoding(self, ):
    
    @torch.no_grad()
    def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
        input_length = input_ids.size(1)

        # Ensure max_ngram_size and num_pred_tokens are valid
        if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
            raise ValueError("Invalid max_ngram_size or num_pred_tokens")

        for ngram_size in range(max_ngram_size, 0, -1):
            # Extract the last n tokens as our search ngram
            ngram = input_ids[0, -ngram_size:].tolist()

            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + num_pred_tokens
                # Ensure we don't go beyond the length of input_ids and avoid self-match
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return input_ids[0, start_idx:end_idx]

        # If no match is found, return an empty tensor
        return torch.tensor([], dtype=torch.long, device=input_ids.device)


