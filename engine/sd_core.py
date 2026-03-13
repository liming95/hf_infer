class HistoryDB:
    def __init__(self):
        self.db = {}
    
    def add(self, input_ids: torch.LongTensor):
        # Add input_ids to the history database
        pass
    
    def query(self, input_ids: torch.LongTensor, max_ngram_size=3, num_pred_tokens=10) -> torch.LongTensor:
        # Query the history database for candidate prediction tokens based on input_ids
        pass
historydb = HistoryDB()

class HistoryLookupCandidateGenerator(PromptLookupCandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        max_ngram_size=3,
        num_pred_tokens=10
    ):
        self.input_ids = input_ids
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
    
    @torch.no_grad()
    def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
        '''
            Finds candidate prediction tokens based on n-gram matching in the input_ids.
            input_ids: Tensor of shape (batch_size, sequence_length) containing token IDs of the input sequence.
            max_ngram_size: Maximum size of n-grams to consider for matching.
            num_pred_tokens: Number of prediction tokens to return after a match is found.
        '''
        input_length = input_ids.size(1)

        if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
            raise ValueError("Invalid max_ngram_size or num_pred_tokens")

        for ngram_size in range(max_ngram_size, 0, -1):
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

    @torch.no_grad()
    def get_candidates(self, input_ids: torch.LongTensor, historydb) -> tuple[torch.LongTensor, None]:
        return self.find_candidate_pred_tokens(input_ids, historydb, self.max_ngram_size, self.num_pred_tokens), None
    
    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Currently does nothing
        return
    



class SDEngine(HFEngine):
    def __init__(self, model_name_or_path, device):
        super().__init__(model_name_or_path, device)

    def generate(self, prompt: str) -> str:
        # prepare input_ids
        # decode
        return super().generate(prompt)
    
    def spec_decoding(self, ):
