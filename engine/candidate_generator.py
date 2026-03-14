from typing import Callable, Optional
import torch

class HistoryDB:
    def __init__(self):
        self.database = {} # {prompt_ids: [sequence_length]} tensor of token ids, or a more efficient data structure for storing and querying n-grams
    
    def add(self, prompt_ids, token_ids: torch.LongTensor):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        self.database[tuple(prompt_ids)] = token_ids
    
    @torch.no_grad()
    def _find_candidate_pred_tokens(self, key_ids, history_token_ids, num_pred_tokens, filter: Callable):
        filtered_chosen_ids = None
        match_found = False

        history_length = history_token_ids.size(1)
        for ngram_size in range(len(key_ids), 0, -1):
            # Create sliding windows of size ngram_size
            windows = history_token_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = torch.tensor(key_ids[-ngram_size:], device=history_token_ids.device).unsqueeze(0)

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            # TODO (joao): this finds the first valid candidates (left to right), but perhaps we should find the
            # longest valid candidates?
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + num_pred_tokens
                end_idx = min(end_idx, history_length)  # Ensure we don't go beyond the length of history_token_ids

                if start_idx < end_idx:
                    chosen_ids = history_token_ids[0, start_idx:end_idx]
                    filtered_chosen_ids, match_found = filter(chosen_ids)

                    if match_found:
                        break
            if match_found:
                break
        return filtered_chosen_ids
    
    def query(self, prompt_ids, key_ids: torch.LongTensor, num_pred_tokens=10, filter: Callable = None) -> torch.LongTensor:
        # Query the history database for candidate prediction tokens based on input_ids
        history_token_ids = self.database.get(tuple(prompt_ids), None)
        if history_token_ids is None:
            return None
        # history_token_ids = history_token_ids  # Add batch dimension
        candidate_tokens = self._find_candidate_pred_tokens(key_ids, history_token_ids, num_pred_tokens, filter)
        return candidate_tokens

class HistoryLookupCandidateGenerator():
    def __init__(
        self,
        eos_token_id: torch.Tensor | None = None,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = 4,
        max_length: int = 20,
        logits_processor: Optional["LogitsProcessorList"] = None,
        vocab_size: int | None = None,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    @torch.no_grad()
    def get_candidates(self, prompt_ids, input_ids: torch.LongTensor, historydb) -> tuple[torch.LongTensor, None]:
        bsz, input_length = input_ids.shape

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        if self.max_length == input_length + 1:
            return input_ids, None

        n_gram_size = min(self.max_matching_ngram_size, input_length - 1)
        key_ids = input_ids[0, -n_gram_size :].tolist()
        num_pred_tokens = min(self.num_output_tokens, self.max_length - input_length - 1)

        def filter_candidates(chosen_ids: torch.LongTensor) -> tuple[torch.LongTensor, bool]:
            ''' 
                Check if the each new candidate token is forbidden according to the logits processor. If all
                tokens are allowed, we keep `chosen_ids` as is.
                1. create random logits.
                2. apply the logits processor to get output logits for the next token, using the arbitrary
                    logits as input.
                3. compare the output logits with the next candidate token. If they are -inf, then the next
                    candidate token is forbidden and we don't want to generate it.
            '''

            if self.logits_processor is not None:
                sequence_with_candidate = input_ids
                fake_input_logits = torch.ones(
                    (bsz, self.vocab_size), device=input_ids.device, dtype=torch.float32
                )
                for candidate_idx, new_candidate_token in enumerate(chosen_ids):
                    fake_output_logits = self.logits_processor(sequence_with_candidate, fake_input_logits)
                    fake_candidate_logits = fake_output_logits[0, new_candidate_token]
                    # next candidate token is forbidden -> crop chosen_ids accordingly
                    if fake_candidate_logits in (-float("Inf"), torch.finfo(fake_candidate_logits.dtype).min):
                        chosen_ids = chosen_ids[:candidate_idx]
                        break
                    else:
                        sequence_with_candidate = torch.cat(
                            (input_ids, chosen_ids[: candidate_idx + 1].unsqueeze(0)), dim=1
                        )
                # no valid candidate tokens -> look for a different match
                if chosen_ids.shape[0] == 0:
                    return chosen_ids, False

            match_found = True

            # remove remaining candidate ids if an "eos" token is found, otherwise the target model may
            # accept eos and the rest as valid, thus not stopping generation after "eos"
            # NOTE: below code is written based on the fact that assisted decoding supports only bs=1
            mask = torch.isin(chosen_ids, self.eos_token_id)
            match_indices_eos = torch.nonzero(mask)
            if match_indices_eos.numel() > 0:
                first_eos_index = match_indices_eos[0].item()
                chosen_ids = chosen_ids[:first_eos_index]
            
            return chosen_ids, match_found

        chosen_ids = historydb.query(prompt_ids, key_ids, num_pred_tokens, filter_candidates)
        if chosen_ids is None or chosen_ids.numel() == 0:
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None

    
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