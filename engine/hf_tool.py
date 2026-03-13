from transformers import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, BaseStreamer
from .sd_core import HistoryLookupCandidateGenerator
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)

ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]

def _prepare_attention_mask(model_kwargs: dict[str, Any], new_length: int, is_encoder_decoder: bool) -> dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)

    # Handle cross attention models
    if "cross_attention_mask" in model_kwargs:
        # Mllama case
        cross_mask = model_kwargs["cross_attention_mask"]
        if mask_length_diff < 0:
            model_kwargs["cross_attention_mask"] = cross_mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            new_mask = cross_mask[:, -1:, :, :].repeat(1, mask_length_diff, 1, 1)
            model_kwargs["cross_attention_mask"] = torch.cat([cross_mask, new_mask], dim=1)
    elif "image_attention_mask" in model_kwargs:
        # IDEFICS case
        cross_mask = model_kwargs["image_attention_mask"]
        if mask_length_diff < 0:
            model_kwargs["image_attention_mask"] = cross_mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            new_mask = cross_mask[:, -1:, :].repeat(1, mask_length_diff, 1)
            model_kwargs["image_attention_mask"] = torch.cat([cross_mask, new_mask], dim=1)

    return model_kwargs


def _prepare_position_ids(model_kwargs: dict[str, Any], new_length: int, is_encoder_decoder: bool) -> dict[str, Any]:
    """Expands or crops the model's position ids for decoding purposes, to the defined length"""

    position_key = "decoder_position_ids" if is_encoder_decoder else "position_ids"
    if model_kwargs.get(position_key) is None:
        return model_kwargs

    positions = model_kwargs[position_key]
    position_length_diff = new_length - positions.shape[-1]

    if position_length_diff < 0:
        model_kwargs[position_key] = positions[:, :position_length_diff]
    elif position_length_diff > 0:
        # Works for 2D and 3D position tensors
        required_dim = [1] * (positions.dim() - 1) + [-1]
        next_position_ids = (
            torch.arange(position_length_diff, dtype=positions.dtype, device=positions.device).view(*required_dim)
            + positions[..., -1:]
            + 1
        )
        next_position_ids = torch.cat([positions, next_position_ids], dim=-1)
        model_kwargs[position_key] = next_position_ids

    return model_kwargs


def _prepare_token_type_ids(model_kwargs: dict[str, Any], new_length: int) -> dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs

    token_type_ids = model_kwargs["token_type_ids"]
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    type_length_diff = new_length - token_type_ids.shape[1]

    if type_length_diff < 0:
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    return model_kwargs

def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
):
    """
    Applies sampling as in the speculative decoding paper (https://huggingface.co/papers/2211.17192, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches


def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs


def history_speculative_decoding(
        model: "GenerativePreTrainedModel",
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        inputs_tensor: torch.FloatTensor | None = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        assistant_tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:

            # The cache must be dynamic for assisted generation, and the check must happen AFTER preparing cache
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"] or (
            "past_key_values" in model_kwargs
            and hasattr(model_kwargs["past_key_values"], "layers")
            and any(getattr(l, "is_compileable", False) for l in model_kwargs["past_key_values"].layers)
        ):
            raise ValueError("assisted generate is not supported with Static cache classes`")
        # my candidate generator
        candidate_generator = HistoryLookupCandidateGenerator(input_ids, historydb=historydb)
        # init values
        do_sample = generation_config.do_sample
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        this_peer_finished = False
        is_first_iteration = True  # to preserve the same API in the output as other generation methods
        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            cur_len = input_ids.shape[1]

            #  1. Fetch candidate sequences from a `CandidateGenerator` and move to the correct device
            candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)
            candidate_input_ids = candidate_input_ids.to(model.device)
            if candidate_logits is not None:
                candidate_logits = candidate_logits.to(model.device)

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            is_done_candidate = stopping_criteria(candidate_input_ids, None)

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, candidate_input_ids.shape[1], model.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
            if (position_ids := candidate_kwargs.get("position_ids")) is not None and candidate_length > 0:
                new_length = candidate_length + position_ids.shape[-1]
                candidate_kwargs = _prepare_position_ids(candidate_kwargs, new_length, model.config.is_encoder_decoder)

            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(candidate_length, device=input_ids.device, dtype=torch.long) + cur_len,
                    ),
                    dim=0,
                )

            next_sequence_length = candidate_length + 1 if not is_first_iteration else None
            model_inputs = model.prepare_inputs_for_generation(
                candidate_input_ids,
                next_sequence_length=next_sequence_length,
                is_first_iteration=is_first_iteration,
                **candidate_kwargs,
            )

            if "logits_to_keep" in model_inputs:
                model_inputs["logits_to_keep"] = candidate_length + 1

            # 2.2. Run a forward pass on the candidate sequence

            outputs = model(**model_inputs)

            # 2.3. Process the new logits
            # .float() is needed to retain precision for later logits manipulations
            new_logits = outputs.logits[:, -candidate_length - 1 :].to(
                dtype=torch.float32, device=input_ids.device
            )  # excludes the input prompt if present
            next_token_logits = new_logits.clone()
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            # 3. Select the accepted tokens. There are two possible cases:
            # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
            # 👉 Apply algorithm 1 from the speculative decoding paper (https://huggingface.co/papers/2211.17192).
            if do_sample and candidate_logits is not None:
                valid_tokens, n_matches = _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate,
                )

            # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
            # original model logits with the candidate tokens. We can keep the candidate tokens until the first
            # mismatch, or until the max length is reached.
            else:
                if do_sample:
                    probs = new_logits.softmax(dim=-1)
                    selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                else:
                    selected_tokens = new_logits.argmax(dim=-1)

                candidate_new_tokens = candidate_input_ids[:, cur_len:]
                n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

                # Ensure we don't generate beyond max_len or an EOS token
                if is_done_candidate and n_matches == candidate_length:
                    n_matches -= 1
                valid_tokens = selected_tokens[:, : n_matches + 1]

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            if streamer is not None:
                streamer.put(valid_tokens.cpu())
            new_cur_len = input_ids.shape[1]

            # 4.2. Discard past key values relative to unused assistant tokens
            outputs.past_key_values.crop(new_cur_len - 1)

            # 5. Update the candidate generation strategy if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
                num_new_tokens=n_matches + 1,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                newly_added_length = n_matches + 1
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(newly_added_length))
                if output_logits:
                    raw_logits += tuple(next_token_logits[:, i, :] for i in range(newly_added_length))

                newly_added_length = new_cur_len if is_first_iteration else newly_added_length
                if output_attentions:
                    if model.config.is_encoder_decoder:
                        cross_attentions = _split_model_outputs(
                            cross_attentions, outputs.cross_attentions, cur_len, newly_added_length
                        )
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.decoder_attentions,
                            cur_len,
                            newly_added_length,
                            is_decoder_attention=True,
                        )
                    # some (V)LLMs have hard requirement on SDPA and thus never return attn
                    elif outputs.attentions[0] is not None:
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.attentions,
                            cur_len,
                            newly_added_length,
                            is_decoder_attention=True,
                        )
                if output_hidden_states:
                    if model.config.is_encoder_decoder:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.decoder_hidden_states, cur_len, newly_added_length
                        )
                    else:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.hidden_states, cur_len, newly_added_length
                        )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            is_first_iteration = False

        if streamer is not None:
            streamer.end()

        # if (
        #     isinstance(candidate_generator, AssistedCandidateGenerator)
        #     and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        # ):
        #     candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
        #         candidate_generator.num_assistant_tokens
        #     )
        if return_dict_in_generate:
            cache = None
            if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
                cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
                cache = model_kwargs[cache_key]
            if model.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=cache,
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=cache,
                )
        else:
            return input_ids
