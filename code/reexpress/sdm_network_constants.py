# Copyright Reexpress AI, Inc. All rights reserved.

from typing import Optional, Dict, List, Tuple


def get_conv(prompt_text):
    # Create the conversation
    conv = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt_text}]
    return conv


def create_masked_labels(mask_prefix, tokenizer, mask_until_pattern, input_ids: List[int], assistant_response: str,
                         disable_negative_masking=False) -> List[int]: #, attention_mask: List[int]) -> List[int]:
    """Create labels with selective masking based on the response structure.
    Note: Padding tokens are NOT masked here since we use dynamic padding.
    The collator will pad labels with -100.

    Note: The final negative sequence label is ALWAYS <verified>No</verified> in this version.
    Assumptions:
        1. <|assistant|> does not appear in assistant_response
        2. mask_until_pattern == "No</verified>"
        3. We assume a single turn conversation (i.e., one <|assistant|> response)

    disable_negative_masking is primarily for DPO-style losses and baselines. Setting to True will disable the
    intended masking convention, so use with caution.
    """
    labels = input_ids.copy()

    assert mask_until_pattern == "No</verified>"
    full_negative_sequence_suffix_label_pattern = "<verified>No</verified>"
    # override mask_until_pattern. The current convention is to treat the full suffix as an atomic unit.
    mask_until_pattern = full_negative_sequence_suffix_label_pattern

    if not mask_prefix:
        assert False, "Not expected in this version."
        # return labels

    # Decode the input_ids to find where the assistant response starts
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Find where the assistant response starts in the tokenized sequence
    # This is typically after the last occurrence of a role marker. We assume this does not otherwise appear in the
    # assistant text. It is important to use the full marker sequence, otherwise subsequent re-tokenization may be
    # mis-aligned. Check this behavior with each new tokenizer.
    assistant_marker = "<|assistant|>"
    assistant_start_idx = full_text.rfind(assistant_marker)
    if assistant_marker in assistant_response:
        print(f"WARNING: MASKING: The assistant tag ({assistant_marker}) occurs verbatim in the assistant text. The "
              f"parsing to create the Contrastive Mask may lead to unexpected results.")

    if assistant_start_idx == -1:  # this should not occur with Phi 3.5
        assert False, f"MASKING ERROR: The assistant marker was not found. " \
                      f"The input does not follow the expected chat template. The behavior will be unexpected. " \
                      f"input_ids: {input_ids}; assistant_response: {assistant_response}"
        # # If we can't find the assistant marker, mask everything except the response
        # # Try to find where the actual response content starts
        # response_start = full_text.find(assistant_response[:50])  # Use first 50 chars as anchor
        # if response_start != -1:
        #     # Convert character position to token position (approximate)
        #     prefix_text = full_text[:response_start]
        #     prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        #     mask_until_token_idx = len(prefix_tokens)
        # else:
        #     mask_until_token_idx = 0
    else:
        # Check if the response contains any "No" verifications. As input, the string is assumed to have been
        # re-formatted, so assigned negatives should always have "<verified>No</verified>" present
        if full_negative_sequence_suffix_label_pattern in assistant_response and not disable_negative_masking:
            # Find where the LAST "No</verified>" appears in the response
            last_no_pattern_idx = assistant_response.rfind(mask_until_pattern)

            if last_no_pattern_idx != -1:
                # We want to mask everything up to (but not including) the last mask_until_pattern
                mask_until_text = assistant_response[:last_no_pattern_idx]

                # Find this position in the full tokenized text
                # Note the rfind, as these patterns may occur in the prompt
                # (particularly if the assistant text is otherwise blank except for the control token).
                mask_pattern_position = full_text.rfind(mask_until_text + mask_until_pattern)
                if mask_pattern_position != -1:
                    # Find where the negative control sequence starts
                    no_start_position = mask_pattern_position + len(mask_until_text)
                    prefix_to_mask = full_text[:no_start_position]
                    prefix_tokens = tokenizer.encode(prefix_to_mask, add_special_tokens=False)
                    mask_until_token_idx = min(len(prefix_tokens), len(labels))
                else:
                    assert False, f"MASKING ERROR: Negative sequence pattern found in the assistant response, " \
                                  f"but not in the full text. " \
                                  f"The behavior will be unexpected. " \
                                  f"input_ids: {input_ids}; assistant_response: {assistant_response}"

                    # Fallback: mask the system and user messages only
                    # prefix_text = full_text[:assistant_start_idx]
                    # prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                    # mask_until_token_idx = len(prefix_tokens)
            else:
                assert False, f"MASKING ERROR: Unexpected behavior of index rfind. " \
                              f"The behavior will be unexpected. " \
                              f"input_ids: {input_ids}; assistant_response: {assistant_response}"
                # No pattern found, mask system and user messages
                # prefix_text = full_text[:assistant_start_idx]
                # prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                # mask_until_token_idx = len(prefix_tokens)
        else:
            # No "No" verifications (all answers are correct), only mask system and user messages
            prefix_text = full_text[:assistant_start_idx + len(assistant_marker)]
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            mask_until_token_idx = len(prefix_tokens)

    # Apply masking
    for i in range(min(mask_until_token_idx, len(labels))):
        labels[i] = -100

    return labels


def augment_prompt_with_hint(prompt_text, hint_response):
    hint_text = f"Hint: The answer is not '{hint_response}'."

    prompt_text = f"{prompt_text} {hint_text}"
    return prompt_text


REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_GENAI_DOCUMENT_KEY = "genai_document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

ORIGINAL_DOCUMENT_ORDER_KEY = "original_document"
MULTISET_KEY = "multiset"
GENERATED_RESPONSE_KEY = "generated_response"  # note that REEXPRESS_GENAI_DOCUMENT_KEY is used for ground-truth
# CORRECT_FIRST_HALF_KEY = "correctly_ordered_prefix"
CACHED_PROMPT_KEY = "prompt"
DEFAULT_NEGATIVE_KEY = "default_negative"

HARD_NEGATIVES_KEY = "hard_negatives"
SDM_NETWORK_FINETUNING_EPOCH_KEY = "epoch"

EXPECTED_EMBEDDING_SIZE = 6144
###
MAX_NEW_TOKENS_FOR_GENERATION = 256

IS_SDM_TRAINING_SPLIT_KEY = "is_sdm_training_split_key"  # 0, 1; or -1 if N/A

FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_FILE = "generations_all.jsonl"
FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_VALIDATION_FILE = "generations_validation_all.jsonl"

FILENAME_DISTRIBUTED_EMBEDDINGS_VALIDATION_PREFIX_FILE_NAME = "distributed_embeddings_validation"
FILENAME_DISTRIBUTED_EMBEDDINGS_TRAINING_PREFIX_FILE_NAME = "distributed_embeddings"

# This is not currently used in the present version, but provides a template example of how to add more complex,
# multi-sequence variations. In principle, multi-turn conversations and other variations follow, similarly:
kReHint = False