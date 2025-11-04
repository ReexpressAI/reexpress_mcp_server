# Copyright Reexpress AI, Inc. All rights reserved.

"""
Baseline utils for research on test-time search strategies.
"""

import logging
import torch
import warnings

import sdm_network_constants
import sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings
import sdm_network_finetune_utils_trainer_reward_assignment

# Suppress warnings
warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
warnings.filterwarnings("ignore", message=".*numerical differences.*")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_yes_probability_from_generation(model, response, inputs, tokenizer):
    """
    Extract the probability of "Yes" token after the final <verified> tag. This is inefficient, since we regenerate
    a single token, but we do this to eliminate the possibility of parsing errors conflating the results. This
    parallels the extraction of the hidden states, as used for the verification layer.

    Returns:
        (yes_probability, is_parseable): Tuple of float and int (0 or 1)
    """
    device = next(model.parameters()).device
    _, verification_response = \
        sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
            generated_text=response)
    force_decoded_ids = \
        tokenizer.encode(verification_response, add_special_tokens=False, return_tensors="pt").to(device)
    # We force-decode the generation up to the verification tag, and then extract the probability for "Yes".
    input_ids = torch.cat([inputs["input_ids"], force_decoded_ids], dim=1)
    # if "attention_mask" in inputs:
    #     sentence_tag_mask = torch.ones_like(force_decoded_ids)
    #     inputs["attention_mask"] = torch.cat([inputs["attention_mask"], sentence_tag_mask], dim=1)

    # Ensure input_ids is on the model's device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,  # Ensure batch dimension
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=False  # This avoids the caching/gradient checkpointing conflict
        )
        # hidden_states = outputs.hidden_states
        scores = outputs.scores
        # # Here, the starting no has a prefix underscore. Other models may differ. tokenizer.convert_ids_to_tokens([1939])
        # no_id = tokenizer.vocab['▁No']
        # yes_id = tokenizer.vocab['▁Yes']  # tokenizer.convert_ids_to_tokens([3869])
        # In this case, there is no prefix underscore, since there is no space after the closing XML tag.
        # no_id = tokenizer.vocab['No']
        yes_id = tokenizer.vocab['Yes']
        probs = torch.softmax(scores[0], dim=-1)
        # For reference, this reproduces the final output as input to the softmax:
        # model.lm_head(hidden_states[0][-1][0][-1, :])
        return probs[0:1, yes_id].item()


def generate_response_single_with_llm_logit_scoring(
        model,
        tokenizer,
        prompt_text: str,
        example,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        device: str = "cuda",
        use_file_prompt: bool = True,
        print_output: bool = False,
        args=None
):
    """Generate a response and extract Yes probability from LLM logits."""
    if use_file_prompt:
        conv = sdm_network_constants.get_conv(prompt_text)
    else:
        assert False

    processed_prompt_text = tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        processed_prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2047,
        padding=False
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    if args is not None and args.begin_with_sentence_tag:
        sentence_tag_ids = tokenizer.encode("<sentence>", add_special_tokens=False, return_tensors="pt").to(device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], sentence_tag_ids], dim=1)
        if "attention_mask" in inputs:
            sentence_tag_mask = torch.ones_like(sentence_tag_ids)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], sentence_tag_mask], dim=1)

    # Generate with scores
    with torch.no_grad():
        if do_sample:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

    # Decode only the generated portion
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs.sequences[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if args is not None and args.begin_with_sentence_tag:
        response = "<sentence>" + response

    # Extract label for evaluation
    sentences, verifications = \
        sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.extract_sentences_from_generation(
            generated_text=response)
    if len(sentences) > 0:
        generated_sentence = sentences[0]
    else:
        generated_sentence = ""

    reference_sentence = example.get(sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY, "")
    binary_reward_label = \
        sdm_network_finetune_utils_trainer_reward_assignment.assign_binary_reward_label(
            reference_document_string=reference_sentence,
            generated_sentence=generated_sentence)  # ground-truth, so not used as search metric

    # Extract Yes probability from logits
    yes_prob = extract_yes_probability_from_generation(model, response, inputs, tokenizer)
    is_parseable = int(len(sentences) > 0)

    if binary_reward_label == 0 or print_output:
        print("-------------------")
        print(f"binary_reward_label: {'WRONG' if binary_reward_label == 0 else 'CORRECT'}")
        print(f"Prompt: {prompt_text}")
        print(f"Response: {response}")
        print(f"Yes probability: {yes_prob:.4f}, Parseable: {is_parseable}")

    embedding = None
    return response, embedding, binary_reward_label, yes_prob, is_parseable


def generate_response_single_with_llm_logit_scoring_conditional_branch(
        model,
        tokenizer,
        document: str,
        example,
        search_depth: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        device: str = "cuda",
        use_file_prompt: bool = True,
        args=None
):
    """
    Generate responses with search based on LLM logit scoring.
    Continues searching while Yes probability < min_stopping_probability.
    """
    response_candidates = []
    search_depth_counter = 0
    print_output = False

    min_stopping_prob = args.min_stopping_probability if args is not None else 0.95

    while True:
        response, embedding, label, yes_prob, is_parseable = \
            generate_response_single_with_llm_logit_scoring(
                model,
                tokenizer,
                document,
                example=example,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                device=device,
                use_file_prompt=use_file_prompt,
                print_output=print_output,
                args=args
            )

        response_candidates.append((yes_prob, response, embedding, label))

        search_budget_remaining = search_depth_counter < search_depth
        if search_budget_remaining and (yes_prob < min_stopping_prob):
            search_depth_counter += 1
            do_sample = True
            print(f"sampling, depth: {search_depth_counter}, yes_prob: {yes_prob:.4f}, parseable: {is_parseable}")
        else:
            break

    # Sort by Yes probability
    sorted_candidates = sorted(response_candidates, key=lambda x: x[0], reverse=True)
    best_candidate = sorted_candidates[0]
    softmax_probability_class1, response, embedding, label = \
        best_candidate[0], best_candidate[1], best_candidate[2], best_candidate[3]
    return response, embedding, label
