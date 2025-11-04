# Copyright Reexpress AI, Inc. All rights reserved.

"""
Multi-GPU inference script for fine-tuned Phi-3.5-mini-instruct model using accelerate.
Distributes document processing across multiple GPUs and consolidates results on rank 0.
"""

import argparse
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple

import warnings
import os

from accelerate import Accelerator
import torch.distributed as dist

import sdm_network_constants
import utils_model
import sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings
import constants
import sdm_network_finetune_utils_trainer_reward_assignment
import sdm_network_inference_with_verification_utils_baseline

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


def generate_response_single_with_verification(
        model,
        tokenizer,
        prompt_text: str,
        example,
        search_depth: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        device: str = "cuda",
        use_file_prompt: bool = True,
        verification_layer=None,
        print_output=False,
        args=None
):
    """Generate a response for a single document (no batching to avoid issues)."""
    if use_file_prompt:
        conv = sdm_network_constants.get_conv(prompt_text)
    else:
        assert False

    # Apply chat template without tokenizing first
    processed_prompt_text = tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize with proper truncation
    inputs = tokenizer(
        processed_prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2047,  # Leave room for generation prompt
        padding=False  # No padding for single example
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Optionally prepend "<sentence>" to force the model to start with the tag
    if args is not None and args.begin_with_sentence_tag:
        sentence_tag_ids = tokenizer.encode("<sentence>", add_special_tokens=False, return_tensors="pt").to(device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], sentence_tag_ids], dim=1)
        if "attention_mask" in inputs:
            sentence_tag_mask = torch.ones_like(sentence_tag_ids)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], sentence_tag_mask], dim=1)

    # Generate
    if do_sample:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Decode only the generated portion
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # If we prepended the sentence tag, include it in the response
    if args is not None and args.begin_with_sentence_tag:
        response = "<sentence>" + response

    # verify response
    # First, extract and reformat response
    sentences, verifications = \
        sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.extract_sentences_from_generation(
            generated_text=response)
    if len(sentences) > 0 and len(verifications) > 0:
        generated_sentence = sentences[0]
    else:
        generated_sentence = ""
    embedding = None

    # response_with_constructed_no_tag is only used if a re-hint is subsequently needed (which is not implemented here)
    response_with_constructed_no_tag, deterministically_parsed_verification_response = \
        sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
            sentences=[generated_sentence], verifications=[False])
    reference_sentence = example.get(sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY, "")
    # binary_reward_label is the ground-truth label. This is not used for conditional branching.
    binary_reward_label = \
        sdm_network_finetune_utils_trainer_reward_assignment.assign_binary_reward_label(
            reference_document_string=reference_sentence,
            generated_sentence=generated_sentence)

    if verification_layer is not None:
        # The verification layer gets the full, unparsed input, with the lexical classification label removed.
        _, verification_response = \
            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                generated_text=response)
        verification_input_ids, _ = \
            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_ids_from_prompt_text_and_assistant_response(
                tokenizer, prompt_text, verification_response)
        embedding = \
            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_verification_embedding(
                model=model, input_ids=verification_input_ids)
        prediction_meta_data = verification_layer(embedding.to(device),
                                                  forward_type=constants.FORWARD_TYPE_SINGLE_PASS_TEST)
    else:  # Placeholder
        # response_with_constructed_no_tag is only used if a re-hint is subsequently needed
        # response_with_constructed_no_tag, verification_response = \
        #     sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
        #         sentences=[generated_sentence], verifications=[False])
        response_with_constructed_no_tag, verification_response = \
            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                generated_text=response)
        prediction_meta_data = None

    if binary_reward_label == 0 or print_output:
        print("-------------------")
        print(f"binary_reward_label: {'WRONG' if binary_reward_label == 0 else 'CORRECT'}")
        print(f"Prompt: {prompt_text}")
        print(f"Response: {response}")
        print(f"Verification Response: {verification_response}")
        if prediction_meta_data is not None:
            print(f"prediction_meta_data: {prediction_meta_data}")

    return response, embedding, binary_reward_label, prediction_meta_data, response_with_constructed_no_tag


def generate_response_single_with_verification_conditional_branch(
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
        verification_layer=None,
        args=None
):
    response_candidates = []
    search_depth_counter = 0
    print_output = False

    while True:
        response, embedding, label, prediction_meta_data, response_with_constructed_no_tag = \
            generate_response_single_with_verification(
                model,
                tokenizer,
                document,
                example=example,
                search_depth=search_depth,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                device=device,
                use_file_prompt=use_file_prompt,
                verification_layer=verification_layer,
                print_output=print_output,
                args=args
            )
        predicted_class = prediction_meta_data["prediction"]

        prediction_conditional_distribution__sdm = \
            prediction_meta_data["sdm_output"]

        # prediction_conditional_estimate_of_predicted_class__sdm = \
        #     prediction_conditional_distribution__sdm[predicted_class].item()
        verified_prediction_probability = prediction_conditional_distribution__sdm[1].item()
        is_high_reliability_region = prediction_meta_data["is_high_reliability_region"]
        rescaled_similarity = prediction_meta_data["rescaled_similarity"]

        response_candidates.append((predicted_class, int(is_high_reliability_region), verified_prediction_probability,
                                    rescaled_similarity, response, embedding, label))
        # if verified_prediction_probability < 0.95 and search_depth_counter < search_depth:
        search_budget_remaining = search_depth_counter < search_depth
        if search_budget_remaining and (predicted_class != 1 or not is_high_reliability_region):
            search_depth_counter += 1
            do_sample = True
            # print_output = True
            print(f"sampling, depth: {search_depth_counter}")
        else:
            break

    # sort by predicted class (prefer class 1 over class 0), and then prefer documents in the high reliability region,
    # and then the predicted probability for class 1, and finally the rescaled similarity to break ties in the
    # case of saturated probabilities:
    sorted_candidates = sorted(response_candidates, key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best_candidate = sorted_candidates[0]
    response, embedding, label = best_candidate[4], best_candidate[5], best_candidate[6]

    return response, embedding, label


def generate_response_single_without_search(
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
        # verification_layer=None,
        args=None
):

    print_output = False

    response, _, label, _, _ = \
        generate_response_single_with_verification(
            model,
            tokenizer,
            document,
            example=example,
            search_depth=search_depth,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
            use_file_prompt=use_file_prompt,
            verification_layer=None,
            print_output=print_output,
            args=args
        )
    embedding = None
    return response, embedding, label


def process_documents_subset(
        model,
        tokenizer,
        documents: List[str],
        examples: List[Dict],
        document_indices: List[int],
        search_depth: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        device: str = "cuda",
        use_file_prompt: bool = True,
        args=None,
        verification_layer=None,
        use_llm_logit_scoring=False,
        rank: int = 0
):
    """Process a subset of documents assigned to this rank."""
    
    responses = []
    
    if do_sample:
        print(f"Rank {rank}: Generating with: Sampling: {do_sample}, top_p: {top_p}, "
              f"temperature: {temperature}; Max tokens: {max_new_tokens}")
    else:
        print(f"Rank {rank}: Generating with greedy decoding; Max tokens: {max_new_tokens}")
    
    print(f"Rank {rank}: Processing {len(documents)} documents")
    
    for idx, (doc, example, doc_idx) in enumerate(zip(documents, examples, document_indices)):
        print(f"Rank {rank}: Processing document {idx+1}/{len(documents)} (global index: {doc_idx})")
        try:
            if use_llm_logit_scoring:
                response = \
                    sdm_network_inference_with_verification_utils_baseline.generate_response_single_with_llm_logit_scoring_conditional_branch(
                        model,
                        tokenizer,
                        doc,
                        example=example,
                        search_depth=search_depth,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        device=device,
                        use_file_prompt=use_file_prompt,
                        args=args
                    )
            elif verification_layer is not None:
                response = generate_response_single_with_verification_conditional_branch(
                    model,
                    tokenizer,
                    doc,
                    example=example,
                    search_depth=search_depth,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    device=device,
                    use_file_prompt=use_file_prompt,
                    verification_layer=verification_layer,
                    args=args
                )
            else:
                response = generate_response_single_without_search(
                    model,
                    tokenizer,
                    doc,
                    example=example,
                    search_depth=search_depth,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    device=device,
                    use_file_prompt=use_file_prompt,
                    args=args
                )
            responses.append((doc_idx, response))  # Store with original index
            
            if args is not None and args.print_each:
                print(f"Rank {rank}: >>>>>")
                print("---Document:")
                print(doc)
                print("---Response:")
                print(response[0])
                print("<<<<<")
        except Exception as e:
            logger.warning(f"Rank {rank}: Error generating response for doc {doc_idx}: {e}")
            responses.append((doc_idx, ("", None, None)))  # Add empty response with index

    return responses


def split_documents_for_ranks(documents: List[str], examples: List[Dict], 
                              world_size: int, rank: int) -> Tuple[List[str], List[Dict], List[int]]:
    """Split documents across ranks, maintaining original indices."""
    total_docs = len(documents)
    docs_per_rank = (total_docs + world_size - 1) // world_size  # Ceiling division
    
    start_idx = rank * docs_per_rank
    end_idx = min(start_idx + docs_per_rank, total_docs)
    
    rank_documents = documents[start_idx:end_idx]
    rank_examples = examples[start_idx:end_idx]
    rank_indices = list(range(start_idx, end_idx))
    
    return rank_documents, rank_examples, rank_indices


def save_rank_results(results: List[Tuple[int, Tuple]], output_dir: str, rank: int):
    """Save results from a single rank with original indices."""
    rank_output_file = os.path.join(output_dir, f"rank_{rank}_results.jsonl")
    
    with open(rank_output_file, 'w') as f:
        for doc_idx, (response, embedding, label) in results:
            result_data = {
                sdm_network_constants.REEXPRESS_ID_KEY: doc_idx,
                sdm_network_constants.GENERATED_RESPONSE_KEY: response,
                sdm_network_constants.REEXPRESS_EMBEDDING_KEY:
                    [float(x) for x in embedding[0].cpu().numpy().tolist()] if embedding is not None else None,
                sdm_network_constants.REEXPRESS_LABEL_KEY: label
            }
            f.write(json.dumps(result_data) + '\n')
    
    print(f"Rank {rank}: Saved {len(results)} results to {rank_output_file}")


def consolidate_results(output_dir: str, world_size: int, examples: List[Dict], 
                       output_file: str, save_embedding: bool):
    """Consolidate results from all ranks in original order."""
    print(f"Rank 0: Consolidating results from {world_size} ranks")
    
    # Collect all results
    all_results = {}
    
    for rank in range(world_size):
        rank_output_file = os.path.join(output_dir, f"rank_{rank}_results.jsonl")
        
        if not os.path.exists(rank_output_file):
            logger.warning(f"Missing results file for rank {rank}")
            continue
        
        with open(rank_output_file, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    doc_idx = result[sdm_network_constants.REEXPRESS_ID_KEY]
                    all_results[doc_idx] = (
                        result[sdm_network_constants.GENERATED_RESPONSE_KEY],
                        result.get(sdm_network_constants.REEXPRESS_EMBEDDING_KEY, None),
                        result[sdm_network_constants.REEXPRESS_LABEL_KEY]
                    )
    
    # Write results in original order
    print(f"Rank 0: Writing {len(all_results)} results to {output_file}")
    
    with open(output_file, 'w') as f:
        for idx, example in enumerate(examples):
            if idx in all_results:
                response, embedding, label = all_results[idx]
                example[sdm_network_constants.GENERATED_RESPONSE_KEY] = response
                
                if save_embedding and embedding is not None:
                    example[sdm_network_constants.REEXPRESS_EMBEDDING_KEY] = embedding
                
                example[sdm_network_constants.REEXPRESS_LABEL_KEY] = label
                
                # Simple reformatting to make viewing the output easy in the existing downstream analysis scripts
                example[sdm_network_constants.REEXPRESS_DOCUMENT_KEY] = \
                    f"<reference> {example[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]} </reference> " \
                    f"<generated> {response} </generated>"
            else:
                logger.warning(f"Missing result for document index {idx}")
                example[sdm_network_constants.GENERATED_RESPONSE_KEY] = ""
                example[sdm_network_constants.REEXPRESS_LABEL_KEY] = None
            
            f.write(json.dumps(example) + '\n')
    
    # Cleanup rank files
    for rank in range(world_size):
        rank_output_file = os.path.join(output_dir, f"rank_{rank}_results.jsonl")
        if os.path.exists(rank_output_file):
            os.remove(rank_output_file)
    
    print(f"Rank 0: Consolidation complete. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference with fine-tuned Phi-3.5 model")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--verification_model_dir", type=str,
                        help="Model directory for the verification model")
    parser.add_argument("--no_verification_model", action="store_true", default=False,
                        help="In this case, no verification layer is expected or used.")

    # Input/output arguments
    parser.add_argument("--input_file", type=str, help="Input JSONL file with documents")
    parser.add_argument("--output_file", type=str, help="Output JSONL file for results")
    parser.add_argument("--save_embedding", action="store_true", default=False,
                        help="Save the verification embedding for each final generation. "
                             "These can then be used with the standard eval scripts via reexpress.py.")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling")

    # Precision arguments
    parser.add_argument("--fp16", action="store_true", default=False, help="Use fp16 precision")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bf16 precision")
    parser.add_argument("--use_file_prompt", action="store_true", default=False,
                        help="Use the prompt cached in the input file")
    parser.add_argument("--print_each", action="store_true", default=False,
                        help="Print each output to standard out")

    # Search arguments
    parser.add_argument("--search_depth", type=int, default=1, help="Search depth for verification")

    # Search arguments for use with logits baseline:
    parser.add_argument("--min_stopping_probability", default=0.95, type=float, help="")
    parser.add_argument("--use_llm_logit_scoring", action="store_true", default=False,
                        help="Use LLM logit scoring baseline instead of verification layer")

    # Debugging/Analysis arguments
    parser.add_argument("--line_limit", type=int, default=-1, help="Only consider this many lines")
    
    # Format control arguments
    parser.add_argument("--begin_with_sentence_tag", action="store_true", default=False,
                        help="Force generation to begin with '<sentence>' tag")

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    assert len(args.output_file) > 0 and args.output_file.strip() != args.input_file.strip()

    if accelerator.is_main_process:
        print(f"Arguments: {args}")
        if args.line_limit > -1:
            print(f"WARNING: only {args.line_limit} lines will be considered due to --line_limit")
        print(f"Running on {world_size} GPUs")
        if args.begin_with_sentence_tag:
            print(f"Force decoding each generation with the prefix '<sentence>'.")
        if args.do_sample:
            print(f"The initial generation and all subsequent search generations (if applicable) will use sampling "
                  f"with top_p={args.top_p} and temperature={args.temperature}.")
        else:
            print(f"The initial generation will use greedy decoding "
                  f"and all subsequent search generations (if applicable) will use sampling "
                  f"with top_p={args.top_p} and temperature={args.temperature}.")
        if args.search_depth > 0:
            print(f"\tThe search depth is {args.search_depth}")
        else:
            print(f"\tThe search depth is 0, so only a single document will be generated for each prompt.")

    if not args.no_verification_model:
        assert args.verification_model_dir.strip() != ""

    if args.use_llm_logit_scoring:
        assert args.no_verification_model
        print(f"Scoring generations based on the output logits for the Yes token.")

    # Load tokenizer (all ranks)
    logger.info(f"Rank {rank}: Loading tokenizer from {args.model_path}")
    
    if args.model_path == "microsoft/Phi-3.5-mini-instruct":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=False,
            padding_side="left"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    # Load model (all ranks)
    logger.info(f"Rank {rank}: Loading model from {args.model_path}")
    
    if args.model_path == "microsoft/Phi-3.5-mini-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation="eager",
            use_cache=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation="eager",
            use_cache=True
        )

    # Prepare model with accelerator
    model = accelerator.prepare(model)
    model.eval()

    logger.info(f"Rank {rank}: Model loaded on device {device}")

    if args.no_verification_model or args.use_llm_logit_scoring:
        verification_layer = None
    else:
        # Load verification layer (all ranks need this)
        verification_layer = utils_model.load_model_torch(
            args.verification_model_dir,
            device,
            load_for_inference=True
        )
        print(f"Rank {rank}: Verification layer loaded on device {device}")

    # Load documents (all ranks load full dataset to maintain indices)
    if accelerator.is_main_process:
        logger.info(f"Loading documents from {args.input_file}")

    documents = []
    examples = []

    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                if args.line_limit > -1 and len(documents) >= args.line_limit:
                    break
                example = json.loads(line)
                examples.append(example)
                if args.use_file_prompt:
                    documents.append(example.get(sdm_network_constants.CACHED_PROMPT_KEY, ""))
                else:
                    documents.append(example.get(sdm_network_constants.MULTISET_KEY, ""))

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(documents)} documents total")

    # Split documents across ranks
    rank_documents, rank_examples, rank_indices = split_documents_for_ranks(
        documents, examples, world_size, rank
    )
    
    print(f"Rank {rank}: Assigned {len(rank_documents)} documents "
          f"(indices {rank_indices[0] if rank_indices else 'N/A'} to "
          f"{rank_indices[-1] if rank_indices else 'N/A'})")

    # Create temporary directory for intermediate results
    temp_output_dir = os.path.dirname(args.output_file)
    temp_output_dir = os.path.join(temp_output_dir, f"temp_inference_results")
    
    if accelerator.is_main_process:
        os.makedirs(temp_output_dir, exist_ok=True)
        print(f"Created temporary directory at {temp_output_dir}")

    accelerator.wait_for_everyone()

    # Unwrap model for generation
    unwrapped_model = accelerator.unwrap_model(model)

    # Process documents on this rank
    rank_results = process_documents_subset(
        unwrapped_model,
        tokenizer,
        rank_documents,
        rank_examples,
        rank_indices,
        search_depth=args.search_depth,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=str(device),
        use_file_prompt=args.use_file_prompt,
        args=args,
        verification_layer=verification_layer,
        use_llm_logit_scoring=args.use_llm_logit_scoring,
        rank=rank
    )

    # Save rank results
    save_rank_results(rank_results, temp_output_dir, rank)
    
    # Wait for all ranks to finish
    accelerator.wait_for_everyone()

    # Consolidate results on rank 0
    if accelerator.is_main_process:
        if args.output_file:
            consolidate_results(
                temp_output_dir,
                world_size,
                examples,
                args.output_file,
                args.save_embedding
            )
            
            # Remove temp directory
            try:
                os.rmdir(temp_output_dir)
                print(f"Removed temporary directory at {temp_output_dir}")
            except:
                pass
            
            logger.info("Done!")
        else:
            # Print to stdout (only on rank 0)
            all_results = {}
            for rank_id in range(world_size):
                rank_output_file = os.path.join(temp_output_dir, f"rank_{rank_id}_results.jsonl")
                with open(rank_output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            all_results[result['doc_idx']] = result['response']
            
            for idx, example in enumerate(examples):
                if idx in all_results:
                    response = all_results[idx]
                    print(f"\nMultiset: {example.get(sdm_network_constants.MULTISET_KEY, '')}")
                    print(f"Generated: {response}")
                    if "Document" in example:
                        print(f"Reference: {example[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]}")
                    print("-" * 80)

    accelerator.wait_for_everyone()

    # Destroy process group now that distributed work is done, in preparation for clean exit
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"Rank {rank}: Destroyed process group. Exiting")


if __name__ == "__main__":
    main()
