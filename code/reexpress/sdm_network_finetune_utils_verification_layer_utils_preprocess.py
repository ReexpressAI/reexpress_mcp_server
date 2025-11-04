# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import random
import json
import os

import sdm_network_constants
import sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings

import utils_model
import constants


def determine_verification_layer_training_split(dataset, eval_dataset=None, epoch_seed=None):
    # In-place update of the field to determine whether a document is, or is not, in the training set of the
    # verification layer.
    # Create deterministic RNG for split assignment to maintain consistent splits across devices/ranks
    if epoch_seed is not None:
        split_rng = random.Random(epoch_seed)
    else:
        split_rng = random.Random(0)  # Fallback seed

    for i, example in enumerate(dataset.examples):
        # Use deterministic RNG for split assignment
        if split_rng.random() < 0.5:
            example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] = 1
        else:
            example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] = 0
    # The eval dataset (i.e., the --eval_file argument to sdm_network_finetune.py)
    # is used to train an independent SDM verification layer. Thus, the indicator is relative to that layer.
    # After LLM training is complete, the final SDM network uses this held-out data to construct the final
    # layer used at inference using the best LLM checkpoint.
    # (The 'eval' dataset here is not a final held-out test set. It is a validation set.)
    if eval_dataset is not None:
        for i, example in enumerate(eval_dataset.examples):
            if split_rng.random() < 0.5:
                example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] = 1
            else:
                example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] = 0


def split_dataset_for_distributed_processing(dataset, world_size, rank):
    """
    Split dataset examples for distributed processing.
    
    Args:
        dataset: Dataset with .examples attribute
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        List of examples for this rank to process
    """
    total_examples = len(dataset.examples)
    examples_per_rank = total_examples // world_size
    remainder = total_examples % world_size
    
    # Distribute remainder across first few ranks
    if rank < remainder:
        start_idx = rank * (examples_per_rank + 1)
        end_idx = start_idx + examples_per_rank + 1
    else:
        start_idx = rank * examples_per_rank + remainder
        end_idx = start_idx + examples_per_rank
    
    print(f"Rank {rank}: Processing examples {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
    return dataset.examples[start_idx:end_idx]


def process_distributed_subset(examples_subset, model, tokenizer, all_generations, epoch_seed=None,
                               ignore_default_negative_if_generated_hard_negative_is_available=False):
    """
    Process a subset of examples for distributed embedding generation.
    Always returns embeddings without calculating the normalization statistics, which we defer to
    the consolidation step once all embeddings are available.
    
    Args:
        examples_subset: List of example dicts to process
        model: The LLM model
        tokenizer: The tokenizer
        all_generations: Dict of cached generations
        epoch_seed: Seed for RNG

        ignore_default_negative_if_generated_hard_negative_is_available is primarily intended for training of the
            final verification layer after the initial training procedure has completed and is currently not an
            option in the corresponding sampling in compute_loss.
        
    Returns:
        Tuple of (train_uuids, train_metadata, train_embeddings, train_labels,
                  calibration_uuids, calibration_metadata, calibration_embeddings, calibration_labels)
    """
    # This is designed to be a different seed than in determine_verification_layer_training_split().
    if epoch_seed is not None:
        split_rng = random.Random(epoch_seed + 7)
    else:
        split_rng = random.Random(7)  # Fallback seed

    train_uuids = []
    train_metadata = []
    train_embeddings = []
    train_labels = []
    calibration_uuids = []
    calibration_metadata = []
    calibration_embeddings = []
    calibration_labels = []

    for i, example in enumerate(examples_subset):
        document_id = example[
            sdm_network_constants.REEXPRESS_ID_KEY]  # if isinstance(document_ids, list) else document_ids[i].item()
        default_negative = example[sdm_network_constants.DEFAULT_NEGATIVE_KEY]
        reference_document_string = example[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]
        prompt_text = example[sdm_network_constants.CACHED_PROMPT_KEY]

        if example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] == 1:
            train_uuids.append(document_id)
        elif example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] == 0:
            calibration_uuids.append(document_id)
        else:
            assert False

        # Next, construct the text of the example, either as a ground-truth example, or as a hard-negative:
        # This mirrors the approach in compute_loss

        # half the time, always correct:
        if split_rng.random() < 0.5:
            verification_label = 1
            _, verification_response = \
                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                    sentences=[reference_document_string], verifications=[True])
            input_ids, _ = \
                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_ids_from_prompt_text_and_assistant_response(
                    tokenizer, prompt_text, verification_response)
            embedding = \
                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_verification_embedding(
                    model, input_ids)
            if example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] == 1:
                train_metadata.append(example)
                train_embeddings.append(embedding)
                train_labels.append(verification_label)
            else:
                calibration_metadata.append(example)
                calibration_embeddings.append(embedding)
                calibration_labels.append(verification_label)
        else:
            # This example will contain an error in the response text
            error_candidates = [default_negative]
            # Unlike the counterpart in compute_loss, here we do not generate, but use the cached examples. The
            # dataset stays balanced, because we always have the default negative.
            if document_id in all_generations:
                # Load existing generations, if any
                saved_data = all_generations[document_id]
                available_generated_hard_negatives = saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, [])
                if ignore_default_negative_if_generated_hard_negative_is_available and \
                        len(available_generated_hard_negatives) > 0:
                    # In this case, we remove the default negative from consideration.
                    error_candidates = available_generated_hard_negatives
                else:
                    error_candidates.extend(available_generated_hard_negatives)

            # shuffle error candidates in place
            split_rng.shuffle(error_candidates)
            # The error_candidates text is formatted with the full assistant response including tags.
            # We need to reformat for verification.
            _, verification_response = \
                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                    generated_text=error_candidates[0])  # take the first, after shuffling
            assistant_text = verification_response
            verification_label = 0
            input_ids, _ = \
                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_ids_from_prompt_text_and_assistant_response(
                    tokenizer, prompt_text, assistant_text)
            embedding = \
                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_verification_embedding(
                    model, input_ids)
            if example[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] == 1:
                train_metadata.append(example)
                train_embeddings.append(embedding)
                train_labels.append(verification_label)
            else:
                calibration_metadata.append(example)
                calibration_embeddings.append(embedding)
                calibration_labels.append(verification_label)

    # Concatenate embeddings if any exist
    train_embeddings = torch.cat(train_embeddings, dim=0) if train_embeddings else torch.tensor([])
    calibration_embeddings = torch.cat(calibration_embeddings, dim=0) if calibration_embeddings else torch.tensor([])

    return (train_uuids, train_metadata, train_embeddings, torch.tensor(train_labels),
            calibration_uuids, calibration_metadata, calibration_embeddings, torch.tensor(calibration_labels))


def save_distributed_embeddings(save_dir, rank, epoch_or_step, is_training,
                                train_uuids, train_metadata, train_embeddings, train_labels,
                                calibration_uuids, calibration_metadata, calibration_embeddings, calibration_labels):
    """
    Save embeddings and metadata from a single rank to disk.
    
    Args:
        save_dir: Directory to save files
        rank: Current process rank
        epoch_or_step: Epoch number (training) or global step (eval)
        is_training: Whether this is for training or eval set
        train_uuids: List of training UUIDs
        train_metadata: List of training metadata dicts
        train_embeddings: Tensor of training embeddings
        train_labels: Tensor of training labels
        calibration_uuids: List of calibration UUIDs
        calibration_metadata: List of calibration metadata dicts
        calibration_embeddings: Tensor of calibration embeddings
        calibration_labels: Tensor of calibration labels
    """
    os.makedirs(save_dir, exist_ok=True)
    
    prefix = "train" if is_training else "eval"
    rank_file = os.path.join(save_dir, f"{prefix}_embeddings_rank_{rank}_epoch_{epoch_or_step}.pt")
    
    torch.save({
        'train_uuids': train_uuids,
        'train_metadata': train_metadata,
        'train_embeddings': train_embeddings,
        'train_labels': train_labels,
        'calibration_uuids': calibration_uuids,
        'calibration_metadata': calibration_metadata,
        'calibration_embeddings': calibration_embeddings,
        'calibration_labels': calibration_labels,
    }, rank_file)
    
    print(f"Rank {rank}: Saved {len(train_uuids)} train + {len(calibration_uuids)} calibration examples to {rank_file}")


def load_and_consolidate_distributed_embeddings(save_dir, world_size, epoch_or_step, is_training,
                                                do_not_normalize_input_embeddings=False):
    """
    Load and consolidate embeddings from all ranks, computing summary stats on rank 0.
    
    Args:
        save_dir: Directory containing saved files
        world_size: Total number of processes
        epoch_or_step: Epoch number (training) or global step (eval)
        is_training: Whether this is for training or eval set
        do_not_normalize_input_embeddings: Whether to skip normalization
        
    Returns:
        Tuple of consolidated data structures including summary_stats
    """
    all_train_uuids = []
    all_train_metadata = []
    all_train_embeddings = []
    all_train_labels = []
    all_calibration_uuids = []
    all_calibration_metadata = []
    all_calibration_embeddings = []
    all_calibration_labels = []
    
    prefix = "train" if is_training else "eval"
    
    for rank in range(world_size):
        rank_file = os.path.join(save_dir, f"{prefix}_embeddings_rank_{rank}_epoch_{epoch_or_step}.pt")
        
        if os.path.exists(rank_file):
            data = torch.load(rank_file, map_location='cpu', weights_only=True)
            
            all_train_uuids.extend(data['train_uuids'])
            all_train_metadata.extend(data['train_metadata'])
            if data['train_embeddings'].numel() > 0:  # Check if not empty
                all_train_embeddings.append(data['train_embeddings'])
            all_train_labels.append(data['train_labels'])
            
            all_calibration_uuids.extend(data['calibration_uuids'])
            all_calibration_metadata.extend(data['calibration_metadata'])
            if data['calibration_embeddings'].numel() > 0:  # Check if not empty
                all_calibration_embeddings.append(data['calibration_embeddings'])
            all_calibration_labels.append(data['calibration_labels'])
        else:
            print(f"Warning: Missing rank file {rank_file}")
    
    # Concatenate tensors
    train_embeddings = torch.cat(all_train_embeddings, dim=0) if all_train_embeddings else torch.tensor([])
    train_labels = torch.cat(all_train_labels, dim=0)
    calibration_embeddings = torch.cat(all_calibration_embeddings, dim=0) if all_calibration_embeddings else torch.tensor([])
    calibration_labels = torch.cat(all_calibration_labels, dim=0)
    
    print(f"Consolidated {len(all_train_uuids)} train + {len(all_calibration_uuids)} calibration examples from {world_size} ranks")
    
    # Compute summary stats now on rank 0
    if do_not_normalize_input_embeddings:
        training_embedding_summary_stats = {
            constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: 0.0,
            constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: 1.0
        }
    else:
        training_embedding_summary_stats = utils_model.get_embedding_summary_stats(train_embeddings, is_training=True)
    
    return (all_train_uuids, all_train_metadata, train_embeddings, train_labels,
            all_calibration_uuids, all_calibration_metadata, calibration_embeddings, calibration_labels,
            training_embedding_summary_stats)


def cleanup_distributed_embedding_files(save_dir, world_size, epoch_or_step, is_training):
    """
    Remove intermediate embedding files after consolidation.
    
    Args:
        save_dir: Directory containing saved files
        world_size: Total number of processes
        epoch_or_step: Epoch number (training) or global step (eval)
        is_training: Whether this is for training or eval set
    """
    prefix = "train" if is_training else "eval"
    
    for rank in range(world_size):
        rank_file = os.path.join(save_dir, f"{prefix}_embeddings_rank_{rank}_epoch_{epoch_or_step}.pt")
        if os.path.exists(rank_file):
            os.remove(rank_file)
    
    print(f"Cleaned up {world_size} intermediate embedding files")


class MockDataset:
    """
    Mock dataset class that mimics DocumentOrderingDataset structure
    for standalone verification layer training.
    """

    def __init__(self, examples):
        """
        Args:
            examples: List of dictionaries with required keys for verification training
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _construct_dictionary_from_json(data):
    # Map from JSONL format to the expected internal format. No defaults. All fields must be present
    example = {
        sdm_network_constants.REEXPRESS_ID_KEY: data[sdm_network_constants.REEXPRESS_ID_KEY],
        sdm_network_constants.CACHED_PROMPT_KEY: data[sdm_network_constants.CACHED_PROMPT_KEY],
        sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY:
            data[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY],
        sdm_network_constants.DEFAULT_NEGATIVE_KEY: data[sdm_network_constants.DEFAULT_NEGATIVE_KEY],
        # This will be set during processing
        sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY: -1,
    }

    # # Add any additional fields from the JSONL that might be needed
    # # (preserve them in case they're used elsewhere)
    # for key, value in data.items():
    #     if key not in ['document_id', 'prompt', 'reference_output', 'default_negative']:
    #         example[key] = value
    return example


def load_dataset_from_jsonl(jsonl_path, aux_jsonl_path=None, max_examples=None):
    """
    Load a mock dataset from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file where each line contains:
                   - document_id: unique identifier
                   - prompt: the prompt text
                   - reference_output: the correct/reference output
                   - default_negative: a default incorrect output
        max_examples: Optional limit on number of examples to load

    Returns:
        MockDataset instance
    """
    examples = []
    running_total = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(line.strip()) > 0:
                data = json.loads(line)
                example = _construct_dictionary_from_json(data)
                examples.append(example)
                running_total += 1
            if max_examples and running_total >= max_examples:
                break
    if aux_jsonl_path is not None:
        with open(aux_jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if len(line.strip()) > 0:
                    data = json.loads(line)
                    example = _construct_dictionary_from_json(data)
                    examples.append(example)
                    running_total += 1
                if max_examples and running_total >= max_examples:
                    break
    print(f"Total examples loaded: {len(examples)}")
    return MockDataset(examples)


def load_dataset_with_generations(jsonl_path, aux_jsonl_path=None, generations_jsonl_path=None, max_examples=None):
    """
    Load dataset and existing generations (if available).

    Args:
        jsonl_path: Path to main dataset JSONL
        aux_jsonl_path: Optional additional dataset JSONL
        generations_jsonl_path: Optional path to generations JSONL
                               (format from GenerationSaverCallback)
        max_examples: Optional limit on examples

    Returns:
        Tuple of (MockDataset, all_generations_dict)
    """
    dataset = load_dataset_from_jsonl(jsonl_path, aux_jsonl_path=aux_jsonl_path, max_examples=max_examples)

    all_generations = {}
    if generations_jsonl_path:
        with open(generations_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    doc_id = data.get(sdm_network_constants.REEXPRESS_ID_KEY)
                    if doc_id:
                        all_generations[doc_id] = data
    print(f"Number of examples from training with generated hard negatives: {len(all_generations)}")
    return dataset, all_generations


# def prepare_verification_training_data(
#         train_jsonl_path,
#         model,
#         tokenizer,
#         calibration_jsonl_path=None,
#         generations_jsonl_path=None,
#         do_not_normalize_input_embeddings=False
# ):
#     """
#     Standalone function to prepare verification training data.
#
#     Args:
#         train_jsonl_path: Path to training data JSONL
#         model: The loaded LLM model for embedding extraction
#         tokenizer: The tokenizer
#         calibration_jsonl_path: Optional path to additional data JSONL
#         generations_jsonl_path: Optional path to cached generations
#         do_not_normalize_input_embeddings: Whether to skip normalization
#
#     Returns:
#         All outputs from process_llm_training_data_for_verification_training
#     """
#
#     # Load the dataset and generations
#     dataset, all_generations = load_dataset_with_generations(
#         jsonl_path=train_jsonl_path,
#         aux_jsonl_path=calibration_jsonl_path,
#         generations_jsonl_path=generations_jsonl_path
#     )
#
#     determine_verification_layer_training_split(dataset=dataset, eval_dataset=None, epoch_seed=None)
#
#     # Process for verification training
#     return process_llm_training_data_for_verification_training(
#         dataset=dataset,
#         model=model,
#         tokenizer=tokenizer,
#         all_generations=all_generations,
#         do_not_normalize_input_embeddings=do_not_normalize_input_embeddings
#     )
