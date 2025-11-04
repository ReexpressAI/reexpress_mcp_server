# Copyright Reexpress AI, Inc. All rights reserved.

"""
After fine-tuning the model, run this to train the verification model using the best checkpoint.
This separation makes it easy to re-calibrate the verification layer with additional data over time.
The convention/default is to construct the final D_tr and D_ca for the verification layer using
the calibration/validation set from fine-tuning.
"""
import argparse
import logging
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
import random
from accelerate import Accelerator
import torch.distributed as dist

import sdm_network_constants
import sdm_network_finetune_utils_verification_layer_utils_preprocess
import sdm_network_finetune_utils_verification_layer_utils_sdm

# sdm verification layer:
import constants


def remove_sdm_prefix(options):
    """Remove 'sdm_' prefix from all attributes in the namespace"""
    import argparse
    new_options = argparse.Namespace()
    sdm_prefix_string = 'sdm_'
    for key, value in vars(options).items():
        if key.startswith(sdm_prefix_string):
            new_key = key[len(sdm_prefix_string):]  # Remove first 4 characters ('sdm_')
            setattr(new_options, new_key, value)
        else:
            assert False
            # setattr(new_options, key, value)

    return new_options


def update_sdm_options(options, verification_model_dir):
    """Cache the original model directory. Each epoch will use a model directory identified by the LLM epoch."""
    setattr(options, "base_model_dir", verification_model_dir)
    setattr(options, "model_dir", verification_model_dir)
    setattr(options, "is_sdm_network_verification_layer", True)
    return options


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3.5-mini-instruct")

    # Data arguments
    # parser.add_argument("--train_file", type=str, required=True, help="Path to training data (JSONL)")
    # parser.add_argument("--eval_file", type=str, help="Path to evaluation data (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--verification_model_dir", type=str, required=True,
                        help="Output directory for the verification model")
    # parser.add_argument("--generation_save_dir", type=str, required=True, help="Output directory for generations")
    parser.add_argument("--use_baseline_model",
                        default=False, action='store_true',
                        help="If provided, the script will load 'microsoft/Phi-3.5-mini-instruct'")

    # Optional hard negative arguments
    parser.add_argument("--hard_negatives_file", type=str, default="",
                        help="Input file containing hard negatives (optional) corresponding to one or more documents "
                             "in --sdm_input_calibration_set_file. If not provided, the default negatives "
                             "in the input file will be exclusively used for class 0. "
                             "This file should match the .jsonl format of the "
                             "consolidated generation files constructed during initial training, as with "
                             "sdm_network_constants.FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_VALIDATION_FILE. "
                             "Generated output can be converted to this format using "
                             "sdm_network_process_generation_output_into_hard_negative_format.py.")
    parser.add_argument("--ignore_default_negative_if_generated_hard_negative_is_available",
                        default=False, action='store_true',
                        help="If provided, when sampling a negative, "
                             "the default negative in the input file will be ignored if a generated hard negative "
                             "is available for the instance.")

    # LLM Model arguments
    # parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
    #                     help="Model name or path")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training")

    # Masking arguments
    parser.add_argument("--mask_prefix", action="store_true", default=False,
                        help="Whether to mask the prefix (system + user + wrong answers)")
    parser.add_argument("--mask_until_pattern", type=str, default="No</verified>",
                        help="Pattern to mask until (not including the pattern itself)")

    #####################
    #### SDM verification layer arguments
    #####################
    sdm_parser = argparse.ArgumentParser(add_help=False)  # Disable help to avoid conflicts
    sdm_parser.add_argument("--sdm_input_training_set_file", default="",
                            help=".jsonl format")
    sdm_parser.add_argument("--sdm_input_calibration_set_file", default="",
                            help=".jsonl format")
    sdm_parser.add_argument("--sdm_input_eval_set_file", default="",
                            help=".jsonl format")
    sdm_parser.add_argument("--sdm_class_size", default=2, type=int, help="class_size")
    sdm_parser.add_argument("--sdm_epoch", default=20, type=int, help="number of max epoch")
    sdm_parser.add_argument("--sdm_batch_size", default=50, type=int, help="Batch size during training")
    sdm_parser.add_argument("--sdm_eval_batch_size", default=50, type=int,
                            help="Batch size during evaluation. "
                                 "This can (and should) typically be larger than the "
                                 "training batch size for efficiency.")
    sdm_parser.add_argument("--sdm_learning_rate", default=0.00001, type=float, help="learning rate")

    sdm_parser.add_argument("--sdm_alpha", default=constants.defaultCdfAlpha, type=float, help="alpha in (0,1), "
                                                                                               "typically 0.9 or 0.95")
    sdm_parser.add_argument("--sdm_maxQAvailableFromIndexer", default=constants.maxQAvailableFromIndexer, type=int,
                            help="max q considered")
    sdm_parser.add_argument("--sdm_use_training_set_max_label_size_as_max_q", default=False, action='store_true',
                            help="use_training_set_max_label_size_as_max_q")

    sdm_parser.add_argument("--sdm_eval_only", default=False, action='store_true', help="eval_only")
    # sdm_parser.add_argument("--sdm_model_dir", default="",
    #                         help="model_dir")  # use --verification_model_dir

    sdm_parser.add_argument("--sdm_use_embeddings", default=False, action='store_true', help="")
    sdm_parser.add_argument("--sdm_concat_embeddings_to_attributes", default=False, action='store_true', help="")

    sdm_parser.add_argument("--sdm_number_of_random_shuffles", default=20, type=int,
                            help="number of random shuffles of D_tr and D_ca, each of which is associated with a new"
                                 " f(x) := o of g of h(x), where h(x) is held frozen")
    sdm_parser.add_argument("--sdm_do_not_shuffle_data", default=False, action='store_true',
                            help="In this case, the data is not shuffled. If --number_of_random_shuffles > 1, "
                                 "iterations can still occur (to assess variation in learning, "
                                 "but the data stays fixed. "
                                 "Generally speaking, it's recommended to shuffle the data.")
    sdm_parser.add_argument("--sdm_is_training_support", default=False, action='store_true',
                            help="Include this flag if the eval set is the training set. "
                                 "This ignores the first match when calculating uncertainty, under the assumption that "
                                 "the first match is identity.")
    sdm_parser.add_argument("--sdm_recalibrate_with_updated_alpha", default=False, action='store_true',
                            help="This will update the model in the main directory, updating "
                                 "q'_min based on --alpha. However, note that the corresponding values for each "
                                 "iteration (and the global statistics) do not get updated, since we do not currently "
                                 "save the calibration data for every iteration.")
    sdm_parser.add_argument("--sdm_load_train_and_calibration_from_best_iteration_data_dir",
                            default=False, action='store_true', help="")
    sdm_parser.add_argument("--sdm_do_not_normalize_input_embeddings",
                            default=False, action='store_true',
                            help="Typically only use this if you have already standardized/normalized the embeddings. "
                                 "Our default approach is to mean center based on the training set embeddings. This is "
                                 "a global normalization that is applied in the forward of sdm_model.")
    sdm_parser.add_argument("--sdm_continue_training",
                            default=False, action='store_true', help="")
    sdm_parser.add_argument("--sdm_do_not_resave_shuffled_data",
                            default=False, action='store_true', help="")
    sdm_parser.add_argument("--sdm_exemplar_vector_dimension", default=constants.keyModelDimension, type=int, help="")
    sdm_parser.add_argument("--sdm_label_error_file", default="",
                            help="If provided, possible label annotation errors are saved, "
                                 "sorted by the LOWER predictive "
                                 "probability, where the subset is those that are valid index-conditional predictions.")
    sdm_parser.add_argument("--sdm_predictions_in_high_reliability_region_file", default="",
                            help="If provided, instances with predictions in the High Reliability region are saved, "
                                 "sorted by the output from sdm().")
    sdm_parser.add_argument("--sdm_prediction_output_file", default="",
                            help="If provided, output predictions are saved to this file "
                                 "in the order of the input file.")
    sdm_parser.add_argument("--sdm_update_support_set_with_eval_data", default=False, action='store_true',
                            help="update_support_set_with_eval_data")
    sdm_parser.add_argument("--sdm_skip_updates_already_in_support", default=False, action='store_true',
                            help="If --update_support_set_with_eval_data is provided, this will exclude any document "
                                 "with the same id already in the support set or the calibration set. If you are sure "
                                 "the documents are not already present, this can be excluded.")
    sdm_parser.add_argument("--sdm_main_device", default="cpu",
                            help="")
    sdm_parser.add_argument("--sdm_aux_device", default="cpu",
                            help="")
    sdm_parser.add_argument("--sdm_pretraining_initialization_epochs", default=0, type=int,
                            help="")
    sdm_parser.add_argument("--sdm_pretraining_learning_rate", default=0.00001, type=float, help="")
    sdm_parser.add_argument("--sdm_pretraining_initialization_tensors_file", default="",
                            help="")
    sdm_parser.add_argument("--sdm_ood_support_file", default="",
                            help="")
    sdm_parser.add_argument("--sdm_is_baseline_adaptor",
                            default=False, action='store_true',
                            help="Use this option to train and test a baseline adaptor using "
                                 "cross-entropy and softmax.")
    sdm_parser.add_argument("--sdm_construct_results_latex_table_rows",
                            default=False, action='store_true',
                            help="")
    sdm_parser.add_argument("--sdm_additional_latex_meta_data", default="", help="dataset,model_name")
    sdm_parser.add_argument("--sdm_print_timing",
                            default=False, action='store_true',
                            help="Used for profiling training.")

    args, remaining = parser.parse_known_args()
    options = sdm_parser.parse_args(remaining)
    options = remove_sdm_prefix(options)
    options = update_sdm_options(options, args.verification_model_dir)
    assert options.is_sdm_network_verification_layer
    assert args.mask_prefix
    assert options.input_training_set_file.strip() == "", \
        "The convention is to train the final verification layer using options.input_calibration_set_file"

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)

    # Setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.output_dir}")
    if args.use_baseline_model:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir,
            trust_remote_code=False,
            # Unlike training, for generation, padding is set to left. However, in practice, we generate one document
            # at a time without any padding.
            padding_side="left"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Reloading best model from {args.output_dir} on {rank}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir if not args.use_baseline_model else "microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=False,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        attn_implementation="eager",  # Force eager attention
        use_cache=True  # Enable KV cache
    )

    model = accelerator.prepare(model)
    model.eval()

    main_device = device  # Use accelerator's device
    print(f"Rank {rank}: Using device {main_device} for SDM model training.")

    # Give the model a namespace to reflect that it is from the final checkpoint
    path = Path(options.base_model_dir, f"final")
    path.mkdir(parents=False, exist_ok=True)
    options.model_dir = str(path.as_posix())

    if args.hard_negatives_file.strip() != "":
        hard_negatives_jsonl_path = args.hard_negatives_file.strip()

        if args.ignore_default_negative_if_generated_hard_negative_is_available:
            print(f"Training the verification layer with the default negatives in the input file and the "
                  f"generated negatives in "
                  f"{hard_negatives_jsonl_path}. If a document has one or more generated negatives, "
                  f"the default negative in the input file will be ignored when sampling negatives. Note that the "
                  f"generative negative may still be ignored if the document is not selected for class 0.")
        else:
            print(f"Training the verification layer with the default negatives in the input file and the "
                  f"generated negatives in "
                  f"{hard_negatives_jsonl_path}.")
    else:
        hard_negatives_jsonl_path = None
        print(f"Training the verification layer with the default negatives in the input file.")

    # Load dataset and generations (use this instead of prepare_verification_training_data)
    # Note the use of input_calibration_set_file (instead of input_training_set_file) and
    # hard_negatives_jsonl_path (instead of
    # FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_FILE)
    dataset, all_generations = \
        sdm_network_finetune_utils_verification_layer_utils_preprocess.load_dataset_with_generations(
            jsonl_path=options.input_calibration_set_file,
            aux_jsonl_path=None,
            generations_jsonl_path=hard_negatives_jsonl_path
        )

    data_process_split_seed = 23
    # Determine splits (all ranks)
    sdm_network_finetune_utils_verification_layer_utils_preprocess.determine_verification_layer_training_split(
        dataset=dataset, eval_dataset=None, epoch_seed=data_process_split_seed
    )

    # ===== DISTRIBUTED EMBEDDING GENERATION =====
    print(f"Rank {rank}: Starting distributed embedding generation")

    examples_subset = \
        sdm_network_finetune_utils_verification_layer_utils_preprocess.split_dataset_for_distributed_processing(
            dataset, world_size, rank
        )

    # Unwrap the model to access .generate() method
    unwrapped_model = accelerator.unwrap_model(model)

    train_uuids, train_metadata, train_embeddings, train_labels, \
        calibration_uuids, calibration_metadata, calibration_embeddings, calibration_labels = \
        sdm_network_finetune_utils_verification_layer_utils_preprocess.process_distributed_subset(
            examples_subset=examples_subset,
            model=unwrapped_model,
            tokenizer=tokenizer,
            all_generations=all_generations,
            epoch_seed=data_process_split_seed * 2,  # different from above,
            ignore_default_negative_if_generated_hard_negative_is_available=
            args.ignore_default_negative_if_generated_hard_negative_is_available
        )

    embeddings_save_dir = os.path.join(
        args.verification_model_dir,
        sdm_network_constants.FILENAME_DISTRIBUTED_EMBEDDINGS_VALIDATION_PREFIX_FILE_NAME
    )
    sdm_network_finetune_utils_verification_layer_utils_preprocess.save_distributed_embeddings(
        save_dir=embeddings_save_dir,
        rank=rank,
        epoch_or_step=0,  # Final training
        is_training=False,
        train_uuids=train_uuids,
        train_metadata=train_metadata,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        calibration_uuids=calibration_uuids,
        calibration_metadata=calibration_metadata,
        calibration_embeddings=calibration_embeddings,
        calibration_labels=calibration_labels
    )

    print(f"Rank {rank}: Completed embedding generation and saved results")
    accelerator.wait_for_everyone()

    # Destroy process group now that distributed work is done
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"Rank {rank}: Destroyed process group")

    # ===== RANK 0: CONSOLIDATE AND TRAIN =====
    if accelerator.is_main_process:
        print(f"Rank 0: Consolidating embeddings from {world_size} ranks")

        train_uuids, train_metadata, train_embeddings, train_labels, \
            calibration_uuids, calibration_metadata, calibration_embeddings, \
            calibration_labels, training_embedding_summary_stats = \
            sdm_network_finetune_utils_verification_layer_utils_preprocess.load_and_consolidate_distributed_embeddings(
                save_dir=embeddings_save_dir,
                world_size=world_size,
                epoch_or_step=0,
                is_training=False,
                do_not_normalize_input_embeddings=options.do_not_normalize_input_embeddings
            )

        print(f"Rank 0: Training SDM verification layer")

        sdm_network_finetune_utils_verification_layer_utils_sdm.train_iterative_main(
            options=options,
            rng=rng,
            main_device=main_device,
            train_uuids=train_uuids,
            calibration_uuids=calibration_uuids,
            train_data_json_list=train_metadata,
            calibration_data_json_list=calibration_metadata,
            train_embeddings=train_embeddings,
            calibration_embeddings=calibration_embeddings,
            train_labels=train_labels,
            calibration_labels=calibration_labels,
            training_embedding_summary_stats=training_embedding_summary_stats
        )

        # Cleanup
        sdm_network_finetune_utils_verification_layer_utils_preprocess.cleanup_distributed_embedding_files(
            save_dir=embeddings_save_dir,
            world_size=world_size,
            epoch_or_step=0,
            is_training=False
        )

        print(f"Saving the embeddings for the calibration and training split of the verification layer.")
        embeddings_save_dir = os.path.join(
            args.verification_model_dir,
            sdm_network_constants.FILENAME_DISTRIBUTED_EMBEDDINGS_VALIDATION_PREFIX_FILE_NAME
        )
        sdm_network_finetune_utils_verification_layer_utils_preprocess.save_distributed_embeddings(
            save_dir=embeddings_save_dir,
            rank=rank,
            epoch_or_step="final_consolidated",
            is_training=False,
            train_uuids=train_uuids,
            train_metadata=train_metadata,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            calibration_uuids=calibration_uuids,
            calibration_metadata=calibration_metadata,
            calibration_embeddings=calibration_embeddings,
            calibration_labels=calibration_labels
        )

        print(f"Rank 0: Completed final verification layer training")
    else:
        print(f"Rank {rank}: Exiting after distributed work")
        return  # Clean exit


if __name__ == "__main__":
    main()
