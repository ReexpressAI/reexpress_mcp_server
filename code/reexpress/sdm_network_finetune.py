# Copyright Reexpress AI, Inc. All rights reserved.
"""
Main entry point for fine-tuning LMs as SDM LMs.

This currently is set up for microsoft/Phi-3.5-mini-instruct model for use on the research word ordering task,
but can be readily modified for use with other Hugging Face Transformer models and tasks.
This supports multi-GPU training using Accelerate, which should be used to run the script, as with:
    `accelerate launch --num_processes=NUM_GPUS sdm_network_finetune.py`
(The behavior may be unexpected if running the Python script directly without accelerate.)
"""

import argparse
import logging
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
import numpy as np


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# from transformers import logging as transformers_logging
# Set the logging level for the transformers library to ERROR, which will hide warnings.
# transformers_logging.set_verbosity_error()
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import json
import torch.distributed as dist
import random
import time

import sdm_network_constants
from sdm_network_finetune_utils_saver import GenerationSaverCallback
from sdm_network_finetune_utils_data_collator import CustomDataCollator
from sdm_network_finetune_utils_document_ordering_dataset import DocumentOrderingDataset
from sdm_network_finetune_utils_verification_layer import VerificationLayerCallback
from sdm_network_finetune_utils_trainer import GenerateAndVerifyTrainer

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
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data (JSONL)")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation data (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--generation_save_dir", type=str, required=True, help="Output directory for generations")
    parser.add_argument("--verification_model_dir", type=str, required=True,
                        help="Output directory for the verification model")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
                        help="Model name or path")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    # Train-time generation arguments
    parser.add_argument("--max_generation_new_tokens", type=int,
                        default=sdm_network_constants.MAX_NEW_TOKENS_FOR_GENERATION, help="Maximum sequence length")
    parser.add_argument("--generation_probability_during_training", type=float,
                        default=1.1,
                        help="Default is to always generate for the non-teacher-forced examples. "
                             "0.0 for no generations.")
    parser.add_argument("--do_not_reset_generations_every_round", default=False, action='store_true',
                            help="If not provided, by default, "
                                 "the generations are reset after training the verification layer. "
                                 "For the training set, the reset occurs at the beginning of the epoch after "
                                 "training the verification layer. For the validation set, "
                                 "the reset occurs after each evaluation loop after training the verification layer.")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")

    # Masking arguments
    parser.add_argument("--mask_prefix", action="store_true", default=False,
                        help="Whether to mask the prefix (system + user + wrong answers)")
    parser.add_argument("--mask_until_pattern", type=str, default="No</verified>",
                        help="Pattern to mask until (not including the pattern itself)")
    # Loss arguments
    parser.add_argument("--use_cross_entropy", action="store_true", default=False,
                        help="")
    parser.add_argument("--use_dpo", action="store_true", default=False,
                        help="")
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                        help="")
    parser.add_argument("--dpo_no_reference_model", action="store_true", default=False,
                        help="")
    # # Model type
    # parser.add_argument("--is_baseline", action="store_true", default=False,
    #                     help="Does not train the verification layer and assumes --use_cross_entropy")
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")


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
    assert args.eval_file, "A validation set must be provided via the --eval_file argument."
    assert args.mask_prefix
    if args.use_cross_entropy:
        print(f"Training with a cross-entropy loss.")
    elif args.use_dpo:
        print(f"WARNING: The DPO implementation is not fully tested.")
        print(f"Training with a DPO loss; beta={args.dpo_beta}, "
              f"using the reference model: {not args.dpo_no_reference_model}.")
    else:
        print(f"Training with the SDM next-token loss.")

    if args.generation_probability_during_training > 0.0:
        print(f"Training and validation evaluation will generate hard negatives with a probability of "
              f"{args.generation_probability_during_training}. "
              f"The generations will {'not ' if args.do_not_reset_generations_every_round else ''}be reset each round.")
    else:
        print(f"Training and validation evaluation will use the default negatives provided in the input files.")

    # assert not args.is_baseline, 'not implemented'
    # assert not args.use_cross_entropy, 'not implemented'
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
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=False,
        padding_side="right"
        # padding_side="left"
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=False,
        # trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        attn_implementation="eager",  # Use standard attention
        # Here, we assume accelerate launcher, so device_map must be commented out:
        # device_map="auto"  # Automatically distribute across available GPUs

        # OPTIONAL
        # Make sure you have flash-attn installed: pip install flash-attn
        # attn_implementation = "flash_attention_2",
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = DocumentOrderingDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        mask_prefix=args.mask_prefix,
        mask_until_pattern=args.mask_until_pattern
    )

    eval_dataset = None
    if args.eval_file:
        eval_dataset = DocumentOrderingDataset(
            args.eval_file,
            tokenizer,
            max_length=args.max_length,
            mask_prefix=args.mask_prefix,
            mask_until_pattern=args.mask_until_pattern
        )

    # Create data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=os.path.join(args.output_dir, "logs"),
        dataloader_num_workers=0, #4,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        push_to_hub=False,
        report_to=["tensorboard"],
    )

    generation_saver = GenerationSaverCallback(args.generation_save_dir)
    verification_layer_callback = VerificationLayerCallback(args.verification_model_dir, options=options, rng=rng)

    no_tokens = tokenizer.encode('No', add_special_tokens=False)
    yes_tokens = tokenizer.encode('Yes', add_special_tokens=False)

    if len(no_tokens) != 1 or len(yes_tokens) != 1:
        logger.error("'Yes'/'No' are not single tokens! This will break verification.")
        raise ValueError("Tokenizer incompatible - 'Yes'/'No' must be single tokens")

    # Create custom trainer with verification
    trainer = GenerateAndVerifyTrainer(
        is_sdm_network=not args.use_cross_entropy,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # processing_class=tokenizer,
        tokenizer=tokenizer,  # see comment in class definition
        no_token_id=no_tokens[0],
        yes_token_id=yes_tokens[0],
        mask_prefix=args.mask_prefix,
        mask_until_pattern=args.mask_until_pattern,
        model_max_length=args.max_length,
        generation_probability_during_training=args.generation_probability_during_training,
        reset_generations_every_round=not args.do_not_reset_generations_every_round,
        generation_config={
            "max_new_tokens": args.max_generation_new_tokens,
            "do_sample": False,
            # "temperature": 0.7,
            # "do_sample": True,
            # "top_p": 0.9,
        },
        use_cross_entropy=args.use_cross_entropy,
        use_dpo=args.use_dpo,
        dpo_beta=args.dpo_beta,
        dpo_no_reference_model=args.dpo_no_reference_model,
        generation_save_dir=args.generation_save_dir,
        options=options,
        # do not include verification_layer_callback for baseline
        callbacks=[generation_saver, verification_layer_callback] if not (args.use_cross_entropy or args.use_dpo) else [generation_saver],
    )

    # Manually link the trainer to the callback
    generation_saver.trainer = trainer
    verification_layer_callback.trainer = trainer

    # Train
    logger.info("Starting training...")
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint
    elif os.path.isdir(args.output_dir):
        checkpoint = get_last_checkpoint(args.output_dir)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")

    if trainer.args.local_rank in [-1, 0]:
        start_time = time.time()

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Ensure all ranks complete training
    if dist.is_initialized():
        dist.barrier()
    # Only save on rank 0
    if trainer.args.local_rank in [-1, 0]:
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

        # Save training results
        with open(os.path.join(args.output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)

        logger.info(f"Training completed! Model saved to {args.output_dir}")
        # This separation makes it easy to re-calibrate the verification layer with additional data over time:
        logger.info(f"Ready to complete training of the verification layer using the final best checkpoint via "
                    f"`sdm_network_train_verification_layer.py`.")
        print(f"Total training time: {time.time()-start_time}")


if __name__ == "__main__":
    main()
