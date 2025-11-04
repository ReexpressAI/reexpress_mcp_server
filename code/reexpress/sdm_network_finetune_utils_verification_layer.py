# Copyright Reexpress AI, Inc. All rights reserved.
"""
Callback that handles training and coordination (across devices) of the SDM verification layer during training.
"""

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
from transformers import TrainerCallback
import os
from pathlib import Path

import sdm_network_finetune_utils_verification_layer_utils_sdm
import sdm_network_finetune_utils_verification_layer_utils_preprocess
import utils_model
import sdm_network_constants


class VerificationLayerCallback(TrainerCallback):
    # This assumes the order: GenerationSaverCallback -> VerificationLayerCallback
    def __init__(self, verification_model_dir, options, rng):
        self.verification_model_dir = verification_model_dir
        os.makedirs(verification_model_dir, exist_ok=True)
        self.options = options
        self.rng = rng
        self.trainer = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        trainer = self.trainer

        # Synchronize before starting
        if dist.is_initialized():
            dist.barrier()

        if trainer and hasattr(trainer, 'train_dataset'):
            dataset = trainer.train_dataset
            if hasattr(trainer, 'eval_dataset'):
                eval_dataset = trainer.eval_dataset
            else:
                eval_dataset = None
                assert False, "ERROR: The current version expects a validation set to be provided."
            # Deterministic seed for split assignment - same on all ranks
            epoch_seed = state.epoch * 1000 + 42
            # First, determine whether the LLM training example will be used as the D_tr or D_ca of the verification
            # model. This is an in-place operation on the dataset, so in the case of distributed training, we do
            # this on all ranks.
            sdm_network_finetune_utils_verification_layer_utils_preprocess.determine_verification_layer_training_split(
                dataset=dataset, eval_dataset=eval_dataset, epoch_seed=epoch_seed)

        if dist.is_initialized():
            dist.barrier()

        # ===== DISTRIBUTED EMBEDDING GENERATION =====
        # All ranks participate in embedding generation
        if trainer and hasattr(trainer, 'train_dataset'):
            dataset = trainer.train_dataset
            tokenizer = self.trainer.tokenizer
            model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = args.local_rank if args.local_rank != -1 else 0

            print(f"Rank {rank}: Starting distributed embedding generation for epoch {state.epoch}")

            # Split dataset for this rank
            examples_subset = \
                sdm_network_finetune_utils_verification_layer_utils_preprocess.split_dataset_for_distributed_processing(
                    dataset, world_size, rank
                )

            # Process this rank's subset (always without normalization)
            train_uuids, train_metadata, train_embeddings, train_labels, \
                calibration_uuids, calibration_metadata, calibration_embeddings, calibration_labels = \
                sdm_network_finetune_utils_verification_layer_utils_preprocess.process_distributed_subset(
                    examples_subset=examples_subset,
                    model=model,
                    tokenizer=tokenizer,
                    all_generations=trainer.all_generations,
                    epoch_seed=state.epoch * 30.5
                )

            # Save this rank's results to disk
            embeddings_save_dir = \
                os.path.join(self.verification_model_dir,
                             sdm_network_constants.FILENAME_DISTRIBUTED_EMBEDDINGS_TRAINING_PREFIX_FILE_NAME)
            sdm_network_finetune_utils_verification_layer_utils_preprocess.save_distributed_embeddings(
                save_dir=embeddings_save_dir,
                rank=rank,
                epoch_or_step=state.epoch,
                is_training=True,
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

        # Synchronize after all ranks have saved their embeddings
        if dist.is_initialized():
            dist.barrier()

        # ===== CONSOLIDATION AND TRAINING ON RANK 0 =====
        # Only rank 0 consolidates and trains
        if args.local_rank in [-1, 0]:
            print(f"Rank 0: Consolidating embeddings from all ranks for epoch {state.epoch}")

            world_size = dist.get_world_size() if dist.is_initialized() else 1

            # Load and consolidate all embeddings, computing summary stats
            train_uuids, train_metadata, train_embeddings, train_labels, \
                calibration_uuids, calibration_metadata, calibration_embeddings, calibration_labels, \
                training_embedding_summary_stats = \
                sdm_network_finetune_utils_verification_layer_utils_preprocess.load_and_consolidate_distributed_embeddings(
                    save_dir=embeddings_save_dir,
                    world_size=world_size,
                    epoch_or_step=state.epoch,
                    is_training=True,
                    do_not_normalize_input_embeddings=self.options.do_not_normalize_input_embeddings
                )

            print(f"Rank 0: Consolidated embeddings, now training SDM verification layer")

            # Set up model directory for this epoch
            main_device = args.device
            print(f"The SDM model will use {main_device} as the main device for training.")

            path = Path(self.options.base_model_dir, f"{state.epoch}")
            path.mkdir(parents=False, exist_ok=True)
            epoch_model_dir = str(path.as_posix())
            self.options.model_dir = epoch_model_dir

            # Train the verification layer
            sdm_network_finetune_utils_verification_layer_utils_sdm.train_iterative_main(
                options=self.options,
                rng=self.rng,
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

            # Clean up intermediate files
            sdm_network_finetune_utils_verification_layer_utils_preprocess.cleanup_distributed_embedding_files(
                save_dir=embeddings_save_dir,
                world_size=world_size,
                epoch_or_step=state.epoch,
                is_training=True
            )

            print(f"Rank 0: Completed SDM verification layer training for epoch {state.epoch}")

        # Final synchronization before loading model on all ranks
        if dist.is_initialized():
            dist.barrier()

        if trainer and trainer.reset_generations_every_round:
            # Note: this occurs in GenerationSaverCallback.on_epoch_begin for the non-SDM losses
            print(f"Resetting generations in preparation for the next epoch")
            trainer.all_generations = {}
            if dist.is_initialized():
                dist.barrier()

        # Load the trained model on all ranks
        current_epoch_model_path = Path(self.options.base_model_dir, f"{state.epoch}")
        epoch_model_dir = str(current_epoch_model_path.as_posix())
        main_device = args.device

        # Load for inference is False to load the distance CDFs for training.
        self.trainer.sdm_verification_layer = \
            utils_model.load_model_torch(epoch_model_dir, main_device, load_for_inference=False)

        print(f"Rank {args.local_rank}: Loaded verification model on device: {main_device}")

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        """Additional metrics"""
        if trainer and hasattr(trainer, 'logging_sdm_outputs_for_predicted_class0') and \
                hasattr(trainer, 'logging_sdm_outputs_for_predicted_class1'):
            # Will be length 0 for cross-entropy baseline
            try:
                if len(trainer.logging_sdm_outputs_for_predicted_class0) > 0 or \
                        len(trainer.logging_sdm_outputs_for_predicted_class1) > 0:
                    rank = args.local_rank if args.local_rank != -1 else 0
                    # See note in compute_loss() regarding the variable naming.
                    print(f"Rank: {rank}. Summary statistics of sdm() for the reward label index across the epoch:")
                    number_of_time_sections = 10
                    if len(trainer.logging_sdm_outputs_for_predicted_class0) > 0:
                        sdm_outputs_for_predicted_class0 = torch.cat(trainer.logging_sdm_outputs_for_predicted_class0)
                        print(f"Rank: {rank}. Predicted probability for reward label index 0 mean: "
                              f"{torch.mean(sdm_outputs_for_predicted_class0).item()} "
                              f"out of {sdm_outputs_for_predicted_class0.shape[0]}")
                        sections = torch.tensor_split(sdm_outputs_for_predicted_class0, number_of_time_sections)
                        means = torch.tensor([section.mean() if section.shape[0] > 0 else 0.0 for section in sections])
                        for i, m in enumerate(means):
                            print(f"Rank: {rank}. Epoch progress {i}: "
                                  f"Predicted probability for reward label index 0 mean: {m.item()} "
                                  f"out of {sections[i].shape[0]}")
                        trainer.logging_sdm_outputs_for_predicted_class0 = []

                    if len(trainer.logging_sdm_outputs_for_predicted_class1) > 0:
                        sdm_outputs_for_predicted_class1 = torch.cat(trainer.logging_sdm_outputs_for_predicted_class1)
                        print(f"Rank: {rank}. Predicted probability for reward label index 1 mean: "
                              f"{torch.mean(sdm_outputs_for_predicted_class1).item()} "
                              f"out of {sdm_outputs_for_predicted_class1.shape[0]}")
                        sections = torch.tensor_split(sdm_outputs_for_predicted_class1, number_of_time_sections)
                        means = torch.tensor([section.mean() if section.shape[0] > 0 else 0.0 for section in sections])
                        for i, m in enumerate(means):
                            print(f"Rank: {rank}. Epoch progress {i}: "
                                  f"Predicted probability for reward label index 1 mean: {m.item()} "
                                  f"out of {sections[i].shape[0]}")
                        trainer.logging_sdm_outputs_for_predicted_class1 = []
            except:
                pass

        if dist.is_initialized():
            dist.barrier()
