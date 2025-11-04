# Copyright Reexpress AI, Inc. All rights reserved.

"""
Custom Trainer for training an SDM language model.
"""
import logging
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from transformers import Trainer
import json
import torch.distributed as dist
import random
from pathlib import Path
import uuid
from copy import deepcopy

import sdm_network_constants
import sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings
import utils_eval_batch
import sdm_network_finetune_utils_verification_layer_utils_sdm
import sdm_network_finetune_utils_verification_layer_utils_preprocess
import utils_model
import sdm_network_finetune_utils_trainer_reward_assignment


class GenerateAndVerifyTrainer(Trainer):
    # Note: tokenizer is deprecated for processing_class, but
    # log noise persists even when dropping this and using processing_class,
    # so we keep the tokenizer argument for the time being since we know it works in 4.53.0
    def __init__(
            self,
            tokenizer,
            generation_config=None,
            no_token_id=None,
            yes_token_id=None,
            use_cross_entropy=False,
            use_dpo=False,
            dpo_beta=0.1,  # temperature parameter for DPO
            dpo_no_reference_model=False,  # option to exclude reference model
            generation_save_dir=None,
            mask_prefix: bool = True,
            mask_until_pattern: str = "No</verified>",
            model_max_length: int = 2048,
            is_sdm_network: bool = True,
            options=None,
            generation_probability_during_training=0.5,
            reset_generations_every_round=True,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.mask_prefix = mask_prefix
        self.mask_until_pattern = mask_until_pattern
        # Note the difference between self.model_max_length and self.generation_config["max_new_tokens"]
        self.model_max_length = model_max_length
        self.generation_config = generation_config or {
            "max_new_tokens": sdm_network_constants.MAX_NEW_TOKENS_FOR_GENERATION,
            "do_sample": False,
            # "temperature": 0.7,
            # "do_sample": True,
            # "top_p": 0.9,
        }

        # Get token IDs for Yes/No (based on get_verification_embedding.py)
        self.no_token_id = no_token_id or self.tokenizer.vocab['No']
        self.yes_token_id = yes_token_id or self.tokenizer.vocab['Yes']

        # Setup generation saving
        self.generation_save_dir = generation_save_dir or kwargs.get('args').output_dir
        os.makedirs(self.generation_save_dir, exist_ok=True)

        # Load existing generations if available
        self.all_generations = {}
        # Only called on init. Useful if you want to seed the generations before training starts. Otherwise, the
        # across device synchronization at the beginning of each epoch happens in the callback.
        self.all_generations_validation = {}  # generations for the validation set
        self.load_existing_generations()
        self.load_existing_generations_validation()

        self.is_sdm_network = is_sdm_network
        self.use_cross_entropy = use_cross_entropy

        self.options = options  # For retraining the verification layer
        # Each device has its own copy of the SDM activation layer model, which is trained on_epoch_begin on rank 0.
        # This is handled via callbacks.
        # After training, the SDM model is reloaded with the best LM checkpoint and retrained a final time in
        # preparation for testing.
        self.sdm_verification_layer = None
        self.sdm_verification_layer_validation = None

        self.generation_of_hard_negatives_has_begun = False
        self.generation_probability_during_training = generation_probability_during_training
        self.reset_generations_every_round = reset_generations_every_round

        # For monitoring the evolution of the SDM values over the course of training. These are reset every epoch in
        # VerificationLayerCallback().
        self.logging_sdm_outputs_for_predicted_class0 = []  # list of tensors
        self.logging_sdm_outputs_for_predicted_class1 = []  # list of tensors

        self.use_dpo = use_dpo
        self.dpo_beta = dpo_beta
        self.dpo_no_reference_model = dpo_no_reference_model

        # Store reference model for DPO, if applicable
        self.reference_model = None
        if self.use_dpo and not self.dpo_no_reference_model:
            # Create a deep copy of the model for reference
            # This will be set after model is available
            self.reference_model_needs_init = True
        else:
            self.reference_model_needs_init = False

    def load_existing_generations(self):
        """Load consolidated generations from previous runs"""
        consolidated_file = os.path.join(self.generation_save_dir,
                                         sdm_network_constants.FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_FILE)
        if os.path.exists(consolidated_file):
            with open(consolidated_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        doc_id = data.get(sdm_network_constants.REEXPRESS_ID_KEY)
                        if doc_id:
                            self.all_generations[doc_id] = data
            print(f"Loaded {len(self.all_generations)} existing generations (training)")

    def load_existing_generations_validation(self):
        """
            Load consolidated generations from previous eval runs. This is separate from
            load_existing_generations, because epoch and evaluate steps may not coincide. In particular, evaluate
            calls may be more frequent.
        """
        consolidated_file = \
            os.path.join(self.generation_save_dir,
                         sdm_network_constants.FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_VALIDATION_FILE)
        if os.path.exists(consolidated_file):
            with open(consolidated_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        doc_id = data.get(sdm_network_constants.REEXPRESS_ID_KEY)
                        if doc_id:
                            self.all_generations_validation[doc_id] = data
            print(f"Loaded {len(self.all_generations_validation)} existing generations (validation)")

    def load_and_return_existing_consolidated_generations_validation(self):
        """Load consolidated generations from previous evaluate runs. The equivalent for the training
        generations is in GenerationSaverCallback and called on_epoch_begin"""
        current_global_all_generations_at_epoch_start = {}
        consolidated_file = \
            os.path.join(self.generation_save_dir,
                         sdm_network_constants.FILENAME_TRAIN_TIME_HARD_NEGATIVES_GENERATIONS_VALIDATION_FILE)
        if os.path.exists(consolidated_file):
            with open(consolidated_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        doc_id = data.get(sdm_network_constants.REEXPRESS_ID_KEY)
                        if doc_id:
                            current_global_all_generations_at_epoch_start[doc_id] = data
            print(f"Loaded {len(current_global_all_generations_at_epoch_start)} existing "
                  f"generations (validation)")
        return current_global_all_generations_at_epoch_start

    def save_generations_rank_aware(self, epoch=None):
        """Save generations for current rank"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        epoch = epoch if epoch is not None else (self.state.epoch if hasattr(self, 'state') else 0)

        # Save to rank-specific file
        rank_file = os.path.join(
            self.generation_save_dir,
            f"generations_rank_{rank}_epoch_{epoch}.jsonl"
        )

        with open(rank_file, 'w') as f:
            for doc_id, data in self.all_generations.items():
                f.write(json.dumps(data) + '\n')

        print(f"Rank {rank}: Saved {len(self.all_generations)} generations (training) to {rank_file}")

    def save_generations_rank_aware_validation_at_step(self, global_step=None):
        """Save validation generations for current rank
        In practice, we only run this on rank 0"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        global_step = \
            global_step if global_step is not None else (self.state.global_step if hasattr(self, 'state') else 0)

        rank_file = os.path.join(
            self.generation_save_dir,
            f"generations_validation_rank_{rank}_global_step_{global_step}.jsonl"
        )

        with open(rank_file, 'w') as f:
            for doc_id, data in self.all_generations_validation.items():
                f.write(json.dumps(data) + '\n')

        # logger.info(f"Rank {rank}: "
        #             f"Saved {len(self.all_generations_validation)} generations (validation) to {rank_file}")
        print(f"Rank {rank}: "
                    f"Saved {len(self.all_generations_validation)} generations (validation) to {rank_file}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to retrain verification layer before evaluation with distributed embedding generation"""

        # Synchronize before starting
        if dist.is_initialized():
            dist.barrier()
        print(f"Entered evaluate on rank {self.args.local_rank}")
        self.all_generations_validation = self.load_and_return_existing_consolidated_generations_validation()

        if dist.is_initialized():
            dist.barrier()

        if self.use_cross_entropy or self.use_dpo:
            print(f"Using cross entropy, so SKIPPING training the verification layer for the validation set.")
            if self.reset_generations_every_round:
                # Note: this occurs below after training the verification layer in the case of the SDM loss
                print(f"Resetting generations in preparation for evaluation")
                self.all_generations_validation = {}
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # ===== DISTRIBUTED EMBEDDING GENERATION =====
        # All ranks participate in embedding generation
        dataset = self.eval_dataset
        tokenizer = self.tokenizer
        unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = self.args.local_rank if self.args.local_rank != -1 else 0

        print(f"Rank {rank}: Starting distributed embedding generation for eval at step {self.state.global_step}")

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
                model=unwrapped_model,
                tokenizer=tokenizer,
                all_generations=self.all_generations_validation,
                epoch_seed=self.state.global_step
            )

        # Save this rank's results to disk
        embeddings_save_dir = \
            os.path.join(self.options.base_model_dir,
                         sdm_network_constants.FILENAME_DISTRIBUTED_EMBEDDINGS_VALIDATION_PREFIX_FILE_NAME)
        sdm_network_finetune_utils_verification_layer_utils_preprocess.save_distributed_embeddings(
            save_dir=embeddings_save_dir,
            rank=rank,
            epoch_or_step=self.state.global_step,
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

        print(f"Rank {rank}: Completed eval embedding generation and saved results")

        # Synchronize after all ranks have saved their embeddings
        if dist.is_initialized():
            dist.barrier()

        # ===== CONSOLIDATION AND TRAINING ON RANK 0 =====
        # Only rank 0 consolidates and trains
        if self.args.local_rank in [-1, 0]:
            print(
                f"Rank 0: Training SDM verification layer (VALIDATION) before evaluation at step {self.state.global_step}")
            print(f"Rank 0: Consolidating embeddings from all ranks")

            # Load and consolidate all embeddings, computing summary stats
            train_uuids, train_metadata, train_embeddings, train_labels, \
                calibration_uuids, calibration_metadata, calibration_embeddings, calibration_labels, \
                training_embedding_summary_stats = \
                sdm_network_finetune_utils_verification_layer_utils_preprocess.load_and_consolidate_distributed_embeddings(
                    save_dir=embeddings_save_dir,
                    world_size=world_size,
                    epoch_or_step=self.state.global_step,
                    is_training=False,
                    do_not_normalize_input_embeddings=self.options.do_not_normalize_input_embeddings
                )

            print(f"Rank 0: Consolidated embeddings, now training SDM verification layer")

            # Set up model directory
            main_device = self.args.device
            print(f"The SDM model will use {main_device} as the main device for training.")

            path = Path(self.options.base_model_dir, f"eval_{self.state.global_step}")
            path.mkdir(parents=False, exist_ok=True)
            epoch_model_dir = str(path.as_posix())
            self.options.model_dir = epoch_model_dir

            # Train the verification layer
            sdm_network_finetune_utils_verification_layer_utils_sdm.train_iterative_main(
                options=self.options,
                rng=None,
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
                epoch_or_step=self.state.global_step,
                is_training=False
            )

            print(f"Rank 0: Completed SDM verification layer training for evaluation")

        # Final synchronization before loading model on all ranks
        if dist.is_initialized():
            dist.barrier()

        # Load the newly trained model on all ranks
        current_epoch_model_path = Path(self.options.base_model_dir, f"eval_{self.state.global_step}")
        epoch_model_dir = str(current_epoch_model_path.as_posix())
        main_device = self.args.device

        # Load for inference is False to load the distance CDFs for training.
        self.sdm_verification_layer_validation = \
            utils_model.load_model_torch(epoch_model_dir, main_device, load_for_inference=False)

        print(f"Rank {self.args.local_rank}: Loaded verification model for evaluation on device: {main_device}")

        # Barrier to ensure all ranks have loaded the model before evaluation
        if dist.is_initialized():
            dist.barrier()

        if self.reset_generations_every_round:
            # Note: this occurs above for the non-SDM losses
            print(f"Resetting generations in preparation for evaluation")
            self.all_generations_validation = {}
            if dist.is_initialized():
                dist.barrier()

        # Now run the actual evaluation
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def get_ids_with_padding(self, prompt_text, assistant_response):
        conv = sdm_network_constants.get_conv(prompt_text)
        conv.append({"role": "assistant", "content": assistant_response})

        # Tokenize the conversation
        encoding = self.tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            max_length=self.model_max_length,
            padding=False, #"max_length",
            truncation=True,
            return_dict=True,
            add_generation_prompt=False  # We have the assistant response
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = \
            sdm_network_constants.create_masked_labels(self.mask_prefix, self.tokenizer, self.mask_until_pattern,
                                                       input_ids.tolist(), assistant_response)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def re_collate(self, features):
        # Use tokenizer's built-in padding for input_ids and attention_mask
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"],
              "attention_mask": f["attention_mask"]} for f in features],
            padding=True,  # Pad to longest in batch
            return_tensors="pt"
        )

        # Handle labels separately
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            labels = f["labels"]
            padding_length = max_length - len(labels)
            padded = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=labels.dtype)
            ])
            padded_labels.append(padded)

        batch["labels"] = torch.stack(padded_labels)
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remember the distinction between sentence text and the full response with the verified tokens
        if self.use_cross_entropy:
            return self._compute_baseline_cross_entropy_loss_with_train_time_generations(
                model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)
        elif self.use_dpo:
            return self._compute_baseline_dpo_loss_with_train_time_generations(
                model, inputs, return_outputs, num_items_in_batch=num_items_in_batch
            )
        # In multiple places below (embedding forward passes and generations), we change the state to eval(),
        # so we cache the original state in in_training.
        in_training = model.training
        # Note that the support set of the verification layer is only locally updated when model.training.
        if in_training:
            # print(f"model.training: {model.training}")
            sdm_verification_layer_pointer = self.sdm_verification_layer
            all_generations_pointer = self.all_generations

            self.generation_of_hard_negatives_has_begun = self.state.epoch > 1.0
        else:
            # print(f"model.training: {model.training}")
            sdm_verification_layer_pointer = self.sdm_verification_layer_validation
            all_generations_pointer = self.all_generations_validation

        if not self.is_sdm_network or sdm_verification_layer_pointer is None:
            raise RuntimeError("SDM verification layer not initialized. Ensure VerificationLayerCallback has run.")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        document_ids = inputs["document_id"]  # Get document IDs
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]

        default_negatives = inputs[sdm_network_constants.DEFAULT_NEGATIVE_KEY]
        reference_document_strings = inputs[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]
        prompt_texts = inputs[sdm_network_constants.CACHED_PROMPT_KEY]
        # The parameters of the LLM are changing, and we are introducing novel hard negatives given the prompt,
        # but as a reasonable first approximation, we assume the
        # first match for each training instance is an identity match, which we discard when calculating q and d.
        # (This is relative to the given verification layer. For the validation verification layer, the
        # parameters of the LLM are fixed, but there may still be some non-exact matches due to the randomization
        # of the hard negatives of the response. However, the prompts would still be exact matches, so we always
        # discard the first match.)
        is_sdm_training_split_indicators = inputs[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY]

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Importantly, we need to distinguish from examples that were in the training support set of the
        # verification layer in order to ignore the first match in those cases.
        verification_embedding__is_support_set = []
        verification_embedding__not_is_support_set = []
        is_support_mask = torch.zeros(batch_size).to(device)  # 1 for training; 0 otherwise
        updated_features = []

        # Data structures for support set additions, local to device. We only update the instances that were in the
        # training set split of the verification layer. This only applies to self.sdm_verification_layer. In contrast,
        # with self.sdm_verification_layer_validation this is not applicable, since the layer is retrained
        # before this is run (so there is no divergence relative to the given LLM parameters).
        support_set_update_document_ids = []
        support_set_update_labels = []

        # We maintain separate structures for the document-level reward ground-truth labels.
        # In principle these could be from a
        # stochastic model (e.g., another SDM layer, potentially over a composition of models),
        # but in the present version,
        # these are assumed to be deterministic ("verifiable") rewards.
        support_reward_labels = []
        not_support_reward_labels = []

        # In a distributed setup, the model is wrapped in DDP.
        # We need to access the underlying model via the .module attribute
        # to call methods like .generate()
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # The loss is calculated over ground-truth examples and existing or generated hard negatives. We need
        # to generate from the model to collect candidate hard negatives. Additionally, for each of the inputs,
        # we need to retrieve the embeddings for the verification layer.
        for i in range(batch_size):
            document_id = document_ids[i]
            default_negative = default_negatives[i]
            reference_document_string = reference_document_strings[i]
            prompt_text = prompt_texts[i]
            is_sdm_training_split = is_sdm_training_split_indicators[i]

            # Add validation
            if is_sdm_training_split not in [0, 1]:
                raise ValueError(
                    f"Invalid IS_SDM_TRAINING_SPLIT value: "
                    f"{is_sdm_training_split} on rank {dist.get_rank() if dist.is_initialized() else 0}")

            # Half the time, always correct:
            # if (not model.training) or (random.random() < 0.5):
            if random.random() < 0.5:
                updated_features.append(
                    {
                        "input_ids": input_ids[i].cpu(),
                        "attention_mask": attention_mask[i].cpu(),
                        "labels": labels[i].cpu()
                    }
                )
                # BEGIN -- Get embedding for verification layer
                # Note that the reference_document_string does not contain any existing tags.
                # Contrast this with the use of generate_formatted_assistant_response_from_malformed_output() below.
                _, verification_response = \
                    sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                        sentences=[reference_document_string], verifications=[True])
                verification_input_ids, _ = \
                    sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_ids_from_prompt_text_and_assistant_response(
                        self.tokenizer, prompt_text, verification_response)
                embedding = \
                    sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_verification_embedding(
                        model=unwrapped_model, input_ids=verification_input_ids)
                if is_sdm_training_split == 1:
                    verification_embedding__is_support_set.append(embedding)
                    is_support_mask[i] = 1
                    # for updates to the support set, local to this device:
                    support_set_update_document_ids.append(
                        f"{document_id}_{str(uuid.uuid4())}_{self.state.epoch if hasattr(self, 'state') else 0}")
                    support_set_update_labels.append(1)
                    support_reward_labels.append(1)
                elif is_sdm_training_split == 0:
                    verification_embedding__not_is_support_set.append(embedding)
                    not_support_reward_labels.append(1)
                else:
                    print("WARNING: The training instance is missing the SDM training split indicator")
                    assert False
                # End -- Get embedding for verification layer
            else:
                # This example will contain an error, either in the response text or as a hint example (the latter of which
                # may end in the correct answer or an incorrect answer).
                error_candidates = [default_negative]
                hard_negatives = []
                # Check if we already have generations for this document
                if document_id in all_generations_pointer:
                    # Load existing generation
                    saved_data = all_generations_pointer[document_id]
                    # Reconstruct verification data from saved hard negatives. We keep the list with the default hard
                    # negative separate:
                    error_candidates.extend(saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, []))
                    hard_negatives = saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, [])

                # Generate new
                with torch.no_grad():
                    # Note that we have now changed the internal state of the model in the lifecycle of Trainer.
                    # This is why we cache the original state via in_training
                    # model.eval()
                    unwrapped_model.eval()
                    if self.generation_of_hard_negatives_has_begun and \
                            random.random() < self.generation_probability_during_training:
                        # Use the prompt_input_ids directly
                        prompt_ids = prompt_input_ids[i]
                        prompt_attention = prompt_attention_mask[i]

                        # Remove padding
                        prompt_length = (prompt_attention == 1).sum().item()
                        prompt_ids = prompt_ids[:prompt_length]
                        prompt_attention = prompt_attention[:prompt_length]

                        # Generate
                        generated_outputs = unwrapped_model.generate(
                            input_ids=prompt_ids.unsqueeze(0),
                            attention_mask=prompt_attention.unsqueeze(0),
                            max_new_tokens=self.generation_config.get("max_new_tokens",
                                                                      sdm_network_constants.MAX_NEW_TOKENS_FOR_GENERATION),
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False  # This avoids the caching/gradient checkpointing conflict
                        )

                        # Decode
                        generated_tokens = generated_outputs[0][len(prompt_ids):]
                        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                        generation_matches_reference = \
                            sdm_network_finetune_utils_trainer_reward_assignment.is_exact_match_excluding_boundary_whitespace(
                                reference_document_string=
                                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                                    sentences=[reference_document_string],
                                    verifications=[True])[0],
                                generated_sentence=generated_text)
                        # We also need to ignore the case in which the correct sentence is generated but the
                        # classification is wrong, since that is already handled by the true positive teacher-forcing,
                        # and including it here would force decode the wrong label.
                        generation_matches_reference_text_but_classification_is_wrong = \
                            sdm_network_finetune_utils_trainer_reward_assignment.is_exact_match_excluding_boundary_whitespace(
                                reference_document_string=
                                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                                    sentences=[reference_document_string],
                                    verifications=[False])[0],
                                generated_sentence=generated_text)

                        if not generation_matches_reference and \
                                not generation_matches_reference_text_but_classification_is_wrong:
                            # The response is not an exact match of the text and/or formatting,
                            # so we ensure "<verified>No</verified>\n\n" occurs at the end
                            response, _ = \
                                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                                    generated_text=generated_text)
                            error_candidates.append(response)
                            hard_negatives.append(response)

                    # shuffle error candidates in place
                    random.shuffle(error_candidates)
                    assistant_text = error_candidates[0]

                    # print(f"Hard negative prompt: {prompt_text}\n\nAssistant: {assistant_text}\n\n----------")
                    updated_input_dict = self.get_ids_with_padding(prompt_text, assistant_text)
                    # print(f"updated_dict: {assistant_text}, {updated_input_dict}")
                    updated_features.append(updated_input_dict)

                    # BEGIN -- Get embedding for verification layer
                    # Unlike case with the reference_document_string,
                    # in this case the assistant test will be formatted with tags, so we need to
                    # reformat for verification.
                    _, verification_response = \
                            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                                generated_text=assistant_text)
                    verification_input_ids, _ = \
                        sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_ids_from_prompt_text_and_assistant_response(
                            self.tokenizer, prompt_text, verification_response)
                    embedding = \
                        sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.get_verification_embedding(
                            model=unwrapped_model, input_ids=verification_input_ids)
                    if is_sdm_training_split == 1:
                        verification_embedding__is_support_set.append(embedding)
                        is_support_mask[i] = 1
                        # for updates to the support set, local to this device:
                        support_set_update_document_ids.append(
                            f"{document_id}_{str(uuid.uuid4())}_{self.state.epoch if hasattr(self, 'state') else 0}")
                        support_set_update_labels.append(0)
                        support_reward_labels.append(0)
                    elif is_sdm_training_split == 0:
                        verification_embedding__not_is_support_set.append(embedding)
                        not_support_reward_labels.append(0)
                    else:
                        print("WARNING: The training instance is missing the SDM training split indicator")
                        assert False
                    # End -- Get embedding for verification layer

                    # Store for saving
                    if in_training:
                        progress_reference_key = self.state.epoch if hasattr(self, 'state') else 0
                    else:
                        progress_reference_key = self.state.global_step if hasattr(self, 'state') else 0

                    all_generations_pointer[document_id] = {
                        sdm_network_constants.REEXPRESS_ID_KEY: document_id,
                        # List of incorrect sentences after deduplication. This can be an empty list.
                        sdm_network_constants.HARD_NEGATIVES_KEY: list(set(hard_negatives)),
                        # epoch will be the last epoch/step for which this document id was updated with a hard negative:
                        sdm_network_constants.SDM_NETWORK_FINETUNING_EPOCH_KEY: progress_reference_key,
                    }

        assert len(updated_features) == batch_size, \
            f"Invariant violated: updated_features has {len(updated_features)} items but batch_size is {batch_size}"

        # Todo: Currently there is a device transfer from gpu to cpu to gpu for the embeddings. This is because the
        #       support functions are setup to batch over the full dataset, whereas here we are already processing
        #       at the batch level, so it should be OK to keep on gpu.
        # Calculate q and d, given the embeddings
        present__is_support_set = len(verification_embedding__is_support_set) > 0
        present__not_is_support_set = len(verification_embedding__not_is_support_set) > 0
        need_to_combine = present__is_support_set and present__not_is_support_set
        if present__is_support_set:
            verification_embedding__is_support_set = torch.cat(verification_embedding__is_support_set, dim=0)
            support_dataset_q_values, support_distance_quantile_per_class, support_batch_f_outputs, _, _, \
                support_exemplar_vectors = \
                utils_eval_batch.get_q_and_d_from_embeddings(sdm_verification_layer_pointer,
                                                             eval_batch_size=
                                                             verification_embedding__is_support_set.shape[0],
                                                             eval_embeddings=verification_embedding__is_support_set,
                                                             main_device=device,
                                                             is_training_support=True,
                                                             return_exemplar_vectors=True)
            support_batch_f_outputs = support_batch_f_outputs.to(device)
            support_dataset_q_values = support_dataset_q_values.to(device)
            support_distance_quantile_per_class = support_distance_quantile_per_class.to(device)
            support_batch_sdm = \
                sdm_verification_layer_pointer.soft_sdm_max(
                    support_batch_f_outputs, support_dataset_q_values,
                    distance_quantile_per_class=support_distance_quantile_per_class)
            support_rescaled_similarities, support_predictions = \
                sdm_verification_layer_pointer.get_rescaled_similarity_for_eval_batch(
                    cached_f_outputs=support_batch_f_outputs,
                    dataset_q_values=support_dataset_q_values,
                    sdm_outputs=support_batch_sdm,
                    return_tensors_on_cpu=False, keepdim=True, return_sdm_outputs_for_predicted=False)
            support_reward_labels = torch.tensor(support_reward_labels).to(device)
            support_sdm_outputs_at_reward_label_index = \
                sdm_network_finetune_utils_trainer_reward_assignment.get_sdm_outputs_for_index(
                    sdm_outputs=support_batch_sdm, index_tensor=support_reward_labels,
                    return_tensors_on_cpu=False,
                    keepdim=True)
            assert support_rescaled_similarities.shape[0] == verification_embedding__is_support_set.shape[0]
            assert support_rescaled_similarities.shape[1] == 1

            # update running support set on this device:
            if False and in_training:
                # This is not currently used, but we provide it as an example that may be useful as a tradeoff
                # between full retraining of the verification layer every epoch.
                # Note: Local support updates are not relevant for the validation set, since
                # the LLM parameters are frozen.
                sdm_verification_layer_pointer.add_to_support_batch(
                    torch.tensor(support_set_update_labels),
                    predicted_labels=support_predictions.squeeze(),  # squeeze() since keepdim=True above
                    document_ids=support_set_update_document_ids,
                    exemplar_vectors=support_exemplar_vectors)
            # print(f"Updated support with {len(support_set_update_labels)} new instances.")
        if present__not_is_support_set:
            verification_embedding__not_is_support_set = torch.cat(verification_embedding__not_is_support_set, dim=0)
            not_support_dataset_q_values, not_support_distance_quantile_per_class, not_support_batch_f_outputs, _, _ = \
                utils_eval_batch.get_q_and_d_from_embeddings(sdm_verification_layer_pointer,
                                                             eval_batch_size=verification_embedding__not_is_support_set.shape[0],
                                                             eval_embeddings=verification_embedding__not_is_support_set,
                                                             main_device=device,
                                                             is_training_support=False)
            not_support_batch_f_outputs = not_support_batch_f_outputs.to(device)
            not_support_dataset_q_values = not_support_dataset_q_values.to(device)
            not_support_distance_quantile_per_class = not_support_distance_quantile_per_class.to(device)
            not_support_batch_sdm = \
                sdm_verification_layer_pointer.soft_sdm_max(
                    not_support_batch_f_outputs, not_support_dataset_q_values,
                    distance_quantile_per_class=not_support_distance_quantile_per_class)
            not_support_rescaled_similarities, not_support_predictions = \
                sdm_verification_layer_pointer.get_rescaled_similarity_for_eval_batch(
                    cached_f_outputs=not_support_batch_f_outputs,
                    dataset_q_values=not_support_dataset_q_values,
                    sdm_outputs=not_support_batch_sdm,
                    return_tensors_on_cpu=False, keepdim=True, return_sdm_outputs_for_predicted=False)
            not_support_reward_labels = torch.tensor(not_support_reward_labels).to(device)
            not_support_sdm_outputs_at_reward_label_index = \
                sdm_network_finetune_utils_trainer_reward_assignment.get_sdm_outputs_for_index(
                    sdm_outputs=not_support_batch_sdm, index_tensor=not_support_reward_labels,
                    return_tensors_on_cpu=False,
                    keepdim=True)
        if need_to_combine:
            # Combine support_dataset_q_values with not_support_dataset_q_values
            # Combine support_distance_quantile_per_class and not_support_distance_quantile_per_class
            # so that both are ordered such that the rows match those of updated_features
            assert batch_size == len(updated_features)
            is_support_mask = is_support_mask.bool()
            # Assertions to verify counts match
            assert is_support_mask.sum() == support_dataset_q_values.shape[0], \
                f"Support mask count {is_support_mask.sum()} doesn't match support data size " \
                f"{support_dataset_q_values.shape[0]}"
            assert (~is_support_mask).sum() == not_support_dataset_q_values.shape[0], \
                f"Non-support mask count {(~is_support_mask).sum()} doesn't match non-support data size " \
                f"{not_support_dataset_q_values.shape[0]}"
            # Initialize combined tensors
            combined_sdm_outputs_at_reward_label_index = torch.zeros(
                batch_size,
                1,  # always 1
                device=device,
                dtype=support_sdm_outputs_at_reward_label_index.dtype
            )
            # These additional structures are constructed for reference analysis, but are not used in the
            # loss calculation:
            combined_rescaled_similarities = torch.zeros(
                batch_size,
                1,  # always 1
                device=device,
                dtype=support_rescaled_similarities.dtype
            )
            combined_q_values = torch.zeros(
                batch_size,
                1,  # always 1
                device=device,
                dtype=support_dataset_q_values.dtype
            )
            combined_distance_quantile = torch.zeros(
                batch_size,
                1,  # only using first dimension, rather than support_distance_quantile_per_class.shape[1],
                device=device,
                dtype=support_distance_quantile_per_class.dtype
            )
            # Use boolean indexing to place values in original order
            combined_sdm_outputs_at_reward_label_index[is_support_mask] = \
                support_sdm_outputs_at_reward_label_index.to(device)
            combined_sdm_outputs_at_reward_label_index[~is_support_mask] = \
                not_support_sdm_outputs_at_reward_label_index.to(device)

            if in_training:
                # Monitor the evolution of the SDM values over the course of training. This is only for reference.
                # NOTE: For simplicity, we only log in the case of need_to_combine.
                # This is a repurposed holdover from earlier versions, and the naming of the class attribute is now a
                # misnomer. The variable self.logging_sdm_outputs_for_predicted_class0 is more accurately
                # 'self.logging_sdm_outputs_at_reward_label_index0'.
                # The variable self.logging_sdm_outputs_for_predicted_class1 is more accurately
                # 'self.logging_sdm_outputs_at_reward_label_index1'.
                class_label = 0
                self.logging_sdm_outputs_for_predicted_class0.append(
                    support_sdm_outputs_at_reward_label_index[
                        support_reward_labels.unsqueeze(1) == class_label].detach().cpu())
                self.logging_sdm_outputs_for_predicted_class0.append(
                    not_support_sdm_outputs_at_reward_label_index[
                        not_support_reward_labels.unsqueeze(1) == class_label].detach().cpu())
                class_label = 1
                self.logging_sdm_outputs_for_predicted_class1.append(
                    support_sdm_outputs_at_reward_label_index[
                        support_reward_labels.unsqueeze(1) == class_label].detach().cpu())
                self.logging_sdm_outputs_for_predicted_class1.append(
                    not_support_sdm_outputs_at_reward_label_index[
                        not_support_reward_labels.unsqueeze(1) == class_label].detach().cpu())

            combined_rescaled_similarities[is_support_mask] = support_rescaled_similarities.to(device)
            combined_rescaled_similarities[~is_support_mask] = not_support_rescaled_similarities.to(device)

            combined_q_values[is_support_mask] = support_dataset_q_values.to(device)
            combined_q_values[~is_support_mask] = not_support_dataset_q_values.to(device)
            # Only need first index since the values are the same across classes:
            combined_distance_quantile[is_support_mask] = support_distance_quantile_per_class[:, 0, None].to(device)
            combined_distance_quantile[~is_support_mask] = not_support_distance_quantile_per_class[:, 0, None].to(device)

            # Now combined_q_values and combined_distance_quantile have the same row order as updated_features. Note that
            # q and d are at the document level, so we expand to the sequence level below, with applicable padding for
            # the batch.
        else:
            if present__is_support_set:
                combined_sdm_outputs_at_reward_label_index = support_sdm_outputs_at_reward_label_index.to(device)
                combined_rescaled_similarities = support_rescaled_similarities.to(device)
                combined_q_values = support_dataset_q_values.to(device)
                combined_distance_quantile = support_distance_quantile_per_class[:, 0, None].to(device)
            else:
                assert present__not_is_support_set
                combined_sdm_outputs_at_reward_label_index = not_support_sdm_outputs_at_reward_label_index.to(device)
                combined_rescaled_similarities = not_support_rescaled_similarities.to(device)
                combined_q_values = not_support_dataset_q_values.to(device)
                combined_distance_quantile = not_support_distance_quantile_per_class[:, 0, None].to(device)
        # update padding:
        updated_features_on_cpu = self.re_collate(updated_features)

        if self.use_cross_entropy:
            assert False, "For cross entropy, use self._compute_baseline_cross_entropy_loss_with_train_time_generations"
            # # Standard forward pass and loss computation
            # if in_training:
            #     model.train()
            # else:
            #     model.eval()
            # outputs = model(
            #     input_ids=updated_features_on_cpu["input_ids"].to(device),
            #     attention_mask=updated_features_on_cpu["attention_mask"].to(device),
            #     labels=updated_features_on_cpu["labels"].to(device),
            #     return_dict=True
            # )
            # return (outputs.loss, outputs) if return_outputs else outputs.loss
        else:
            return self.get_sdm_loss(model, sdm_verification_layer_pointer,
                                     updated_features_on_cpu, combined_rescaled_similarities,
                                     combined_q_values, combined_distance_quantile,
                                     combined_sdm_outputs_at_reward_label_index,
                                     batch_size, device, return_outputs=return_outputs, in_training=in_training)

    def get_sdm_loss(self, model, sdm_verification_layer_pointer,
                     updated_features_on_cpu, combined_rescaled_similarities,
                     combined_q_values, combined_distance_quantile,
                     combined_sdm_outputs_at_reward_label_index,
                     batch_size, device, return_outputs=False, in_training=True):
        # combined_rescaled_similarities, combined_q_values,
        # combined_distance_quantile are not currently used, but are included as args for
        # debugging/analysis
        if in_training:
            model.train()
        else:
            model.eval()
        # Get raw logits without computing loss
        outputs = model(
            input_ids=updated_features_on_cpu["input_ids"].to(device),
            attention_mask=updated_features_on_cpu["attention_mask"].to(device),
            labels=None,  # Don't pass labels to avoid automatic loss computation,
            return_dict=True
        )
        # Get the raw logits - shape: (batch_size, seq_len, vocab_size)
        shifted_seq_length = outputs.logits.shape[1] - 1
        logits = outputs.logits
        labels = updated_features_on_cpu["labels"].to(device)

        # Shift for causal LM: predict next token
        # Remove last logit position and first label position
        shifted_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shifted_labels = labels[:, 1:].contiguous()  # (B, L-1)

        # Compute log probabilities with SDM[r(ground-truth, example)] as the base
        batch_sdm_log = \
            sdm_verification_layer_pointer.soft_sdm_max(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),  # (B*(L-1), V)
                combined_sdm_outputs_at_reward_label_index.expand(batch_size, shifted_seq_length).reshape(-1, 1),  # (B*(L-1), 1)
                distance_quantile_per_class=None,
                log=True, change_of_base=True)  # (B*(L-1), V)

        total_loss = F.nll_loss(
            batch_sdm_log,  # (B*(L-1), V)
            shifted_labels.reshape(-1),  # (B*(L-1),)
            ignore_index=-100
        )

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_baseline_cross_entropy_loss_with_train_time_generations(self,
                                                                         model, inputs, return_outputs=False,
                                                                         num_items_in_batch=None):
        # This is similar to compute_loss, but when using cross-entropy, we do not need the calls to
        # the verification layer. We separate these for now, since this case has simplified
        # data structures, since the VerificationLayerCallback will not have been initialized.
        assert self.use_cross_entropy
        # cache the original state in in_training.
        in_training = model.training
        if in_training:
            all_generations_pointer = self.all_generations
            self.generation_of_hard_negatives_has_begun = self.state.epoch > 1.0
        else:
            all_generations_pointer = self.all_generations_validation

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        document_ids = inputs["document_id"]  # Get document IDs
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]

        default_negatives = inputs[sdm_network_constants.DEFAULT_NEGATIVE_KEY]
        reference_document_strings = inputs[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]
        prompt_texts = inputs[sdm_network_constants.CACHED_PROMPT_KEY]

        batch_size = input_ids.shape[0]
        device = input_ids.device

        updated_features = []

        # In a distributed setup, the model is wrapped in DDP.
        # We need to access the underlying model via the .module attribute
        # to call methods like .generate()
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # The loss is calculated over ground-truth examples and existing or generated hard negatives. We need
        # to generate from the model to collect candidate hard negatives. Additionally, for each of the inputs,
        # we need to retrieve the embeddings for the verification layer.
        for i in range(batch_size):
            document_id = document_ids[i]
            default_negative = default_negatives[i]
            reference_document_string = reference_document_strings[i]
            prompt_text = prompt_texts[i]

            # Half the time, always correct:
            # if (not model.training) or (random.random() < 0.5):
            if random.random() < 0.5:
                updated_features.append(
                    {
                        "input_ids": input_ids[i].cpu(),
                        "attention_mask": attention_mask[i].cpu(),
                        "labels": labels[i].cpu()
                    }
                )
            else:
                # This example will contain an error, either in the response text or as a hint example (the latter of
                # which may end in the correct answer or an incorrect answer).
                error_candidates = [default_negative]
                hard_negatives = []
                # Check if we already have generations for this document
                if document_id in all_generations_pointer:
                    # Load existing generation
                    saved_data = all_generations_pointer[document_id]
                    # Reconstruct verification data from saved hard negatives. We keep the list with the default hard
                    # negative separate:
                    error_candidates.extend(saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, []))
                    hard_negatives = saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, [])

                    # Generate new
                with torch.no_grad():
                    # Note that we have now changed the internal state of the model in the lifecycle of Trainer.
                    # This is why we cache the original state via in_training
                    # model.eval()
                    unwrapped_model.eval()
                    if self.generation_of_hard_negatives_has_begun and \
                            random.random() < self.generation_probability_during_training:
                        # Use the prompt_input_ids directly
                        prompt_ids = prompt_input_ids[i]
                        prompt_attention = prompt_attention_mask[i]

                        # Remove padding
                        prompt_length = (prompt_attention == 1).sum().item()
                        prompt_ids = prompt_ids[:prompt_length]
                        prompt_attention = prompt_attention[:prompt_length]

                        # Generate
                        generated_outputs = unwrapped_model.generate(
                            input_ids=prompt_ids.unsqueeze(0),
                            attention_mask=prompt_attention.unsqueeze(0),
                            max_new_tokens=self.generation_config.get("max_new_tokens",
                                                                      sdm_network_constants.MAX_NEW_TOKENS_FOR_GENERATION),
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False  # This avoids the caching/gradient checkpointing conflict
                        )

                        # Decode
                        generated_tokens = generated_outputs[0][len(prompt_ids):]
                        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                        generation_matches_reference = \
                            sdm_network_finetune_utils_trainer_reward_assignment.is_exact_match_excluding_boundary_whitespace(
                                reference_document_string=
                                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                                    sentences=[reference_document_string],
                                    verifications=[True])[0],
                                generated_sentence=generated_text)
                        # We also need to ignore the case in which the correct sentence is generated but the
                        # classification is wrong, since this is already handled by the true positive teacher-forcing.
                        generation_matches_reference_text_but_classification_is_wrong = \
                            sdm_network_finetune_utils_trainer_reward_assignment.is_exact_match_excluding_boundary_whitespace(
                                reference_document_string=
                                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                                    sentences=[reference_document_string],
                                    verifications=[False])[0],
                                generated_sentence=generated_text)

                        if not generation_matches_reference and \
                                not generation_matches_reference_text_but_classification_is_wrong:
                            # The response is not an exact match of the text and/or formatting,
                            # so we ensure "<verified>No</verified>\n\n" occurs at the end
                            response, _ = \
                                sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                                    generated_text=generated_text)
                            error_candidates.append(response)
                            hard_negatives.append(response)

                    # shuffle error candidates in place
                    random.shuffle(error_candidates)
                    assistant_text = error_candidates[0]
                    # print(f"Hard negative prompt: {prompt_text}\n\nAssistant: {assistant_text}\n\n----------")
                    updated_input_dict = self.get_ids_with_padding(prompt_text, assistant_text)
                    # print(f"updated_dict: {assistant_text}, {updated_input_dict}")
                    updated_features.append(updated_input_dict)

                    # Store for saving
                    if in_training:
                        progress_reference_key = self.state.epoch if hasattr(self, 'state') else 0
                    else:
                        progress_reference_key = self.state.global_step if hasattr(self, 'state') else 0

                    all_generations_pointer[document_id] = {
                        sdm_network_constants.REEXPRESS_ID_KEY: document_id,
                        # List of incorrect sentences after deduplication. This can be an empty list.
                        sdm_network_constants.HARD_NEGATIVES_KEY: list(set(hard_negatives)),
                        # epoch will be the last epoch/step for which this document id was updated with a hard negative:
                        sdm_network_constants.SDM_NETWORK_FINETUNING_EPOCH_KEY: progress_reference_key,
                    }

        assert len(updated_features) == batch_size, \
            f"Invariant violated: updated_features has {len(updated_features)} items but batch_size is {batch_size}"
        # update padding:
        updated_features_on_cpu = self.re_collate(updated_features)

        if self.use_cross_entropy:
            # Standard forward pass and loss computation
            if in_training:
                model.train()
            else:
                model.eval()
            outputs = model(
                input_ids=updated_features_on_cpu["input_ids"].to(device),
                attention_mask=updated_features_on_cpu["attention_mask"].to(device),
                labels=updated_features_on_cpu["labels"].to(device),
                return_dict=True
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss
        else:
            assert False

    def _init_reference_model(self, model):
        """Initialize reference model as a frozen copy of the initial model"""
        if self.reference_model_needs_init:
            logger.info("Initializing reference model for DPO...")
            # Get the unwrapped model
            unwrapped = model.module if hasattr(model, 'module') else model
            # Create a deep copy
            self.reference_model = deepcopy(unwrapped)
            # Freeze all parameters
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            # Move to same device as model
            self.reference_model = self.reference_model.to(unwrapped.device)
            self.reference_model_needs_init = False
            logger.info("Reference model initialized and frozen")

    def _compute_log_probs(self, model, input_ids, attention_mask, labels, return_outputs=False):
        """
        Compute log probabilities for the given inputs under the model.

        Args:
            return_outputs: If True, return (log_probs, outputs), otherwise just log_probs

        Returns:
            log_probs: tensor of shape (batch_size,) containing sum of log probs for each example
            outputs: (optional) model outputs object if return_outputs=True
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            return_dict=True
        )

        logits = outputs.logits
        # Shift for causal LM
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1)

        # Gather log probs for actual labels
        batch_size, seq_len, vocab_size = log_probs.shape
        log_probs_flat = log_probs.view(-1, vocab_size)
        labels_flat = shifted_labels.view(-1)

        valid_mask = (labels_flat != -100)
        labels_for_gather = labels_flat.clone()
        labels_for_gather[~valid_mask] = 0  # Replace -100 with safe value

        # Now safe to gather
        token_log_probs = torch.gather(
            log_probs_flat,
            dim=1,
            index=labels_for_gather.unsqueeze(1)
        ).squeeze(1)

        # Reshape back to batch
        token_log_probs = token_log_probs.view(batch_size, seq_len)

        # Mask out padding tokens (label = -100)
        mask = (shifted_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # Sum over sequence length
        sequence_log_probs = token_log_probs.sum(dim=1)

        if return_outputs:
            return sequence_log_probs, outputs
        return sequence_log_probs

    def _compute_baseline_dpo_loss_with_train_time_generations(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Note: This is staged for removal, but we keep it for the moment as an example for how (at a high-level)
        # alternative baseline losses can be added. The difference from existing implementations is that this
        # incorporates online generations. However, this is from an earlier version of the code, so the
        # code should be checked again before using as a baseline. In practice, we are interested in more foundational
        # changes to the behavior of the representations of the model,
        # so DPO-style losses are a priori not typically the right
        # modeling choices, as opposed to supervised token-level losses.
        # As a side note, if such a loss were desired, in principle a
        # change of base would also be possible with the sigmoid function in an analogous manner to that of the
        # softmax. (The same is true for tanh and other activations with e^x.)
        """
        Compute DPO loss using generated hard negatives as rejected examples.

        DPO Loss (standard):
            L = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))

        DPO Loss (no reference model):
            L = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x))))

        Where:
            y_w = chosen/preferred response (ground truth)
            y_l = rejected response (hard negative)
            β = temperature parameter
            σ = sigmoid function
        """
        assert self.use_dpo, "This function should only be called when use_dpo=True"

        # Initialize reference model if needed (only once)
        if self.reference_model_needs_init:
            self._init_reference_model(model)

        in_training = model.training
        if in_training:
            all_generations_pointer = self.all_generations
            self.generation_of_hard_negatives_has_begun = self.state.epoch > 1.0
        else:
            all_generations_pointer = self.all_generations_validation

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        document_ids = inputs["document_id"]
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]

        default_negatives = inputs[sdm_network_constants.DEFAULT_NEGATIVE_KEY]
        reference_document_strings = inputs[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]
        prompt_texts = inputs[sdm_network_constants.CACHED_PROMPT_KEY]

        batch_size = input_ids.shape[0]
        device = input_ids.device

        chosen_features = []  # Positive examples (ground truth)
        rejected_features = []  # Negative examples (errors)

        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Collect chosen and rejected examples
        for i in range(batch_size):
            document_id = document_ids[i]
            default_negative = default_negatives[i]
            reference_document_string = reference_document_strings[i]
            prompt_text = prompt_texts[i]

            # Chosen example (ground truth) - always included
            chosen_features.append({
                "input_ids": input_ids[i].cpu(),
                "attention_mask": attention_mask[i].cpu(),
                "labels": labels[i].cpu()
            })

            # Rejected example (error)
            error_candidates = [default_negative]
            hard_negatives = []

            if document_id in all_generations_pointer:
                saved_data = all_generations_pointer[document_id]
                error_candidates.extend(saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, []))
                hard_negatives = saved_data.get(sdm_network_constants.HARD_NEGATIVES_KEY, [])

            # Generate new hard negative if conditions are met
            with torch.no_grad():
                unwrapped_model.eval()
                if self.generation_of_hard_negatives_has_begun and \
                        random.random() < self.generation_probability_during_training:
                    # Use the prompt_input_ids directly
                    prompt_ids = prompt_input_ids[i]
                    prompt_attention = prompt_attention_mask[i]

                    # Remove padding
                    prompt_length = (prompt_attention == 1).sum().item()
                    prompt_ids = prompt_ids[:prompt_length]
                    prompt_attention = prompt_attention[:prompt_length]

                    # Generate
                    generated_outputs = unwrapped_model.generate(
                        input_ids=prompt_ids.unsqueeze(0),
                        attention_mask=prompt_attention.unsqueeze(0),
                        max_new_tokens=self.generation_config.get("max_new_tokens",
                                                                  sdm_network_constants.MAX_NEW_TOKENS_FOR_GENERATION),
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False  # This avoids the caching/gradient checkpointing conflict
                    )

                    # Decode
                    generated_tokens = generated_outputs[0][len(prompt_ids):]
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                    generation_matches_reference = \
                        sdm_network_finetune_utils_trainer_reward_assignment.is_exact_match_excluding_boundary_whitespace(
                            reference_document_string=
                            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                                sentences=[reference_document_string],
                                verifications=[True])[0],
                            generated_sentence=generated_text)
                    # We also need to ignore the case in which the correct sentence is generated but the
                    # classification is wrong, since this is already handled by the true positive teacher-forcing.
                    generation_matches_reference_text_but_classification_is_wrong = \
                        sdm_network_finetune_utils_trainer_reward_assignment.is_exact_match_excluding_boundary_whitespace(
                            reference_document_string=
                            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_output(
                                sentences=[reference_document_string],
                                verifications=[False])[0],
                            generated_sentence=generated_text)

                    if not generation_matches_reference and \
                            not generation_matches_reference_text_but_classification_is_wrong:
                        # The response is not an exact match of the text and/or formatting,
                        # so we ensure "<verified>No</verified>\n\n" occurs at the end
                        response, _ = \
                            sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                                generated_text=generated_text)
                        error_candidates.append(response)
                        hard_negatives.append(response)

                # Select rejected example
                random.shuffle(error_candidates)
                rejected_text = error_candidates[0]

                rejected_dict = self.get_ids_with_padding_but_no_negative_mask(prompt_text, rejected_text)
                rejected_features.append(rejected_dict)

                # Store generations
                if in_training:
                    progress_reference_key = self.state.epoch if hasattr(self, 'state') else 0
                else:
                    progress_reference_key = self.state.global_step if hasattr(self, 'state') else 0

                all_generations_pointer[document_id] = {
                    sdm_network_constants.REEXPRESS_ID_KEY: document_id,
                    sdm_network_constants.HARD_NEGATIVES_KEY: list(set(hard_negatives)),
                    sdm_network_constants.SDM_NETWORK_FINETUNING_EPOCH_KEY: progress_reference_key,
                }

        # Collate chosen and rejected batches
        # chosen_batch = self.re_collate(chosen_features)
        # rejected_batch = self.re_collate(rejected_features)

        all_features = chosen_features + rejected_features
        all_batch = self.re_collate(all_features)

        # Set model to correct mode
        if in_training:
            model.train()
        else:
            model.eval()

        # Split into chosen and rejected
        concatenated_input_ids = all_batch["input_ids"].to(device)
        concatenated_attention_mask = all_batch["attention_mask"].to(device)
        concatenated_labels = all_batch["labels"].to(device)

        # Single forward pass for both chosen and rejected
        concatenated_log_probs, concatenated_outputs = self._compute_log_probs(
            model,
            concatenated_input_ids,
            concatenated_attention_mask,
            concatenated_labels,
            return_outputs=True
        )

        # Split results
        chosen_log_probs = concatenated_log_probs[:batch_size]
        rejected_log_probs = concatenated_log_probs[batch_size:]

        chosen_outputs = concatenated_outputs[:batch_size]

        # Compute log probabilities under reference model if needed
        if not self.dpo_no_reference_model:
            with torch.no_grad():
                ref_concatenated_log_probs = self._compute_log_probs(
                    self.reference_model,
                    concatenated_input_ids,
                    concatenated_attention_mask,
                    concatenated_labels,
                    return_outputs=False
                )
                ref_chosen_log_probs = ref_concatenated_log_probs[:batch_size]
                ref_rejected_log_probs = ref_concatenated_log_probs[batch_size:]

            # DPO loss with reference model
            logits_diff = (chosen_log_probs - rejected_log_probs) - \
                          (ref_chosen_log_probs - ref_rejected_log_probs)
        else:
            # DPO loss without reference model (simplified)
            logits_diff = chosen_log_probs - rejected_log_probs

        # Apply temperature and compute loss
        # Loss = -log(sigmoid(β * logits_diff))
        # This is equivalent to binary cross entropy with label=1
        loss = -F.logsigmoid(self.dpo_beta * logits_diff).mean()

        if return_outputs:
            # Return the outputs from the chosen examples forward pass
            # This matches the pattern used in cross-entropy and SDM losses
            return loss, chosen_outputs
        else:
            return loss

    def get_ids_with_padding_but_no_negative_mask(self, prompt_text, assistant_response):
        conv = sdm_network_constants.get_conv(prompt_text)
        conv.append({"role": "assistant", "content": assistant_response})

        # Tokenize the conversation
        encoding = self.tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            max_length=self.model_max_length,
            padding=False, #"max_length",
            truncation=True,
            return_dict=True,
            add_generation_prompt=False  # We have the assistant response
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # Note the use of disable_negative_masking=True
        labels = \
            sdm_network_constants.create_masked_labels(self.mask_prefix, self.tokenizer, self.mask_until_pattern,
                                                       input_ids.tolist(), assistant_response,
                                                       disable_negative_masking=True)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
