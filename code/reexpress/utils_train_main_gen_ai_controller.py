# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import torch.nn as nn

import numpy as np
import argparse
import copy
from pathlib import Path
import math

from collections import defaultdict


import time
import os

import utils_model
# import utils_gen
import utils_train_main_gen_ai_router


def train_genai_controller(options, rng, taskCategory=None, llmType=None, gen_ai_model=None, tokenizer=None, main_device=None):
    import utils_gen
    start_time = time.time()
    best_shuffle_index = 0
    max_calibration_balanced_accuracy = 0
    max_calibration_balanced_accuracy_shuffle_iteration = -1

    max_calibration_balanced_median_q = 0
    max_calibration_balanced_median_q_shuffle_iteration = -1

    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu")).to(main_device)
    print(f"Continuing training from the model stored in {options.model_dir}")

    train_file = options.input_training_set_file
    calibration_file = options.input_calibration_set_file
    if options.load_train_and_calibration_from_best_iteration_data_dir:
        best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
        best_iteration_data_dir = str(best_iteration_data_path.as_posix())

        train_file = os.path.join(best_iteration_data_dir, "train.jsonl")
        calibration_file = os.path.join(best_iteration_data_dir, "calibration.jsonl")

    # One-time initial force-decoded generation, which we then cache for training:
    # (Note that model=None since it is not needed for generation pre-processing in this case since we are
    # force-decoding over the input and training is only updating the LLM's final linear layer. If the lower layers are
    # updated, then use model=model for generation, since changing the lower layers will change the embeddings.
    # However, that case is different in that generation should occur in the loop of the epoch (i.e., as is standard
    # with sequence-model training), and thus this type of one-time caching for efficiency (and on-device training)
    # is not applicable.
    # Full [accepted-distribution-] model parameter updating --- or via
    # LoRA adaptors --- is not currently implemented in this code base, but it follows directly from this example.)
    train_meta_data, _ = \
        utils_gen.get_metadata_lines_for_gen(options, gen_ai_model, tokenizer, options.max_length, train_file,
                                             calculate_summary_stats=False, is_training=True,
                                             taskCategory=None,
                                             modelCategory=utils_gen.modelCategories.generation_force_decode__token_level,
                                             top_logits_k=options.top_logits_k,
                                             model=None,
                                             return_text=False,
                                             llmType=llmType,
                                             eval_label="TRAIN (initial force-decoded generation)"
                                             )

    train_token_level_embeddings = train_meta_data["embeddings"].to(main_device)
    print(f"train_token_level_embeddings.shape: {train_token_level_embeddings.shape}")

    for class_label in range(options.class_size):
        print(
            f"Training class {class_label}: {len([x for x in train_meta_data['labels'] if x == class_label])} documents")

    train_shifted_token_labels = torch.tensor(train_meta_data["shifted_token_labels"]).to(main_device)
    assert train_token_level_embeddings.shape[0] == train_shifted_token_labels.shape[
        0], f"{train_token_level_embeddings.shape[0]}, {train_shifted_token_labels.shape[0]}"

    utils_train_main_gen_ai_router.train(options, train_file, calibration_file, gen_ai_model, tokenizer,
                                         llmType, train_token_level_embeddings=train_token_level_embeddings,
                                         train_shifted_token_labels=train_shifted_token_labels,
                                         train_token_level_uuids=train_meta_data["uuids"],
                                         use_balanced_accuracy=options.use_balanced_accuracy,
                                         main_device=main_device,
                                         model_dir=options.model_dir, model=model)
    """
    In principle, on could train with an iterative shuffle, as with utils_train_iterative_main.train_iterative_main().
    However, since 
    the uncertainty estimates are derived from the iterated sdm network, we already capture the meaningful
    signals for final decision-making over the output.
    """