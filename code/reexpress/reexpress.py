# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import torch.nn as nn

import numpy as np
import argparse
import copy
from pathlib import Path
import math

from collections import defaultdict

import codecs
import time

import json
import copy
import os

import utils_train_main
import utils_classification
import uncertainty_statistics
import uuid
import constants
import utils_model
import sdm_model
import utils_gen
import utils_train_main_gen_ai_router
import utils_preprocess

from mlx_lm import load

import data_validator
import utils_train_iterative_main
import utils_train_main_gen_ai_controller

import utils_test
import utils_update


def main():
    parser = argparse.ArgumentParser(description="-----[Train and eval sdm estimators and networks]-----")
    # Note that not all options are currently implemented and/or used in this research codebase. See the
    # Tutorials for replicating the paper's experiments, rather than the argument help messages and in-line comments
    # in the code, which may not reflect the currently released research codebase version.
    parser.add_argument("--input_training_set_file", default="",
                        help=".jsonl format")
    parser.add_argument("--input_calibration_set_file", default="",
                        help=".jsonl format")
    parser.add_argument("--input_eval_set_file", default="",
                        help=".jsonl format")

    parser.add_argument("--class_size", default=2, type=int, help="class_size")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--use_gpu", default=False, action='store_true',
                        help="TODO: not currently implemented. The code currently runs on cpu.")
    parser.add_argument("--use_mps", default=False, action='store_true',
                        help="")

    parser.add_argument("--use_json_input_instead_of_torch_file", default=False, action='store_true',
                        help="use_json_input_instead_of_torch_file")
    parser.add_argument("--epoch", default=20, type=int, help="number of max epoch")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size during training "
                                                                   "(excluding the rescaler, which always uses a batch size of 1)")
    parser.add_argument("--eval_batch_size", default=50, type=int, help="batch size during evaluation; "
                                                                        "typically can be larger than that used during training")
    parser.add_argument("--learning_rate", default=0.00001, type=float, help="learning rate")

    parser.add_argument("--alpha", default=constants.defaultCdfAlpha, type=float, help="alpha in (0,1), typically 0.95")
    parser.add_argument("--maxQAvailableFromIndexer", default=constants.maxQAvailableFromIndexer, type=int,
                        help="max q considered")
    parser.add_argument("--use_training_set_max_label_size_as_max_q", default=False, action='store_true',
                        help="use_training_set_max_label_size_as_max_q")

    parser.add_argument("--eval_only", default=False, action='store_true', help="eval_only")

    parser.add_argument("--model_dir", default="",
                        help="model_dir")

    parser.add_argument("--use_embeddings", default=False, action='store_true', help="")
    parser.add_argument("--concat_embeddings_to_attributes", default=False, action='store_true', help="")
    parser.add_argument("--output_eval_meta_structure_file_prefix", default="",
                        help=".pickle format (TEMPORARY)")

    parser.add_argument("--number_of_random_shuffles", default=20, type=int,
                        help="number of random shuffles of D_tr and D_ca, each of which is associated with a new"
                             " f(x) := o of g of h(x), where h(x) is held frozen")
    parser.add_argument("--do_not_shuffle_data", default=False, action='store_true',
                        help="In this case, the data is not shuffled. If --number_of_random_shuffles > 1, "
                             "iterations can still occur (to assess variation in learning, but the data stays fixed. "
                             "Generally speaking, it's recommended to shuffle the data.")
    parser.add_argument("--export_reexpression_attributes", default=False, action='store_true', help="")
    parser.add_argument("--output_file_for_resaving_eval_file_with_reexpression_attributes", default="",
                        help="Only applicable if --export_reexpression_attributes")
    parser.add_argument("--is_training_support", default=False, action='store_true',
                        help="Include this flag if the eval set is the training set. "
                             "This ignores the first match when calculating uncertainty, under the assumption that "
                             "the first match is identity.")
    parser.add_argument("--recalibrate", default=False, action='store_true', help="")
    parser.add_argument("--show_graph", default=False, action='store_true', help="")
    parser.add_argument("--warm_up_epochs", default=0, type=int, help="Epochs of initial training with standard "
                                                                       "softmax and CrossEntropy loss.")
    parser.add_argument("--use_balanced_accuracy", default=False, action='store_true',
                        help="Training is determined by highest balanced accuracy. If not provided, the max "
                             "balanced median q is used.")
    parser.add_argument("--train_rescaler", default=False, action='store_true',
                        help="If training exited without training the rescaler, use this option. "
                             "Remember to use --load_train_and_calibration_from_best_iteration_data_dir if the "
                             "model was trained with shuffling. (Alternatively, just point "
                             "to the best_iteration_data directory.)")
    parser.add_argument("--only_update_rescaler_alpha",
                        default=False, action='store_true', help="Only used if --train_rescaler. If provided, "
                                                                 "the min bin for achieving the updated alpha "
                                                                 "value is also re-determined; however, the "
                                                                 "rescaler weights are not updated.")
    parser.add_argument("--load_train_and_calibration_from_best_iteration_data_dir",
                        default=False, action='store_true', help="")
    parser.add_argument("--model_rescaler_training_max_epochs", default=1000, type=int, help="")
    parser.add_argument("--model_rescaler_training_learning_rate", default=0.0001, type=float, help="")
    parser.add_argument("--do_not_normalize_input_embeddings",
                        default=False, action='store_true',
                        help="Typically only use this if you have already standardized/normalized the embeddings. "
                             "Our default approach is to mean center based on the training set embeddings. This is "
                             "a global normalization that is applied in the forward of sdm_model."
                             "Note that we do not apply this normalization over the token-level input to the final "
                             "linear layer of the generative AI LLM, instead using the original normalization approach "
                             "(e.g., RMSNorm) used in initial training.")
    parser.add_argument("--continue_training",
                        default=False, action='store_true', help="")
    parser.add_argument("--do_not_resave_shuffled_data",
                        default=False, action='store_true', help="")
    parser.add_argument("--exemplar_vector_dimension", default=constants.keyModelDimension, type=int, help="")

    parser.add_argument("--gen_ai_model_path", default="",
                        help="")
    parser.add_argument("--gen_ai_model_lm_head_weights_file", default="",
                        help="")
    parser.add_argument("--max_length", default=500, type=int,
                        help="")
    parser.add_argument("--gen_ai_vocab", default=32064, type=int,
                        help="")
    parser.add_argument("--is_gen_ai",
                        default=False, action='store_true', help="")
    parser.add_argument("--router_warm_up_epochs", default=0, type=int,
                        help="")
    parser.add_argument("--top_logits_k", default=constants.top_logits_k, type=int,
                        help="Top logits")
    parser.add_argument("--composition_attributes_size", default=0, type=int,
                        help="Use with --init_gen_ai_model")

    # Options to one time cache the sdm-genai input embeddings:
    parser.add_argument("--cache_embeddings_for_classification_with_force_decoded_generation__document_level",
                        default=False, action='store_true', help="")
    parser.add_argument("--cache_embeddings_for_classification_with_generation__document_level",
                        default=False, action='store_true', help="")
    parser.add_argument("--only_cache_eval",
                        default=False, action='store_true', help="If provided, only --input_eval_set_file is cached.")
    parser.add_argument("--cache_directory", default="", help="Directory to save the cached embeddings. The same "
                                                              "filename as the input is used.")
    parser.add_argument("--cache_using_existing_sdm_model",
                        default=False, action='store_true', help="")

    # --taskCategory is not currently used, since the values are pre-assigned in the input JSON lines files.
    # parser.add_argument("--taskCategory", default=0, type=int,
    #                     help="int; 0 for sentiment; 1 for factcheck.")
    parser.add_argument("--llmType", default=0, type=int,
                        help="int; 0 for phi 3.5; 1 for phi4 (not yet fully implemented)")

    parser.add_argument("--init_gen_ai_model",
                        default=False, action='store_true',
                        help="This assumes the input data is cached with "
                             "--cache_embeddings_for_classification_with_force_decoded_generation__document_level.")
    parser.add_argument("--train_gen_ai_model",
                        default=False, action='store_true',
                        help="This assumes the process run by --init_gen_ai_model has been completed and "
                             "the input files include the cached results from "
                             "--cache_embeddings_for_classification_with_generation__document_level "
                             "for calculating the initial q values.")
    parser.add_argument("--generation_directory", default="generation_dir",
                        help="Directory to save the generated output. The same filename as the input is used. "
                             "When training, 'epoch_X_' is appended as a prefix.")
    parser.add_argument("--reset_gen_ai_model_weights",
                        default=False, action='store_true',
                        help="The LLM weights are reset to the original pre-trained values, "
                             "after which the script is exited. The model must be previously initialized.")
    parser.add_argument("--gen_ai_training_min_beta", default=0.01, type=float,
                        help="Must be >= 0 and <= --gen_ai_training_max_beta")
    parser.add_argument("--gen_ai_training_max_beta", default=1.0, type=float,
                        help="Must be >= --gen_ai_training_min_beta")
    parser.add_argument("--label_error_file", default="",
                        help="If provided, possible label annotation errors are saved, sorted by the LOWER predictive "
                             "probability, where the subset is those that are valid index-conditional predictions.")
    parser.add_argument("--valid_index_conditional_file", default="",
                        help="If provided, instances with valid index-conditional predictions are saved, "
                             "sorted by the LOWER predictive probability.")
    parser.add_argument("--eval_gen_ai", default=False, action='store_true', help="eval_gen_ai")
    parser.add_argument("--update_support_set_with_eval_data", default=False, action='store_true',
                        help="update_support_set_with_eval_data")

    options = parser.parse_args()

    # Setting seed
    torch.manual_seed(options.seed_value)
    np.random.seed(options.seed_value)
    # random.seed(options.seed_value)
    rng = np.random.default_rng(seed=options.seed_value)

    if options.eval_gen_ai:
        assert options.eval_only, f"Currently we assume generation-decoded gen AI evaluation is accompanied with " \
                                  f"--eval_only, which will also eval the force-decoded classifier."

    print("Currently unlabeled and OOD labels are not supported in training and calibration (i.e., -1 and -99, "
          "respectively, in Reexpress one). We assume at least two "
          "labels per class (but typically you will want on the order of 1000's per class).")
    if options.use_gpu:
        print("Currently, NVIDIA gpu is not supported. Exiting.")
        exit()
        # main_device = torch.device("cuda:0")
    elif options.use_mps:  # mps is Apple Silicon
        main_device = torch.device("mps")
    else:
        # cpu still assumes mps for the sdm network, which uses mlx for generation
        main_device = torch.device("cpu")

    if options.is_gen_ai:
        gen_ai_model, tokenizer = load(options.gen_ai_model_path)
    else:
        gen_ai_model, tokenizer = None, None

    if options.reset_gen_ai_model_weights:
        assert options.is_gen_ai
        model = utils_model.load_model_torch(options.model_dir, torch.device("cpu")).to(main_device)
        gen_ai_model_lm_head_weights = \
            utils_gen.get_gen_ai_model_lm_head_weights_file(options.gen_ai_model_lm_head_weights_file)
        model.reset_llm_weights(gen_ai_model_lm_head_weights)
        utils_model.save_model(model, options.model_dir)
        print(f"Model saved with original LLM weights to {options.model_dir}")
        # re-save for mlx generation:
        utils_model.save_llm_weights_for_mlx_generation(options, model, save_as_final_llm_weights=True)
        print(f"LLM weight reset complete. Exiting.")
        exit()
    # assert options.taskCategory in utils_gen.taskCategories
    assert options.llmType in utils_gen.llmTypes
    # taskCategory = options.taskCategory
    llmType = options.llmType
    if int(options.cache_embeddings_for_classification_with_force_decoded_generation__document_level) + \
        int(options.cache_embeddings_for_classification_with_generation__document_level) == 1:
        # print(f"taskCategory: {utils_gen.taskCategories._fields[taskCategory]}")
        print(f"llmType: {utils_gen.llmTypes._fields[llmType]}")
        if options.cache_embeddings_for_classification_with_force_decoded_generation__document_level:
            modelCategory = utils_gen.modelCategories.classification_with_force_decoded_generation__document_level
        elif options.cache_embeddings_for_classification_with_generation__document_level:
            modelCategory = utils_gen.modelCategories.classification_with_generation__document_level
        else:
            assert False
        if options.cache_using_existing_sdm_model:
            assert False, "Not implemented"
            model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)
        else:
            model = None
        utils_gen.cache_embeddings_for_classification(options, gen_ai_model,
                                                      tokenizer, modelCategory,
                                                      None, #taskCategory,
                                                      llmType,
                                                      model=model)
        print("Caching completed. Exiting.")
        exit()

    if not options.train_gen_ai_model:
        utils_train_iterative_main.train_iterative_main(options, rng,
                                                        taskCategory=None, #taskCategory,
                                                        llmType=llmType,
                                                        gen_ai_model=gen_ai_model,
                                                        tokenizer=tokenizer,
                                                        main_device=main_device)

    if options.train_gen_ai_model and not options.eval_only and not options.train_rescaler:
        utils_train_main_gen_ai_controller.train_genai_controller(options, rng,
                                                        taskCategory=None, #taskCategory,
                                                        llmType=llmType,
                                                        gen_ai_model=gen_ai_model,
                                                        tokenizer=tokenizer,
                                                        main_device=main_device)
        print(f"LLM training complete. Exiting.")
        exit()

    if options.train_rescaler:
        print(f"Reloading best model for training the model re-scaler")
        best_model_min_valid_qbin_for_class_conditional_accuracy, \
            best_model_predicted_class_to_bin_to_median_output_magnitude = \
            utils_train_main.train_rescaler(options, model_dir=options.model_dir)
        global_uncertainty_statistics = utils_model.load_global_uncertainty_statistics_from_disk(options.model_dir)
        print(f"updating global stats")
        global_uncertainty_statistics.update_min_valid_qbin(
            min_valid_qbin=best_model_min_valid_qbin_for_class_conditional_accuracy)
        global_uncertainty_statistics.update_output_magnitudes_for_bin(
            best_model_predicted_class_to_bin_to_median_output_magnitude)

    utils_test.test(options, main_device)
    if options.update_support_set_with_eval_data:
        utils_update.batch_support_update(options, main_device)

    if options.eval_gen_ai:
        utils_test.test_gen_ai(options, main_device, gen_ai_model, tokenizer, options.input_eval_set_file, llmType)


if __name__ == "__main__":
    main()

