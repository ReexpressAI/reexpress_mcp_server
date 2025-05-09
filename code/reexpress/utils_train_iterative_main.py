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


def train_iterative_main(options, rng, taskCategory=None, llmType=None, gen_ai_model=None, tokenizer=None, main_device=None):
    start_time = time.time()

    global_uncertainty_statistics = \
        uncertainty_statistics.UncertaintyStatistics(
            globalUncertaintyModelUUID=str(uuid.uuid4()),
            numberOfClasses=options.class_size,
            min_valid_qbin_across_iterations=None,
            predicted_class_to_bin_to_median_output_magnitude_of_iteration=None,
            cauchy_quantile=options.alpha
        )

    if not options.eval_only and not options.train_rescaler:
        best_shuffle_index = 0
        max_calibration_balanced_accuracy = 0
        max_calibration_balanced_accuracy_shuffle_iteration = -1

        max_calibration_balanced_median_q = 0
        max_calibration_balanced_median_q_shuffle_iteration = -1

        assert options.number_of_random_shuffles >= 0
        for shuffle_index in range(max(options.number_of_random_shuffles, 1)):
            if options.continue_training:
                model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"))
                print(f"Continuing training from the model stored in {options.model_dir}")
            else:
                model = None
            path = Path(options.model_dir, f"{shuffle_index}")
            path.mkdir(parents=False, exist_ok=True)
            shuffle_index_model_dir = str(path.as_posix())

            if not options.do_not_shuffle_data:
                best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
                best_iteration_data_path.mkdir(parents=False, exist_ok=True)
                best_iteration_data_dir = str(best_iteration_data_path.as_posix())

                print(f"Current D_tr, D_ca shuffle index {shuffle_index}")
                # Generally speaking, the training file should have balanced labels, but we do not currently enforce this when
                # randomly shuffling. If your dataset is unbalanced, currently you will need to manually shuffle.
                all_data = utils_preprocess.get_data(options.input_training_set_file)
                all_data.extend(utils_preprocess.get_data(options.input_calibration_set_file))
                rng.shuffle(all_data)
                # this gets resaved if best epoch
                train_data_json_list = all_data[0:len(all_data)//2]
                calibration_data_json_list = all_data[len(all_data)//2:]
                train_meta_data, training_embedding_summary_stats = utils_preprocess.get_metadata_lines_from_json_list(options, train_data_json_list,
                                                                    reduce=False,
                                                                    use_embeddings=options.use_embeddings,
                                                                    concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                                    calculate_summary_stats=True, is_training=True)
                calibration_meta_data, _ = utils_preprocess.get_metadata_lines_from_json_list(options, calibration_data_json_list,
                                                                          use_embeddings=options.use_embeddings,
                                                                          concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                                          calculate_summary_stats=False, is_training=False)

            else:
                train_file = options.input_training_set_file
                calibration_file = options.input_calibration_set_file
                if options.load_train_and_calibration_from_best_iteration_data_dir:
                    best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
                    best_iteration_data_dir = str(best_iteration_data_path.as_posix())

                    train_file = os.path.join(best_iteration_data_dir, "train.jsonl")
                    calibration_file = os.path.join(best_iteration_data_dir, "calibration.jsonl")

                train_meta_data, training_embedding_summary_stats = utils_preprocess.get_metadata_lines(options, train_file,
                                                     reduce=False,
                                                     use_embeddings=options.use_embeddings,
                                                     concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                     calculate_summary_stats=True, is_training=True)
                calibration_meta_data, _ = utils_preprocess.get_metadata_lines(options, calibration_file,
                                                           use_embeddings=options.use_embeddings,
                                                           concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                           calculate_summary_stats=False, is_training=False)

            train_embeddings = train_meta_data["embeddings"].to(main_device)
            calibration_embeddings = calibration_meta_data["embeddings"].to(main_device)
            train_labels = torch.tensor(train_meta_data["labels"]).to(main_device)
            calibration_labels = torch.tensor(calibration_meta_data["labels"]).to(main_device)

            assert train_embeddings.shape[0] == train_labels.shape[0], f"{train_embeddings.shape[0]}, {train_labels.shape[0]}"
            assert calibration_embeddings.shape[0] == calibration_labels.shape[0], f"{calibration_embeddings.shape[0]}, {calibration_labels.shape[0]}"
            assert train_embeddings.shape[1] == calibration_embeddings.shape[1], f"{train_embeddings.shape[1]}, {calibration_embeddings.shape[1]}"

            print(f"train_embeddings.shape: {train_embeddings.shape}")
            print(f"calibration_embeddings.shape: {calibration_embeddings.shape}")

            for class_label in range(options.class_size):
                print(f"Training class {class_label}: {len([x for x in train_meta_data['labels'] if x == class_label])} documents")

            maxQAvailableFromIndexer = options.maxQAvailableFromIndexer
            if options.use_training_set_max_label_size_as_max_q:
                max_training_set_label_cardinality = 0
                label_set_cardinality = {}
                for label in range(options.class_size):
                    label_set_cardinality[label] = 0
                for label in train_labels:
                    if data_validator.isKnownValidLabel(label=label, numberOfClasses=options.class_size):
                        label = label.item()
                        label_set_cardinality[label] += 1
                for label in range(options.class_size):
                    print(f"Training label {label} support cardinality: {label_set_cardinality[label]}")
                    if label_set_cardinality[label] > max_training_set_label_cardinality:
                        max_training_set_label_cardinality = label_set_cardinality[label]
                maxQAvailableFromIndexer = max_training_set_label_cardinality
            print(f"Considering a max q value up to {maxQAvailableFromIndexer}")
            if options.is_gen_ai:
                if options.init_gen_ai_model:
                    if llmType == utils_gen.llmTypes.phiThreePointFive:
                        train_meta_data["embedding_size"] = 3072
                        train_meta_data["global_embedding_size"] = 3072
                        train_meta_data["composition_attributes_size"] = options.composition_attributes_size #options.top_logits_k * 2 * 2
                    else:
                        assert False, "Not implemented"
                gen_ai_model_lm_head_weights = \
                    utils_gen.get_gen_ai_model_lm_head_weights_file(options.gen_ai_model_lm_head_weights_file)
            else:
                gen_ai_model_lm_head_weights = None
            model_params = {"version": constants.ProgramIdentifiers_version,
                            "uncertaintyModelUUID": str(uuid.uuid4()),
                            "numberOfClasses": options.class_size,
                            "embedding_size": train_meta_data["embedding_size"] if "embedding_size" in train_meta_data else train_embeddings.shape[1],
                            "train_labels": train_labels.cpu().detach().numpy(),
                            "train_predicted_labels": None,
                            "train_uuids": train_meta_data["uuids"],
                            "cdfThresholdTolerance": constants.defaultCdfThresholdTolerance,
                            "exemplar_vector_dimension": options.exemplar_vector_dimension,
                            "trueClass_To_dCDF": None,
                            "trueClass_To_qCumulativeSampleSizeArray": None,
                            "trueClass_To_unrescaledOutputCDF": None,
                            "non_odd_thresholds": None,
                            "non_odd_class_conditional_accuracy": 0.0,
                            "alpha": options.alpha,
                            "maxQAvailableFromIndexer": maxQAvailableFromIndexer,
                            "calibration_training_stage": 0,
                            "min_valid_qbin_for_class_conditional_accuracy": np.inf, #constants.min_valid_qbin_for_class_conditional_accuracy,
                            "training_embedding_summary_stats": training_embedding_summary_stats,
                            # the following can all be None at test-time to save memory, if desired:
                            "calibration_labels": calibration_labels,  # torch tensor
                            "calibration_predicted_labels": None,
                            "calibration_uuids": calibration_meta_data["uuids"],
                            "calibration_unrescaled_CDFquantiles": None,
                            "calibration_soft_qbins": None,
                            "calibration_is_ood_indicators": None,
                            "gen_ai_model_lm_head_weights": gen_ai_model_lm_head_weights,
                            "is_gen_ai": options.is_gen_ai,
                            "gen_ai_vocab": options.gen_ai_vocab,
                            "global_embedding_size": train_meta_data["global_embedding_size"] if "global_embedding_size" in train_meta_data else 0,
                            "composition_attributes_size": train_meta_data["composition_attributes_size"] if "composition_attributes_size" in train_meta_data else 0,
                            "top_logits_k": options.top_logits_k,
                            "train_trueClass_To_dCDF": None
                            }
            one_shuffle_index__max_dev_balanced_acc, one_shuffle_index_max_dev_balanced_median_q, \
                one_shuffle_index__min_valid_qbin_for_class_conditional_accuracy, \
                predicted_class_to_bin_to_median_output_magnitude = \
                utils_train_main.train(options, train_embeddings=train_embeddings,
                                       calibration_embeddings=calibration_embeddings,
                                       train_labels=train_labels,
                                       calibration_labels=calibration_labels,
                                       model_params=model_params,
                                       use_balanced_accuracy=options.use_balanced_accuracy, main_device=main_device,
                                       model_dir=shuffle_index_model_dir, model=model)

            global_uncertainty_statistics.update_min_valid_qbin(
                min_valid_qbin=one_shuffle_index__min_valid_qbin_for_class_conditional_accuracy)
            global_uncertainty_statistics.update_output_magnitudes_for_bin(
                predicted_class_to_bin_to_median_output_magnitude)
            if one_shuffle_index__max_dev_balanced_acc >= max_calibration_balanced_accuracy:
                max_calibration_balanced_accuracy = one_shuffle_index__max_dev_balanced_acc
                max_calibration_balanced_accuracy_shuffle_iteration = shuffle_index

            print(f"Max calibration balanced accuracy (used to determine shuffle index: "
                  f"{options.use_balanced_accuracy}) of {max_calibration_balanced_accuracy} at "
                  f"shuffle index {max_calibration_balanced_accuracy_shuffle_iteration}")

            if one_shuffle_index_max_dev_balanced_median_q >= max_calibration_balanced_median_q:
                max_calibration_balanced_median_q = one_shuffle_index_max_dev_balanced_median_q
                max_calibration_balanced_median_q_shuffle_iteration = shuffle_index

            print(f"Max calibration balanced MEDIAN Q (used to determine shuffle index: "
                  f"{not options.use_balanced_accuracy}) of {max_calibration_balanced_median_q} at "
                  f"shuffle index {max_calibration_balanced_median_q_shuffle_iteration}")

            if options.use_balanced_accuracy:
                save_this_shuffle_index = max_calibration_balanced_accuracy_shuffle_iteration == shuffle_index
            else:
                save_this_shuffle_index = max_calibration_balanced_median_q_shuffle_iteration == shuffle_index

            if save_this_shuffle_index:
                # load best epoch (still same shuffle index) in order to re-save to the best iteration directory,
                # which is currently the parent directory:
                best_shuffle_index = shuffle_index
                model = utils_model.load_model_torch(shuffle_index_model_dir, torch.device("cpu"))
                utils_model.save_model(model, options.model_dir)
                print(f"Saved current index ({shuffle_index}) as the best shuffle iteration in the parent directory: {options.model_dir}")

                if not options.do_not_shuffle_data and not options.do_not_resave_shuffled_data:
                    utils_model.save_json_lines(os.path.join(best_iteration_data_dir, "train.jsonl"), train_data_json_list)
                    utils_model.save_json_lines(os.path.join(best_iteration_data_dir, "calibration.jsonl"), calibration_data_json_list)
            # the running global uncertainty statistics are saved in the main directory after every iteration:
            utils_model.save_global_uncertainty_statistics(global_uncertainty_statistics, options.model_dir)

            cumulative_time = time.time() - start_time
            print(f"Cumulative running time: {cumulative_time}")
            print(f"Average running time per shuffle iteration: {cumulative_time/(shuffle_index+1)} out of "
                  f"{shuffle_index+1} iterations.")
        print(f"Best overall shuffle index: {best_shuffle_index}.")