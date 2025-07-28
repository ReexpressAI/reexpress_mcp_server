# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import constants
import uncertainty_statistics

import torch
import torch.nn as nn

import numpy as np
import faiss
# import copy

# import math
import json
import codecs
from os import path
from typing import Optional
from pathlib import Path


def read_json_file(filename_with_path):
    try:
        with open(filename_with_path, 'r') as f:
            json_obj = json.load(f)
        return json_obj
    except:
        return None


def read_jsons_lines_file(filename_with_path):
    json_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list


def save_llm_weights_for_mlx_generation(options, model, save_as_final_llm_weights=True):
    if not save_as_final_llm_weights:
        llm_weights_directory_data_path = Path(options.model_dir, constants.DIRNAME_RUNNING_LLM_WEIGHTS_DIR)
        llm_weights_directory_data_path.mkdir(parents=False, exist_ok=True)
        llm_weights_dir = str(llm_weights_directory_data_path.as_posix())
        print(f"Saving LLM weights to {constants.DIRNAME_RUNNING_LLM_WEIGHTS_DIR} for "
              f"conversion to mlx before running generation. "
              f"Any existing weights will be overwritten.")
    else:
        print(f"Saving LLM weights for conversion to mlx to the main directory before running generation. "
              f"Any existing weights will be overwritten.")
        llm_weights_dir = options.model_dir
    torch.save(model.conv.weight.data, f"{path.join(llm_weights_dir, 'exemplar_conv_weight.pt')}")
    torch.save(model.conv.bias.data, f"{path.join(llm_weights_dir, 'exemplar_conv_bias.pt')}")
    torch.save(model.fc.weight.data, f"{path.join(llm_weights_dir, 'exemplar_fc_weight.pt')}")
    torch.save(model.fc.bias.data, f"{path.join(llm_weights_dir, 'exemplar_fc_bias.pt')}")

    torch.save(model.fc_original.weight.data, f"{path.join(llm_weights_dir, 'ai_fc_original_weight.pt')}")
    torch.save(model.fc_negative.weight.data, f"{path.join(llm_weights_dir, 'ai_fc_negative_weight.pt')}")
    if model.fc_negative.bias:
        torch.save(model.fc_negative.bias.data, f"{path.join(llm_weights_dir, 'ai_fc_negative_bias.pt')}")
    torch.save(model.fc_positive.weight.data, f"{path.join(llm_weights_dir, 'ai_fc_positive_weight.pt')}")
    if model.fc_positive.bias:
        torch.save(model.fc_positive.bias.data, f"{path.join(llm_weights_dir, 'ai_fc_positive_bias.pt')}")


def save_generated_lines(generation_directory, eval_file, generated_lines, uuids, taskCategoryInts,
                         reliability_indicators, probability_of_acceptances,
                         generation_output_label,
                         task_predictions,
                         original_task_labels,
                         correct_task_predictions
                         ):
    try:
        generation_directory_data_path = Path(generation_directory)
        generation_directory_data_path.mkdir(parents=False, exist_ok=True)
        generation_directory_data_path_data_dir = str(generation_directory_data_path.as_posix())

        eval_file_filename = f"{generation_output_label}_{Path(eval_file).name}"
        filename_save_path_as_string = str(Path(generation_directory_data_path_data_dir, eval_file_filename).as_posix())
        json_lines = []
        for document_id, generated_output, taskCategory, reliability_indicator, probability_of_acceptance, \
            task_prediction, original_task_label, correct_task_prediction in \
                zip(uuids, generated_lines, taskCategoryInts, reliability_indicators, probability_of_acceptances,
                    task_predictions,
                    original_task_labels,
                    correct_task_predictions
                    ):
            json_obj = {
                "id": document_id,
                "taskCategory": taskCategory,
                "task_label": original_task_label,
                "task_prediction": task_prediction,
                "correct_task_prediction": correct_task_prediction,
                "reliable_estimate": reliability_indicator,
                "probability_of_acceptance": probability_of_acceptance,
                "generated_output": generated_output
            }
            json_lines.append(json_obj)
        save_json_lines(filename_save_path_as_string, json_lines)
        print(f"Generated output saved to {filename_save_path_as_string}")
    except:
        print(f"Unable to save the generated output to {generation_directory}")


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def normalize_embeddings(embeddings, summary_stats):
    # return (embeddings - summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean]) / \
    #     summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std]
    return SimilarityDistanceMagnitudeCalibrator.normalize_embeddings(
        embeddings, summary_stats)


def get_embedding_summary_stats(embeddings, is_training):
    assert is_training, f"ERROR: This must be the training/support set."
    print(f">>Collecting training set embeddings summary stats<<")
    training_embedding_mean = torch.mean(embeddings).item()
    training_embedding_std = torch.std(embeddings, correction=1).item()

    summary_stats = {
        constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: training_embedding_mean,
        constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: training_embedding_std
    }
    return summary_stats


def save_support_set_updates(model, model_dir):
    """
    Save current state of the support set

    This only saves the current model.support_index, model.train_labels, model.train_predicted_labels,
    model.train_uuids. It is intended as lighter-weight than saving the full model, since user-initiated
    updates can be done
    interactively, and typically the model is loaded for inference without the calibration data structures, so
    we need to avoid over-writing them.
    """
    #
    save_index(model.support_index, model_dir)
    # save support arrays
    np.save(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS), model.train_labels,
            allow_pickle=False)
    np.save(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED), model.train_predicted_labels,
            allow_pickle=False)

    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_UUID), "w", encoding="utf-8") as f:
        f.write(json.dumps({constants.STORAGE_KEY_UNCERTAINTY_STATISTICS_SUPPORT_UUID: model.train_uuids}, ensure_ascii=True))


def save_model(model, model_dir, optimizer=None):  #, retain_support_index_after_archiving=True):
    # Note that the caller is responsible for maintaining the state of the LLM weights via
    # save_llm_weights_for_mlx_generation()
    support_index = model.support_index
    model.support_index = None  # set to None to avoid saving in main weights file
    save_index(support_index, model_dir)
    save_uncertainty_metadata(model, model_dir)
    model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
    torch.save(model.state_dict(), model_statedict_output_file)
    # re-set support index
    model.support_index = support_index
    # if retain_support_index_after_archiving:  # re-set support index
    #     model.support_index = support_index
    # else:
    #     print(f"WARNING: The support index has been set to None. This is an option for legacy code and "
    #           f"typically not what you want for further "
    #           f"use of the model without a re-load. "
    #           f"Re-load the model, and in the future, use retain_support_index_after_archiving=True.")
    #     exit()


def load_model_torch(model_dir, main_device, load_for_inference=False):
    try:
        support_index = load_index(model_dir)
        model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
        model_params, json_dict = load_uncertainty_statistics_from_disk(model_dir,
                                                                        load_for_inference=load_for_inference)

        model = SimilarityDistanceMagnitudeCalibrator(**model_params).to(main_device)
        state_dict = torch.load(model_statedict_output_file, weights_only=True, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        model.q_rescale_offset = int(json_dict[constants.STORAGE_KEY_q_rescale_offset])
        model.ood_limit = int(json_dict[constants.STORAGE_KEY_ood_limit])

        model.import_properties_from_dict(json_dict, load_for_inference=load_for_inference)
        model.set_support_index(support_index)

        model.eval()
        print(f"Model loaded successfully, set to eval() mode.")
        return model
    except:
        print(f"ERROR: The model file is missing or incomplete. Exiting.")
        exit()

def save_index(index, model_dir):
    index_output_file = path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_INDEX)
    if constants.USE_GPU_FAISS_INDEX:
        index = faiss.index_gpu_to_cpu(index)
    serialized_index = faiss.serialize_index(index)
    np.save(index_output_file, serialized_index, allow_pickle=False)
    # faiss.write_index(index, index_output_file)


def load_index(model_dir):
    index_output_file = path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_INDEX)
    loaded_index = np.load(index_output_file, allow_pickle=False)
    if constants.USE_GPU_FAISS_INDEX:
        loaded_index = faiss.deserialize_index(loaded_index)
        gpu_id = 0
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, gpu_id, loaded_index)
    else:
        return faiss.deserialize_index(loaded_index)
    # return faiss.read_index(index_output_file)


def save_global_uncertainty_statistics(global_uncertainty_statistics_object, model_dir):
    # build archive as json object
    json_dict = global_uncertainty_statistics_object.export_properties_to_dict()
    with codecs.open(path.join(model_dir, constants.FILENAME_GLOBAL_UNCERTAINTY_STATISTICS_JSON), "w", encoding="utf-8") as f:
        f.write(json.dumps(json_dict, ensure_ascii=True))
    print(f"Global uncertainty statistics have been saved to disk.")


def load_global_uncertainty_statistics_from_disk(model_dir):
    try:
        json_dict = {}
        with codecs.open(path.join(model_dir, constants.FILENAME_GLOBAL_UNCERTAINTY_STATISTICS_JSON), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                json_dict = json.loads(line)

        if json_dict[constants.STORAGE_KEY_version] == constants.ProgramIdentifiers_version:
            global_uncertainty_statistics = \
                uncertainty_statistics.UncertaintyStatistics(
                    globalUncertaintyModelUUID=str(json_dict[constants.STORAGE_KEY_globalUncertaintyModelUUID]),
                    numberOfClasses=int(json_dict[constants.STORAGE_KEY_numberOfClasses]),
                    min_valid_qbin_across_iterations= \
                        [float(x) for x in json_dict[constants.STORAGE_KEY_min_valid_qbin_across_iterations]],
                    predicted_class_to_bin_to_median_output_magnitude_of_iteration=None,
                    cauchy_quantile=float(json_dict[constants.STORAGE_KEY_cauchy_quantile])
                )
            global_uncertainty_statistics.import_properties_from_dict(json_dict)
            print(f"Global uncertainty statistics have been loaded.")
            return global_uncertainty_statistics
        else:
            print(f"WARNING: Unable to load the global uncertainty statistics since the file is from an "
                  f"incompatible version.")
    except:
        print(f"WARNING: Unable to load the global uncertainty statistics from {model_dir}")
    return None


def save_uncertainty_metadata(model, model_dir):
    # build archive as json object
    json_dict = model.export_properties_to_dict()
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS), "w", encoding="utf-8") as f:
        f.write(json.dumps(json_dict, ensure_ascii=True))

    # save support arrays
    np.save(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS), model.train_labels,
            allow_pickle=False)
    np.save(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED), model.train_predicted_labels,
            allow_pickle=False)

    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_UUID), "w", encoding="utf-8") as f:
        f.write(json.dumps({constants.STORAGE_KEY_UNCERTAINTY_STATISTICS_SUPPORT_UUID: model.train_uuids}, ensure_ascii=True))

    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_uuids), "w",
                     encoding="utf-8") as f:
        f.write(json.dumps({constants.STORAGE_KEY_UNCERTAINTY_STATISTICS_calibration_uuids: model.calibration_uuids},
                           ensure_ascii=True))

    torch.save(model.calibration_labels,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_labels_TENSOR))
    torch.save(model.calibration_predicted_labels,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_predicted_labels))
    torch.save(model.calibration_unrescaled_CDFquantiles,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_unrescaled_CDFquantiles))
    torch.save(model.calibration_soft_qbins,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_soft_qbins))


def load_uncertainty_statistics_from_disk(model_dir, load_for_inference=False):
    train_labels = np.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS),
            allow_pickle=False)
    train_predicted_labels = np.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED),
            allow_pickle=False)

    train_uuids = []
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_UUID), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_dict = json.loads(line)
            train_uuids = json_dict[constants.STORAGE_KEY_UNCERTAINTY_STATISTICS_SUPPORT_UUID]

    if load_for_inference:
        calibration_labels = None
        calibration_predicted_labels = None
        calibration_uuids = None
        calibration_unrescaled_CDFquantiles = None
        calibration_soft_qbins = None
        calibration_is_ood_indicators = []
    else:  # calibration_is_ood_indicators is loaded later, since it is part of the JSON dictionary
        calibration_uuids = []
        with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_uuids), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                json_dict = json.loads(line)
                calibration_uuids = json_dict[constants.STORAGE_KEY_UNCERTAINTY_STATISTICS_calibration_uuids]

        calibration_labels = torch.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_labels_TENSOR),
                                        weights_only=True, map_location=torch.device("cpu"))
        calibration_predicted_labels = torch.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_predicted_labels),
                   weights_only=True, map_location=torch.device("cpu"))
        calibration_unrescaled_CDFquantiles = torch.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_unrescaled_CDFquantiles),
                   weights_only=True, map_location=torch.device("cpu"))
        calibration_soft_qbins = torch.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_soft_qbins),
                   weights_only=True, map_location=torch.device("cpu"))

    json_dict = {}
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_dict = json.loads(line)

    if not load_for_inference:
        calibration_is_ood_indicators = json_dict[constants.STORAGE_KEY_calibration_is_ood_indicators]

    if len(train_uuids) > 0 and len(json_dict) > 0 and json_dict[constants.STORAGE_KEY_version] == constants.ProgramIdentifiers_version:
        model_params = {"version": str(json_dict[constants.STORAGE_KEY_version]),
                        "uncertaintyModelUUID": str(json_dict[constants.STORAGE_KEY_uncertaintyModelUUID]),
                        "numberOfClasses": int(json_dict[constants.STORAGE_KEY_numberOfClasses]),
                        "embedding_size": int(json_dict[constants.STORAGE_KEY_embedding_size]),
                        "train_labels": train_labels,
                        "train_predicted_labels": train_predicted_labels,
                        "train_uuids": train_uuids,
                        "cdfThresholdTolerance": float(json_dict[constants.STORAGE_KEY_cdfThresholdTolerance]),
                        "exemplar_vector_dimension": int(json_dict[constants.STORAGE_KEY_exemplar_vector_dimension]),
                        "trueClass_To_dCDF": None,
                        "trueClass_To_qCumulativeSampleSizeArray": None,
                        "trueClass_To_unrescaledOutputCDF": None,
                        "non_odd_thresholds": np.array(json_dict[constants.STORAGE_KEY_non_odd_thresholds]),
                        "non_odd_class_conditional_accuracy": float(json_dict[constants.STORAGE_KEY_non_odd_class_conditional_accuracy]),
                        "alpha": float(json_dict[constants.STORAGE_KEY_alpha]),
                        "maxQAvailableFromIndexer": int(json_dict[constants.STORAGE_KEY_maxQAvailableFromIndexer]),
                        "calibration_training_stage": int(json_dict[constants.STORAGE_KEY_calibration_training_stage]),
                        "min_valid_qbin_for_class_conditional_accuracy": float(json_dict[constants.STORAGE_KEY_min_valid_qbin_for_class_conditional_accuracy]),
                        "training_embedding_summary_stats":
                            json_dict[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats],

                        "is_gen_ai": bool(json_dict[constants.STORAGE_KEY_is_gen_ai]),
                        "gen_ai_vocab": int(json_dict[constants.STORAGE_KEY_gen_ai_vocab]),
                        "global_embedding_size": int(json_dict[constants.STORAGE_KEY_global_embedding_size]),
                        "composition_attributes_size": int(json_dict[constants.STORAGE_KEY_composition_attributes_size]),
                        "top_logits_k": int(json_dict[constants.STORAGE_KEY_top_logits_k]),

                        "calibration_labels": calibration_labels,  # torch tensor
                        "calibration_predicted_labels": calibration_predicted_labels,
                        "calibration_uuids": calibration_uuids,
                        "calibration_unrescaled_CDFquantiles": calibration_unrescaled_CDFquantiles,
                        "calibration_soft_qbins": calibration_soft_qbins,
                        "calibration_is_ood_indicators": calibration_is_ood_indicators,

                        "gen_ai_model_lm_head_weights": None,
                        "train_trueClass_To_dCDF": None
                        }
        # the following are added after class init:
        # self.q_rescale_offset,
        # self.ood_limit
        # self.trueClass_To_dCDF
        # self.trueClass_To_unrescaledOutputCDF
        # self.train_trueClass_To_dCDF, if is_gen_ai
        return model_params, json_dict

    return None, None

