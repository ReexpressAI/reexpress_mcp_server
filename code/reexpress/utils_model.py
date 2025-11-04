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
    torch.save(model.train_labels,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS))
    torch.save(model.train_predicted_labels,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED))
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_UUID), "w", encoding="utf-8") as f:
        f.write(json.dumps({constants.STORAGE_KEY_UNCERTAINTY_STATISTICS_SUPPORT_UUID: model.train_uuids}, ensure_ascii=True))


def save_model(model, model_dir, optimizer=None):
    # Note that the caller is responsible for maintaining the state of the LLM weights via
    support_index = model.support_index
    model.support_index = None  # set to None to avoid saving in main weights file
    save_index(support_index, model_dir)
    save_uncertainty_metadata(model, model_dir)
    model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
    torch.save(model.state_dict(), model_statedict_output_file)
    # re-set support index
    model.support_index = support_index


def load_model_torch(model_dir, main_device, load_for_inference=False):
    try:
        support_index = load_index(model_dir, main_device)
        model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
        model_params, json_dict = load_uncertainty_statistics_from_disk(model_dir,
                                                                        load_for_inference=load_for_inference)

        model = SimilarityDistanceMagnitudeCalibrator(**model_params).to(torch.device("cpu"))
        state_dict = torch.load(model_statedict_output_file, weights_only=True, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        model.q_rescale_offset = int(json_dict[constants.STORAGE_KEY_q_rescale_offset])
        model.ood_limit = int(json_dict[constants.STORAGE_KEY_ood_limit])

        model.import_properties_from_dict(json_dict, load_for_inference=load_for_inference)
        model.set_support_index(support_index)

        model.eval()
        print(f"Model loaded successfully, set to eval() mode.")
        return model.to(main_device)
    except:
        print(f"ERROR: The model file is missing or incomplete. Exiting.")
        exit()


def save_index(index, model_dir):
    index_output_file = path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_INDEX)
    # if isinstance(index, faiss.GpuIndex): # this is not present in older versions (e.g., 1.8.0)
    if "Gpu" in type(index).__name__:
        index = faiss.index_gpu_to_cpu(index)
    serialized_index = faiss.serialize_index(index)
    np.save(index_output_file, serialized_index, allow_pickle=False)


def load_index(model_dir, main_device):
    index_output_file = path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_INDEX)
    loaded_index = np.load(index_output_file, allow_pickle=False)
    if main_device.type == 'cuda':
        loaded_index = faiss.deserialize_index(loaded_index)
        gpu_id = main_device.index
        res = faiss.StandardGpuResources()
        print(f"Model is on a CUDA device, so the loaded FAISS index has been moved to cuda:{gpu_id}.")
        return faiss.index_cpu_to_gpu(res, gpu_id, loaded_index)
    else:
        return faiss.deserialize_index(loaded_index)


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
                    min_rescaled_similarity_across_iterations= \
                        [float(x) for x in json_dict[constants.STORAGE_KEY_min_rescaled_similarity_across_iterations]]
                )
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

    # save support tensors
    torch.save(model.train_labels,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS))
    torch.save(model.train_predicted_labels,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED))

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
    torch.save(model.calibration_sdm_outputs,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_sdm_outputs))
    torch.save(model.calibration_rescaled_similarity_values,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_rescaled_similarity_values))

    torch.save(model.hr_output_thresholds,
               path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_hr_output_thresholds))


def load_uncertainty_statistics_from_disk(model_dir, load_for_inference=False):
    train_labels = torch.load(
        path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS),
        weights_only=True, map_location=torch.device("cpu"))
    train_predicted_labels = torch.load(
        path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED),
        weights_only=True, map_location=torch.device("cpu"))

    hr_output_thresholds = torch.load(
        path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_hr_output_thresholds),
        weights_only=True, map_location=torch.device("cpu"))

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
        calibration_sdm_outputs = None
        calibration_rescaled_similarity_values = None
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
        calibration_sdm_outputs = torch.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_sdm_outputs),
                   weights_only=True, map_location=torch.device("cpu"))
        calibration_rescaled_similarity_values = torch.load(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS_calibration_rescaled_similarity_values),
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
                        "hr_output_thresholds": hr_output_thresholds,
                        "hr_class_conditional_accuracy": float(json_dict[constants.STORAGE_KEY_hr_class_conditional_accuracy]),
                        "alpha": float(json_dict[constants.STORAGE_KEY_alpha]),
                        "maxQAvailableFromIndexer": int(json_dict[constants.STORAGE_KEY_maxQAvailableFromIndexer]),
                        "calibration_training_stage": int(json_dict[constants.STORAGE_KEY_calibration_training_stage]),
                        "min_rescaled_similarity_to_determine_high_reliability_region": float(json_dict[constants.STORAGE_KEY_min_rescaled_similarity_to_determine_high_reliability_region]),
                        "training_embedding_summary_stats":
                            json_dict[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats],

                        "is_sdm_network_verification_layer":
                            bool(json_dict[constants.STORAGE_KEY_is_sdm_network_verification_layer]),

                        "calibration_labels": calibration_labels,  # torch tensor
                        "calibration_predicted_labels": calibration_predicted_labels,
                        "calibration_uuids": calibration_uuids,
                        "calibration_sdm_outputs": calibration_sdm_outputs,
                        "calibration_rescaled_similarity_values": calibration_rescaled_similarity_values,
                        "calibration_is_ood_indicators": calibration_is_ood_indicators,
                        "train_trueClass_To_dCDF": None
                        }
        # the following are added after class init:
        # self.q_rescale_offset,
        # self.ood_limit
        # self.trueClass_To_dCDF
        # self.train_trueClass_To_dCDF, if is_sdm_network_verification_layer
        return model_params, json_dict

    return None, None
