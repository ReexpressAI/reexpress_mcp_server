# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import constants

import torch
import json
import codecs
from os import path


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


def save_baseline_model(model, model_dir, optimizer=None):  #, retain_support_index_after_archiving=True):
    model.support_index = None  # set to None to avoid saving in main weights file
    save_baseline_uncertainty_metadata(model, model_dir)
    model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
    torch.save(model.state_dict(), model_statedict_output_file)


def load_baseline_model_torch(model_dir, main_device, load_for_inference=False):
    # There is no support index in this case.
    if True: #try:
        model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
        model_params, json_dict = load_uncertainty_statistics_from_disk_for_baseline(
            model_dir,
            load_for_inference=load_for_inference)

        model = SimilarityDistanceMagnitudeCalibrator(**model_params).to(torch.device("cpu"))
        state_dict = torch.load(model_statedict_output_file, weights_only=True, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        model.q_rescale_offset = int(json_dict[constants.STORAGE_KEY_q_rescale_offset])
        model.ood_limit = int(json_dict[constants.STORAGE_KEY_ood_limit])

        model.import_properties_from_dict(json_dict, load_for_inference=load_for_inference)

        model.eval()
        print(f"Model loaded successfully, set to eval() mode.")
        return model.to(main_device)
    else: # except:
        print(f"ERROR: The model file is missing or incomplete. Exiting.")
        exit()


def save_baseline_uncertainty_metadata(model, model_dir):
    # build archive as json object
    json_dict = model.export_properties_to_dict()
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS), "w", encoding="utf-8") as f:
        f.write(json.dumps(json_dict, ensure_ascii=True))


def load_uncertainty_statistics_from_disk_for_baseline(model_dir, load_for_inference=False):

    calibration_labels = None
    calibration_predicted_labels = None
    calibration_uuids = None
    calibration_sdm_outputs = None
    calibration_similarity_values = None
    calibration_is_ood_indicators = []

    json_dict = {}
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_dict = json.loads(line)

    model_params = {"version": str(json_dict[constants.STORAGE_KEY_version]),
                    "uncertaintyModelUUID": str(json_dict[constants.STORAGE_KEY_uncertaintyModelUUID]),
                    "numberOfClasses": int(json_dict[constants.STORAGE_KEY_numberOfClasses]),
                    "embedding_size": int(json_dict[constants.STORAGE_KEY_embedding_size]),
                    "train_labels": None,
                    "train_predicted_labels": None,
                    "train_uuids": None,
                    "cdfThresholdTolerance": float(json_dict[constants.STORAGE_KEY_cdfThresholdTolerance]),
                    "exemplar_vector_dimension": int(json_dict[constants.STORAGE_KEY_exemplar_vector_dimension]),
                    "trueClass_To_dCDF": None,
                    "trueClass_To_qCumulativeSampleSizeArray": None,
                    "hr_output_thresholds": None,
                    "hr_class_conditional_accuracy":
                        float(json_dict[constants.STORAGE_KEY_hr_class_conditional_accuracy]),
                    "alpha": float(json_dict[constants.STORAGE_KEY_alpha]),
                    "maxQAvailableFromIndexer": int(json_dict[constants.STORAGE_KEY_maxQAvailableFromIndexer]),
                    "calibration_training_stage": int(json_dict[constants.STORAGE_KEY_calibration_training_stage]),
                    "min_rescaled_similarity_to_determine_high_reliability_region": float(
                        json_dict[constants.STORAGE_KEY_min_rescaled_similarity_to_determine_high_reliability_region]),
                    "training_embedding_summary_stats":
                        json_dict[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats],
                    "calibration_labels": calibration_labels,  # torch tensor
                    "calibration_predicted_labels": calibration_predicted_labels,
                    "calibration_uuids": calibration_uuids,
                    "calibration_sdm_outputs": calibration_sdm_outputs,
                    "calibration_rescaled_similarity_values": calibration_similarity_values,
                    "calibration_is_ood_indicators": calibration_is_ood_indicators,
                    "is_sdm_network_verification_layer":
                        bool(json_dict[constants.STORAGE_KEY_is_sdm_network_verification_layer]),
                    "train_trueClass_To_dCDF": None
                    }
    return model_params, json_dict


