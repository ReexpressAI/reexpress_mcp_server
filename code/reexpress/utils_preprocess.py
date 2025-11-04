# Copyright Reexpress AI, Inc. All rights reserved.

import data_validator
import utils_model
import constants

import torch
import numpy as np

import json
import codecs


def get_data(filename_with_path):
    """
    Get the preprocessed data
    :param filename_with_path: A filepath to the preprocessed data. See the Tutorial for details.
    :return: A list of dictionaries
    """
    json_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list


def get_metadata_lines_from_json_list(options, json_list, reduce=False, reduce_size=20, use_embeddings=True,
                                      concat_embeddings_to_attributes=False, calculate_summary_stats=False, is_training=False):
    lines = []
    line_ids = []
    line_id = 0
    labels = []
    original_labels = []
    original_predictions = []
    embeddings = []
    uuids = []
    uuid2idx = {}
    refusals = []
    for json_obj in json_list:
        uuids.append(json_obj["id"])
        uuid2idx[json_obj["id"]] = line_id
        label = int(json_obj['label'])
        # if not data_validator.isKnownValidLabel(label=label, numberOfClasses=numberOfClasses):
        #     print("Currently we do not support ")
        if "refusal" in json_obj:
            refusals.append(json_obj["refusal"])
        if "original_label" in json_obj:
            original_label = int(json_obj["original_label"])
            original_labels.append(original_label)
        # This can be useful for comparing against tasks in which the input is a textual representation
        # of the output, which could (in principle) differ from the calibrated version.
        if "original_prediction" in json_obj:
            original_prediction = int(json_obj["original_prediction"])
            original_predictions.append(original_prediction)
        labels.append(label)
        lines.append(json_obj.get('document', ''))
        line_ids.append(line_id)
        if concat_embeddings_to_attributes:
            embedding = torch.tensor(json_obj["embedding"] + json_obj["attributes"])
        elif use_embeddings:
            embedding = torch.tensor(json_obj["embedding"])
        else:
            embedding = torch.tensor(json_obj["attributes"])
        embeddings.append(embedding.unsqueeze(0))
        line_id += 1
        if reduce and line_id == reduce_size:
            break
    assert len(lines) == len(line_ids)

    embeddings = torch.cat(embeddings, dim=0)

    summary_stats = None
    if calculate_summary_stats:
        if options.do_not_normalize_input_embeddings:
            summary_stats = {
                constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: 0.0,
                constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: 1.0
            }
        else:
            summary_stats = utils_model.get_embedding_summary_stats(embeddings, is_training)

    print(f"Total existing metadata lines: {len(lines)}")
    return {"lines": lines,
            "line_ids": line_ids,
            "original_labels": original_labels,  # the original task labels, if applicable
            "original_predictions": original_predictions,  # the original LLM prediction, if applicable
            "labels": labels,
            "refusals": refusals,
            "embeddings": embeddings,
            "uuids": uuids,
            "uuid2idx": uuid2idx}, summary_stats


def get_metadata_lines(options, filepath_with_name, reduce=False, reduce_size=20, use_embeddings=True,
                       concat_embeddings_to_attributes=False, calculate_summary_stats=False, is_training=False):
    lines = []
    line_ids = []
    line_id = 0
    labels = []
    original_labels = []
    original_predictions = []
    embeddings = []
    uuids = []
    uuid2idx = {}
    refusals = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)

            uuids.append(json_obj["id"])
            uuid2idx[json_obj["id"]] = line_id
            label = int(json_obj['label'])
            labels.append(label)
            if "refusal" in json_obj:
                refusals.append(json_obj["refusal"])
            if "original_label" in json_obj:
                original_label = int(json_obj["original_label"])
                original_labels.append(original_label)
            # This can be useful for comparing against tasks in which the input is a textual representation
            # of the output, which could (in principle) differ from the calibrated version.
            if "original_prediction" in json_obj:
                original_prediction = int(json_obj["original_prediction"])
                original_predictions.append(original_prediction)
            lines.append(json_obj.get('document', ''))
            line_ids.append(line_id)
            if concat_embeddings_to_attributes:
                embedding = torch.tensor(json_obj["embedding"] + json_obj["attributes"])
            elif use_embeddings:
                embedding = torch.tensor(json_obj["embedding"])
            else:
                embedding = torch.tensor(json_obj["attributes"])
            embeddings.append(embedding.unsqueeze(0))
            line_id += 1
            if reduce and line_id == reduce_size:
                break
        assert len(lines) == len(line_ids)

    embeddings = torch.cat(embeddings, dim=0)
    summary_stats = None
    if calculate_summary_stats:
        if options.do_not_normalize_input_embeddings:
            summary_stats = {
                constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: 0.0,
                constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: 1.0
            }
        else:
            summary_stats = utils_model.get_embedding_summary_stats(embeddings, is_training)

    print(f"Total existing metadata lines: {len(lines)}")
    return {"lines": lines,
            "line_ids": line_ids,
            "original_labels": original_labels,  # the original task labels, if applicable
            "original_predictions": original_predictions,  # the original LLM prediction, if applicable
            "labels": labels,
            "refusals": refusals,
            "embeddings": embeddings,
            "uuids": uuids,
            "uuid2idx": uuid2idx}, summary_stats