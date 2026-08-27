# Copyright Reexpress AI, Inc. All rights reserved.

import data_validator
import utils_model
import constants

import torch
import numpy as np

import json
import codecs
from os import path


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


def is_reexpress_dataset_directory(filepath_with_name) -> bool:
    # A converted (binary) dataset directory (see convert_to_reexpress_dataset.py) is identified by the
    # presence of the metadata dictionary, which the converter writes last as a completeness marker.
    return path.isdir(filepath_with_name) and \
        path.exists(path.join(filepath_with_name, constants.FILENAME_DATASET_METADATA))


def get_documents_from_dataset_directory(dataset_directory):
    # Each line is a single JSON string (see convert_to_reexpress_dataset.py): json.dumps escapes all control
    # and (with the default ensure_ascii=True) non-ASCII characters, so arbitrary document content (embedded
    # newlines, quotes, Unicode line separators, etc.) is exactly one physical line here, and json.loads
    # restores it exactly.
    documents = []
    with codecs.open(path.join(dataset_directory, constants.FILENAME_DATASET_DOCUMENTS), "r",
                     encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def load_reexpress_dataset_source(dataset_directory, include_documents=False):
    """
    Load one converted dataset directory. The input embeddings are memory-mapped (torch.from_file with
    shared=False, i.e., a private, copy-on-write mapping), so rows are only read from disk when accessed
    (e.g., when gathering a split, or slicing an eval batch). Documents are only read when
    include_documents=True (they are not needed for training).
    """
    dataset_metadata = torch.load(path.join(dataset_directory, constants.FILENAME_DATASET_METADATA),
                                  weights_only=True, map_location="cpu")
    assert dataset_metadata[constants.STORAGE_KEY_DATASET_format] == constants.DATASET_FORMAT_version, \
        f"ERROR: Unexpected dataset format in {dataset_directory}."
    n = dataset_metadata[constants.STORAGE_KEY_DATASET_n]
    input_dim = dataset_metadata[constants.STORAGE_KEY_DATASET_input_dim]
    input_embeddings_file = path.join(dataset_directory, constants.FILENAME_DATASET_INPUT_EMBEDDINGS)
    expected_bytes = n * input_dim * 4  # float32
    assert path.getsize(input_embeddings_file) == expected_bytes, \
        f"ERROR: {input_embeddings_file} has an unexpected size " \
        f"({path.getsize(input_embeddings_file)} bytes vs. the expected {expected_bytes})."
    embeddings = torch.from_file(input_embeddings_file, shared=False, size=n * input_dim,
                                 dtype=torch.float32).view(n, input_dim)
    documents = None
    if include_documents:
        documents = get_documents_from_dataset_directory(dataset_directory)
        assert len(documents) == n
    labels = dataset_metadata[constants.STORAGE_KEY_DATASET_labels]
    uuids = dataset_metadata[constants.STORAGE_KEY_DATASET_uuids]
    assert labels.shape[0] == n and len(uuids) == n
    return {"dataset_directory": dataset_directory,
            "n": n,
            "input_dim": input_dim,
            "input_composition": dataset_metadata[constants.STORAGE_KEY_DATASET_input_composition],
            "embeddings": embeddings,
            "labels": labels,
            "uuids": uuids,
            "documents": documents}


def gather_split_from_dataset_sources(sources, split_indices, include_documents=False):
    """
    Materialize one split by gathering the rows of split_indices (a 1-D torch.long tensor of indices into the
    logical concatenation of the sources, in order), preserving the order of split_indices. This is the
    replacement for shuffling and splitting the raw data itself: the (tiny) index tensors fully determine the
    split, given the source directories.
    """
    total_n = sum(source["n"] for source in sources)
    split_size = split_indices.shape[0]
    if split_size > 0:
        assert 0 <= split_indices.min().item() and split_indices.max().item() < total_n
    input_dim = sources[0]["input_dim"]
    embeddings = torch.zeros(split_size, input_dim, dtype=torch.float32)
    labels = torch.zeros(split_size, dtype=torch.long)
    uuids = [None] * split_size
    documents = [None] * split_size if include_documents else None
    offset = 0
    for source in sources:
        assert source["input_dim"] == input_dim, \
            f"ERROR: The dataset sources have inconsistent input dimensions."
        in_source = (split_indices >= offset) & (split_indices < offset + source["n"])
        source_row_indices = split_indices[in_source] - offset
        embeddings[in_source] = source["embeddings"][source_row_indices]
        labels[in_source] = source["labels"][source_row_indices]
        destination_positions = torch.nonzero(in_source, as_tuple=True)[0].tolist()
        source_row_indices_list = source_row_indices.tolist()
        for destination_i, source_i in zip(destination_positions, source_row_indices_list):
            uuids[destination_i] = source["uuids"][source_i]
            if include_documents:
                documents[destination_i] = source["documents"][source_i]
        offset += source["n"]
    return {"embeddings": embeddings, "labels": labels, "uuids": uuids, "documents": documents}


def get_meta_data_dict(embeddings, labels_list, uuids, documents):
    # The common return structure of the data loading functions. The "original_labels",
    # "original_predictions", and "refusals" fields of the legacy .jsonl format are no longer used and are
    # empty for converted datasets.
    uuid2idx = {}
    for line_id, uuid_value in enumerate(uuids):
        uuid2idx[uuid_value] = line_id
    return {"lines": documents,
            "line_ids": list(range(len(uuids))),
            "original_labels": [],
            "original_predictions": [],
            "labels": labels_list,
            "refusals": [],
            "embeddings": embeddings,
            "uuids": uuids,
            "uuid2idx": uuid2idx}


def load_reexpress_dataset_directory(options, dataset_directory, include_documents=True,
                                     calculate_summary_stats=False, is_training=False):
    """
    Load a converted dataset directory with the same return structure as get_metadata_lines(). Note that the
    network input composition (the 'embedding' and/or 'attributes' fields of the source .jsonl) was determined
    at conversion time (see convert_to_reexpress_dataset.py), so the --use_embeddings and
    --concat_embeddings_to_attributes flags have no effect here.
    """
    source = load_reexpress_dataset_source(dataset_directory, include_documents=include_documents)
    print(f"Loaded the converted dataset in {dataset_directory}: {source['n']} documents with input "
          f"dimension {source['input_dim']} (input composition, determined at conversion time: "
          f"{source['input_composition']}).")
    summary_stats = None
    if calculate_summary_stats:
        if options.do_not_normalize_input_embeddings:
            summary_stats = {
                constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: 0.0,
                constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: 1.0
            }
        else:
            summary_stats = utils_model.get_embedding_summary_stats(source["embeddings"], is_training)
    documents = source["documents"] if include_documents else [""] * source["n"]
    return get_meta_data_dict(source["embeddings"], source["labels"].tolist(), source["uuids"],
                              documents), summary_stats


def get_meta_data_dict_and_summary_stats_for_training_split(options, gathered_training_split):
    # The training path does not need documents, so empty strings are used for the "lines" field:
    if options.do_not_normalize_input_embeddings:
        summary_stats = {
            constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: 0.0,
            constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: 1.0
        }
    else:
        summary_stats = utils_model.get_embedding_summary_stats(gathered_training_split["embeddings"],
                                                                is_training=True)
    return get_meta_data_dict(gathered_training_split["embeddings"],
                              gathered_training_split["labels"].tolist(),
                              gathered_training_split["uuids"],
                              [""] * len(gathered_training_split["uuids"])), summary_stats


def get_best_iteration_split_dict_and_sources(model_dir, include_documents=False):
    split_indices_file = path.join(model_dir, "best_iteration_data",
                                   constants.FILENAME_BEST_ITERATION_SPLIT_INDICES)
    assert path.exists(split_indices_file), \
        f"ERROR: {split_indices_file} not found. The best-iteration split indices are saved when training " \
        f"with converted dataset directories as input."
    split_dict = torch.load(split_indices_file, weights_only=True, map_location="cpu")
    sources = [load_reexpress_dataset_source(dataset_directory, include_documents=include_documents)
               for dataset_directory in split_dict[constants.STORAGE_KEY_SPLIT_source_directories]]
    return split_dict, sources


def load_best_iteration_splits(model_dir):
    """
    Reconstruct the training and calibration splits of the best training iteration from the saved split
    indices (see utils_train_iterative_main), as gathered split dictionaries (without documents).
    """
    split_dict, sources = get_best_iteration_split_dict_and_sources(model_dir, include_documents=False)
    train_split = gather_split_from_dataset_sources(
        sources, split_dict[constants.STORAGE_KEY_SPLIT_train_indices], include_documents=False)
    calibration_split = gather_split_from_dataset_sources(
        sources, split_dict[constants.STORAGE_KEY_SPLIT_calibration_indices], include_documents=False)
    print(f"Reconstructed the best-iteration splits: {train_split['labels'].shape[0]} training and "
          f"{calibration_split['labels'].shape[0]} calibration documents.")
    return train_split, calibration_split


def load_best_iteration_calibration_split(model_dir):
    """
    Reconstruct the calibration split of the best training iteration from the saved split indices (see
    utils_train_iterative_main), with the same return structure as get_metadata_lines(). Documents are
    included (e.g., for the eval output files).
    """
    split_dict, sources = get_best_iteration_split_dict_and_sources(model_dir, include_documents=True)
    gathered = gather_split_from_dataset_sources(
        sources, split_dict[constants.STORAGE_KEY_SPLIT_calibration_indices], include_documents=True)
    print(f"Reconstructed the best-iteration calibration split: {gathered['labels'].shape[0]} documents.")
    return get_meta_data_dict(gathered["embeddings"], gathered["labels"].tolist(), gathered["uuids"],
                              gathered["documents"])


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
    if is_reexpress_dataset_directory(filepath_with_name):
        # The converted (binary) dataset directory format; the flags reduce, use_embeddings, and
        # concat_embeddings_to_attributes have no effect on this path (the input composition was determined
        # at conversion time):
        return load_reexpress_dataset_directory(options, filepath_with_name, include_documents=True,
                                                calculate_summary_stats=calculate_summary_stats,
                                                is_training=is_training)
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