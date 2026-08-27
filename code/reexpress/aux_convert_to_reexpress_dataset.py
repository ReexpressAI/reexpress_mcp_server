# Copyright Reexpress AI, Inc. All rights reserved.

# Convert a .jsonl data file (the legacy input format) to the binary dataset directory format
# (constants.DATASET_FORMAT_version), which is memory-mapped at load time. Each line of the input .jsonl must
# contain the fields "id", "label", and "embedding" and/or "attributes" (with "document" optional, and only
# needed for the eval output files). The network input is composed here at conversion time (from "embedding"
# and/or "attributes", controlled by --use_embeddings and --concat_embeddings_to_attributes, with the same
# semantics as the corresponding training/eval flags), so the converted directory has a single input column.
#
# Conversion is streaming (two passes over the input file), so peak memory is O(1) in the number of documents:
# the input matrix is written row-by-row to a file-backed tensor (torch.from_file). The metadata dictionary is
# written last, as a completeness marker: a directory without it is treated as invalid by the loaders.
#
# Additional per-document metadata fields (for downstream analysis scripts) can be retained with
# --additional_fields_to_retain "fieldA,fieldB", which are stored as JSON lines (aligned by row) in
# constants.FILENAME_DATASET_ADDITIONAL_FIELDS. The training and eval code ignores that file; to add new
# fields, no changes to the pipeline are needed.

import constants
import data_validator

import torch

import argparse
import json
import codecs
import os
from os import path


def compose_input_values(json_obj, use_embeddings, concat_embeddings_to_attributes, line_index):
    if concat_embeddings_to_attributes:
        assert "embedding" in json_obj and "attributes" in json_obj, \
            f"ERROR: Line {line_index}: --concat_embeddings_to_attributes requires both 'embedding' and " \
            f"'attributes' fields."
        return json_obj["embedding"] + json_obj["attributes"]
    elif use_embeddings:
        assert "embedding" in json_obj, f"ERROR: Line {line_index}: missing the 'embedding' field."
        return json_obj["embedding"]
    else:
        assert "attributes" in json_obj, f"ERROR: Line {line_index}: missing the 'attributes' field."
        return json_obj["attributes"]


def get_input_composition_label(use_embeddings, concat_embeddings_to_attributes):
    if concat_embeddings_to_attributes:
        return "embedding+attributes"
    elif use_embeddings:
        return "embedding"
    return "attributes"


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="-----[Convert a .jsonl data file to the binary dataset "
                                                 "directory format]-----")
    parser.add_argument("--input_file", required=True, help=".jsonl format (the legacy input format)")
    parser.add_argument("--output_dataset_directory", required=True,
                        help="Output directory. Must not already contain a converted dataset.")
    parser.add_argument("--class_size", default=2, type=int,
                        help="Used to validate the 'label' field. Labels outside "
                             "{0, ..., class_size-1, unlabeledLabel, oodLabel} produce a warning.")
    parser.add_argument("--use_embeddings", default=False, action='store_true',
                        help="The network input is the 'embedding' field. (Default: the 'attributes' field.)")
    parser.add_argument("--concat_embeddings_to_attributes", default=False, action='store_true',
                        help="The network input is the concatenation of the 'embedding' and 'attributes' "
                             "fields.")
    parser.add_argument("--additional_fields_to_retain", default="",
                        help="Comma-separated field names to retain (aligned by row) in "
                             f"{constants.FILENAME_DATASET_ADDITIONAL_FIELDS}, for downstream analysis "
                             "scripts. The training and eval code ignores these.")
    options = parser.parse_args()

    additional_field_names = [field_name.strip() for field_name in
                              options.additional_fields_to_retain.split(",") if field_name.strip() != ""]
    metadata_file = path.join(options.output_dataset_directory, constants.FILENAME_DATASET_METADATA)
    assert not path.exists(metadata_file), \
        f"ERROR: {options.output_dataset_directory} already contains a converted dataset."
    os.makedirs(options.output_dataset_directory, exist_ok=True)

    # Pass 1: count the documents and determine the input dimension from the first line:
    n = 0
    input_dim = None
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if input_dim is None:
                json_obj = json.loads(line)
                input_dim = len(compose_input_values(json_obj, options.use_embeddings,
                                                     options.concat_embeddings_to_attributes, line_index=0))
            n += 1
    assert n > 0 and input_dim is not None and input_dim > 0, \
        f"ERROR: No documents found in {options.input_file}."
    print(f"Converting {n} documents with input dimension {input_dim} "
          f"(input composition: "
          f"{get_input_composition_label(options.use_embeddings, options.concat_embeddings_to_attributes)}).")

    # Pass 2: stream the rows into the file-backed input tensor and the JSON-lines sidecar files:
    input_embeddings_file = path.join(options.output_dataset_directory,
                                      constants.FILENAME_DATASET_INPUT_EMBEDDINGS)
    input_embeddings = torch.from_file(input_embeddings_file, shared=True, size=n * input_dim,
                                       dtype=torch.float32)
    labels = torch.zeros(n, dtype=torch.long)
    uuids = []
    count_unexpected_labels = 0
    documents_file = codecs.open(path.join(options.output_dataset_directory,
                                           constants.FILENAME_DATASET_DOCUMENTS), "w", encoding="utf-8")
    additional_fields_file = None
    if len(additional_field_names) > 0:
        additional_fields_file = codecs.open(path.join(options.output_dataset_directory,
                                                       constants.FILENAME_DATASET_ADDITIONAL_FIELDS),
                                             "w", encoding="utf-8")
    row_index = 0
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            json_obj = json.loads(line)
            input_values = compose_input_values(json_obj, options.use_embeddings,
                                                options.concat_embeddings_to_attributes, line_index=row_index)
            assert len(input_values) == input_dim, \
                f"ERROR: Line {row_index}: input dimension {len(input_values)} != {input_dim}."
            input_embeddings[row_index * input_dim:(row_index + 1) * input_dim] = \
                torch.tensor(input_values, dtype=torch.float32)
            label = int(json_obj["label"])
            if not data_validator.isValidLabel(label=label, numberOfClasses=options.class_size):
                count_unexpected_labels += 1
            labels[row_index] = label
            uuids.append(json_obj["id"])
            # Note: json.dumps escapes all control characters (including newlines) and, with the default
            # ensure_ascii=True, all non-ASCII characters (including the Unicode line separators
            # U+2028/U+2029/NEL), so each document is exactly one physical (pure-ASCII) line, for arbitrary
            # document content. Do not switch to ensure_ascii=False: the raw Unicode line separators are legal
            # inside JSON strings and would split records under the reader's codecs line iteration (see
            # utils_preprocess.get_documents_from_dataset_directory()).
            documents_file.write(json.dumps(json_obj.get("document", "")) + "\n")
            if additional_fields_file is not None:
                additional_fields_file.write(
                    json.dumps({field_name: json_obj.get(field_name, None)
                                for field_name in additional_field_names}) + "\n")
            row_index += 1
    documents_file.close()
    if additional_fields_file is not None:
        additional_fields_file.close()
    assert row_index == n
    assert len(set(uuids)) == len(uuids), "ERROR: The 'id' fields are not unique."
    if count_unexpected_labels > 0:
        print(f"WARNING: {count_unexpected_labels} of {n} documents have a label outside "
              f"{data_validator.allValidLabelsAsArray(options.class_size)}.")
    del input_embeddings  # flush the file-backed tensor

    # The metadata dictionary is written last, as a completeness marker:
    dataset_metadata = {
        constants.STORAGE_KEY_version: constants.ProgramIdentifiers_version,
        constants.STORAGE_KEY_DATASET_format: constants.DATASET_FORMAT_version,
        constants.STORAGE_KEY_DATASET_n: n,
        constants.STORAGE_KEY_DATASET_input_dim: input_dim,
        constants.STORAGE_KEY_DATASET_input_composition:
            get_input_composition_label(options.use_embeddings, options.concat_embeddings_to_attributes),
        constants.STORAGE_KEY_DATASET_labels: labels,
        constants.STORAGE_KEY_DATASET_uuids: uuids,
        constants.STORAGE_KEY_DATASET_has_documents: True,
        constants.STORAGE_KEY_DATASET_has_additional_fields: len(additional_field_names) > 0,
        constants.STORAGE_KEY_DATASET_additional_field_names: additional_field_names,
    }
    torch.save(dataset_metadata, metadata_file)
    print(f"Converted dataset saved to {options.output_dataset_directory}")


if __name__ == "__main__":
    main()
