# Copyright Reexpress AI, Inc. All rights reserved.


import json
import argparse
import time
import codecs

import sdm_network_constants
import sdm_network_finetune_utils_trainer_reward_assignment
import sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_hard_negative_lines(filepath_with_name):
    json_lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            document_id = line[sdm_network_constants.REEXPRESS_ID_KEY]
            label = line[sdm_network_constants.REEXPRESS_LABEL_KEY]
            reference_document_string = line[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY]
            generated_text = line[sdm_network_constants.GENERATED_RESPONSE_KEY]

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
                    not generation_matches_reference_text_but_classification_is_wrong and label == 0:
                # The label may be 1 if the generation has extraneous information. In our initial experiments, this
                # only occurred with the baseline supervised models. Here, we ignore these cases by adding the
                # restriction that label == 0, since we parse
                # before passing the input to the final verification layer.
                # assert label == 0, f"ERROR: These cases should already have been checked in the generation script. " \
                #                    f"reference: {reference_document_string} generated: {generated_text}"

                # The response is not an exact match of the text and/or formatting,
                # so we ensure "<verified>No</verified>\n\n" occurs at the end
                response, _ = \
                    sdm_network_finetune_utils_verification_layer_utils_sdm_embeddings.generate_formatted_assistant_response_from_malformed_output(
                        generated_text=generated_text)
                # format matches that used in compute_loss during training
                json_obj = {
                    sdm_network_constants.REEXPRESS_ID_KEY: document_id,
                    sdm_network_constants.HARD_NEGATIVES_KEY: [response],
                    sdm_network_constants.SDM_NETWORK_FINETUNING_EPOCH_KEY: "post-training",
                }
                json_lines.append(json_obj)
    return json_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Preprocess data]-----")
    parser.add_argument("--input_file", required=True, help="")
    parser.add_argument("--output_hard_negatives_jsonl_file", required=True, help="")

    options = parser.parse_args()

    start_time = time.time()

    json_lines = get_hard_negative_lines(options.input_file)

    save_json_lines(options.output_hard_negatives_jsonl_file, json_lines)
    print(f"Documents with generated hard negatives: {len(json_lines)}")

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
