# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import constants
import uncertainty_statistics
import data_validator
import utils_model

from mlx_lm import load, generate
import mlx.core as mx
import torch
import numpy as np

# import math
import json
import codecs
from os import path
from typing import Optional
from collections import namedtuple
from pathlib import Path
import time

LLMType = namedtuple("LLMType", ["phiThreePointFive", "phi4"])
llmTypes = LLMType(0, 1)

ModelCategory = namedtuple("ModelCategory", ["generation_force_decode__token_level",
                                             "classification_with_generation__document_level",
                                             "classification_with_force_decoded_generation__document_level"])
modelCategories = ModelCategory(0, 1, 2)
TaskCategory = namedtuple("TaskCategory", ["sentiment",
                                           "factcheck"])
taskCategories = TaskCategory(0, 1)


def get_response(gen_ai_model, tokenizer, prompt: str, max_tokens=2):
    assert max_tokens >= 2, f"max_tokens must be at least 2."
    response = generate(gen_ai_model, tokenizer, prompt=prompt,
                        message_text_to_force_decode=None,
                        max_tokens=max_tokens,
                        verbose=False,
                        return_embeddings=True,
                        prefill_step_size=2048)
    return response


def parse_generation(decoded_text, original_label, taskCategory, original_task_accurcy_per_task_per_class,
                     json_parse_error_by_task):
    print_parse_errors = False
    prediction = data_validator.oodLabel
    try:
        right_bracket = decoded_text.find("}")
        left_bracket = decoded_text.find("{")
        if left_bracket != -1 and right_bracket != -1:
            json_obj = json.loads(decoded_text[left_bracket:right_bracket + 1])
            # prediction = -1
            if taskCategory == taskCategories.sentiment:
                prediction_str = json_obj["sentiment"]
                if prediction_str == "negative":
                    prediction = 0
                elif prediction_str == "positive":
                    prediction = 1
                else:
                    assert False
            elif taskCategory == taskCategories.factcheck:
                prediction_bool = bool(json_obj["correctness"])
                if not prediction_bool:
                    prediction = 0
                else:
                    prediction = 1
            assert prediction in [0, 1]
            original_task_accurcy_per_task_per_class[taskCategory][original_label].append(
                original_label == prediction)
    except:
        if print_parse_errors:
            print(f"Parse error: {decoded_text}; original label: {original_label}")
        # parsing errors are treated as wrong predictions:
        original_task_accurcy_per_task_per_class[taskCategory][original_label].append(False)
        json_parse_error_by_task[taskCategory] += 1
    return original_task_accurcy_per_task_per_class, json_parse_error_by_task, prediction


def get_running_average_embeddings(embeddings):
    cumulative_sum = torch.cumsum(embeddings, dim=0)
    cumulative_n = torch.arange(1, embeddings.size(0) + 1)
    return cumulative_sum / cumulative_n[:, None]

def get_metadata_lines_for_gen(options, gen_ai_model, tokenizer, max_length, filepath_with_name, return_text=False,
                               calculate_summary_stats=False, is_training=False,
                               taskCategory=None,
                               modelCategory=None, top_logits_k=3, model=None, llmType=None,
                               kComposition_attributes_size: int=0, eval_label="",
                               load_final_llm_weights=True):
    if llmType == llmTypes.phiThreePointFive:
        print("Using Phi3.5")
    elif llmType == llmTypes.phi4:
        print("Using Phi4")
    if modelCategory == modelCategories.classification_with_generation__document_level:
        assert model is not None
    if model is not None:
        if not load_final_llm_weights:
            llm_weights_dir = str(Path(options.model_dir, constants.DIRNAME_RUNNING_LLM_WEIGHTS_DIR).as_posix())
            print(f"Loading non-final LLM weights from {llm_weights_dir}")
        else:
            llm_weights_dir = options.model_dir
            print(f"Loading final LLM weights from {llm_weights_dir}")
        gen_ai_model.add_adaptors(llm_weights_dir)
        gen_ai_model.switch_distribution(3)
    else:
        print(f"Using the original un-adapted LLM model.")

    uuid2idx = {}
    lines = []
    generated_lines = []
    line_id = 0
    original_labels = []
    labels = []
    shifted_token_labels = []
    end_of_document_indicators = []  # 1 if final token to determine document-level classification; 0 otherwise
    embeddings = []
    uuids = []
    construct_negative_indicators = []
    taskCategoryInts = []
    prompts_no_assistant = []

    task_predictions = []

    embedding_size = 0
    global_embedding_size = 0
    composition_attributes_size = 0

    numberOfClasses = options.class_size
    original_task_accurcy_per_task_per_class = {}
    for task_label in range(len([taskCategories.sentiment, taskCategories.factcheck])):
        original_task_accurcy_per_task_per_class[task_label] = {}
        for class_label in range(numberOfClasses):
            original_task_accurcy_per_task_per_class[task_label][class_label] = []
    json_parse_error_by_task = {}
    for task_label in range(len([taskCategories.sentiment, taskCategories.factcheck])):
        json_parse_error_by_task[task_label] = 0

    gen_ai_vocab = options.gen_ai_vocab
    assert numberOfClasses == 2, f"ERROR: Currently only binary preferences are implemented."
    if eval_label != "":
        eval_label = f"{eval_label}: "
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 100 == 0:
                print(f"{eval_label}Caching document: {line_id}")
            line = line.strip()
            json_obj = json.loads(line)

            construct_negative = bool(json_obj['construct_negative'])
            if modelCategory == modelCategories.generation_force_decode__token_level and construct_negative:
                # Note that hard/constructed negatives do not participate in the LLM next-token fine-tuning.
                line_id += 1
                continue

            construct_negative_indicators.append(construct_negative)
            document = json_obj['document'][0:max_length].strip()  # TODO: Currently character level

            max_generation_tokens = json_obj['max_generation_tokens']
            if line_id == 0:
                # Note this will only print for the first task seen if there are multiple tasks:
                print(f"Considering a max generation of {max_generation_tokens} tokens for training and eval.")
            label = int(json_obj['label'])  # verification label
            original_label = int(json_obj['original_label'])  # original classification task's labels
            taskCategoryInts.append(json_obj["taskCategory"])
            prompt = json_obj["prompt"]
            prompt_no_assistant = json_obj["prompt_no_assistant"]
            prompts_no_assistant.append(prompt_no_assistant)  # used for subsequent generation during training
            prefix_mask = json_obj["prefix_mask"]
            if json_obj["taskCategory"] == 0:  #'"}' since the JSON value is a string
                eos_token_id = 9092
            elif json_obj["taskCategory"] == 1:  # "}" since the JSON value is a bool
                eos_token_id = 29913
            else:
                assert False

            assert label in [0, 1], f"ERROR: Currently only binary preferences are implemented."
            if modelCategory == modelCategories.classification_with_generation__document_level:
                response = get_response(gen_ai_model, tokenizer, prompt=prompt_no_assistant,
                                        max_tokens=max_generation_tokens)
            else:
                response = get_response(gen_ai_model, tokenizer, prompt=prompt,
                                        max_tokens=2)

            text, logprobs, mlx_embeddings, force_decoded_input_tokens, generated_tokens = response
            if modelCategory == modelCategories.generation_force_decode__token_level:
                assert is_training, "Generative eval is via classification over document-level generation using " \
                                    "modelCategories.classification_with_generation__document_level"
                assert label == 1, "Constructed negatives do not participate in next-token fine-tuning."
                input_token_ids = np.array(force_decoded_input_tokens[-1]).tolist()  # these are the "prompt" tokens
                # need to extend tokens by label
                shifted_labels = []
                # label in {0,1}; however, note that in the current version, only label 1 is seen
                # for force-decoded generation since label 0 is skipped above
                offset = label * gen_ai_vocab
                token_index = 1
                assert input_token_ids[-3] == eos_token_id, input_token_ids
                ignore_prefix = True
                if ignore_prefix:
                    prefix_ignore_len = len(prefix_mask)
                    max_length_with_prefix = prefix_ignore_len + max_length
                else:
                    prefix_ignore_len = 0
                    max_length_with_prefix = max_length
                for shift_label in input_token_ids[1:-2]:  # note the [1:] to shift the label by 1; final token '"}' is at -3
                    if token_index < len(prefix_mask) and prefix_mask[token_index] == 1:
                        # original vocab for user and system messages and existing good generations (optional)
                        shifted_labels.append(shift_label)
                    else:
                        shifted_labels.append(shift_label + offset)
                    token_index += 1
                shifted_token_labels.extend(shifted_labels[prefix_ignore_len:max_length_with_prefix])
                length_considered = len(shifted_labels[prefix_ignore_len:max_length_with_prefix])

                original_labels.extend([original_label] * length_considered)
                labels.extend([label] * length_considered)
                final_token_indicators = [0.0] * length_considered
                final_token_indicators[-1] = 1.0
                end_of_document_indicators.extend(final_token_indicators)
                # We drop the trailing embedding (see 0:-1) to match shifted labels. leading 0 in [0, 0:-1, :]
                # is because
                # currently we are forwarding through the network with batch_size == 1; batch processing not currently implemented in the mlx code base
                input_embeddings = torch.tensor(np.array(mlx_embeddings[0][0, prefix_ignore_len:-3, :].astype(mx.float32), dtype=np.float32), dtype=torch.float32)
                # input_embeddings = torch.tensor(
                #     np.array(mlx_embeddings[0][0, 0:-1, :].astype(mx.float16), dtype=np.float16), dtype=torch.float16)
                # input_embeddings = torch.tensor(
                #     np.array(mlx_embeddings[0][0, 0:-1, :].astype(mx.float32), dtype=np.float32), dtype=torch.bfloat16)
                # embeddings[0].shape (1, 21, 3072)
                assert len(shifted_labels[prefix_ignore_len:max_length_with_prefix]) == input_embeddings[0:max_length, :].shape[0], \
                    f"prefill_step_size is too small: full embeddings won't be returned (note that this is currently " \
                    f"a quick approach to return the embeddings for our research example without making significant " \
                    f"changes to the existing mlx codebase; consider modifying generate() for " \
                    f"your use case: {document}, {prefix_ignore_len}, {len(shifted_labels)}, " \
                    f"{input_embeddings.shape}, {input_embeddings[0:max_length, :].shape[0]}"
                embedding_size = input_embeddings[0:max_length, :].shape[1]
                if False:  # can use this approach for running averages when partial decoding
                    assert False, 'add attributes'
                    global_embedding_size = embedding_size  # currently set to equal per-token LLM embedding size
                    composition_attributes_size = kComposition_attributes_size #top_logits_k * 2 * 2  # (2 for rejected and accepted distributions) * 2 for mean and token-level
                    embeddings.append(
                        torch.cat([get_running_average_embeddings(input_embeddings[0:max_length, :]), input_embeddings[0:max_length, :]], dim=1))
                else:
                    global_embedding_size = embedding_size  # currently set to equal per-token LLM embedding size
                    composition_attributes_size = kComposition_attributes_size #top_logits_k * 2 * 2  # (2 for rejected and accepted distributions) * 2 for mean and token-level
                    embeddings.append(input_embeddings[0:max_length, :])  # only cache LLM embedding for the token-level to save memory
                uuids.extend([json_obj["id"]] * length_considered)
                if return_text:
                    lines.extend([json_obj['document']]*length_considered)
            elif modelCategory in [modelCategories.classification_with_force_decoded_generation__document_level,
                                   modelCategories.classification_with_generation__document_level]:
                if modelCategory == modelCategories.classification_with_force_decoded_generation__document_level:
                    # all tokens are covered by the 'prompt'
                    # mlx_embeddings[0][0, -1, :] is for the first generated token
                    input_embeddings = torch.tensor(np.array(mlx_embeddings[0][0, 0:max_length, :].astype(mx.float32),
                                                             dtype=np.float32), dtype=torch.float32)
                    # parse prompt to find final index up to and *including* "}
                    # "} occurs at -3 for phi3 (assuming not truncated by max length) and also need to skip final index
                    # i.e., tokenizer.decode(mx.argmax(model.lm_head(mlx_embeddings[0][:,-4,:]), axis=-1).item()) should be eos_token_id
                    input_embeddings = input_embeddings[0:-3, :]
                elif modelCategory == modelCategories.classification_with_generation__document_level:
                    # first get 'prompt' embeddings. mlx_embeddings[0][0, -1, :] is for the first generated token
                    prompt_embeddings = torch.tensor(np.array(mlx_embeddings[0][0, :, :].astype(mx.float32),
                                                              dtype=np.float32), dtype=torch.float32)
                    try:
                        truncated_generated_tokens = list(generated_tokens[0:-1])  # skip final index since a duplicated token
                        eos_right_bracket_plus_one = len(truncated_generated_tokens) - truncated_generated_tokens[::-1].index(eos_token_id)
                        # next, embeddings for each of the generated tokens
                        generated_token_embeddings = [prompt_embeddings]
                        for emb_i in range(len(mlx_embeddings[0:-1])):
                            if emb_i > 0:
                                # first token id skipped since initial token in mlx_embeddings[0][0, -1, :]
                                if emb_i == eos_right_bracket_plus_one:
                                    break
                                generated_token_embeddings.append(
                                    torch.tensor(np.array(mlx_embeddings[emb_i][0, :, :].astype(mx.float32), dtype=np.float32),
                                                 dtype=torch.float32)
                                )
                        input_embeddings = torch.cat(generated_token_embeddings, dim=0)[0:max_length, :]
                    except:
                        print(f"WARNING: Parsing error at id {json_obj['id']}. "
                              f"This should rarely, if ever, occur. We assume JSON output.")
                        print(tokenizer.decode([x for x in generated_tokens]))
                        print(f"----")
                        input_embeddings = prompt_embeddings[0:max_length, :]

                original_labels.append(original_label)
                labels.append(label)
                embedding_size = input_embeddings.shape[1]
                global_embedding_size = embedding_size  # currently set to equal per-token LLM embedding size
                composition_attributes_size = kComposition_attributes_size #top_logits_k * 2 * 2  # (2 for rejected and accepted distributions) * 2 for mean and token-level
                # embeddings.append(torch.mean(input_embeddings, dim=0).unsqueeze(0))
                embeddings.append(
                    torch.cat([
                        torch.mean(input_embeddings, dim=0),
                        input_embeddings[-1, :]  # final token embedding
                    ], dim=0).unsqueeze(0))
                uuids.append(f'{json_obj["id"]}')
                uuid2idx[json_obj["id"]] = line_id
                if modelCategory == modelCategories.classification_with_generation__document_level:
                    generated_tokens = generated_tokens[0:-1]  # skip final index since a duplicated token
                    decoded_text = tokenizer.decode([x for x in generated_tokens])
                    # print(decoded_text)
                    original_task_accurcy_per_task_per_class, json_parse_error_by_task, task_prediction = \
                        parse_generation(decoded_text, original_label, json_obj["taskCategory"],
                                         original_task_accurcy_per_task_per_class, json_parse_error_by_task)
                    task_predictions.append(task_prediction)
                if return_text:
                    lines.append(json_obj['document'])
                    if modelCategory == modelCategories.classification_with_generation__document_level:
                        # decoded_text = tokenizer.decode([x for x in generated_tokens])
                        generated_lines.append(decoded_text)
                        # print(decoded_text)

            # if line_id >= 20: #1000:
            #     print(f"Temp breaking")
            #     break
            # if (0 <= line_id < 100) or (350 <= line_id < 450) :
            #     pass
            # else:
            #     continue
                # print(f"Temp breaking")
                # break
            line_id += 1

    embeddings = torch.cat(embeddings, dim=0)
    summary_stats = None
    if modelCategory == modelCategories.generation_force_decode__token_level or not is_training:
        assert not calculate_summary_stats
        summary_stats = None  # only updated for classification on the training set
    else:
        if calculate_summary_stats:
            if options.do_not_normalize_input_embeddings:
                summary_stats = {
                    constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: 0.0,
                    constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: 1.0
                }
            else:
                summary_stats = utils_model.get_embedding_summary_stats(embeddings, is_training)

        # normalize in forward, instead; embeddings = utils_model.normalize_embeddings(options, embeddings, summary_stats)
    print(f"\t{eval_label}Total existing instances considered: {embeddings.shape[0]}")
    if modelCategory == modelCategories.classification_with_generation__document_level:
        for task_label in range(len([taskCategories.sentiment, taskCategories.factcheck])):
            overall_accuracy_per_task = []
            for class_label in range(numberOfClasses):
                if len(original_task_accurcy_per_task_per_class[task_label][class_label]) > 0:
                    overall_accuracy_per_task.extend(list(original_task_accurcy_per_task_per_class[task_label][class_label]))
                    original_task_accurcy_per_task_per_class[task_label][class_label] = \
                        np.mean(original_task_accurcy_per_task_per_class[task_label][class_label])
                else:
                    original_task_accurcy_per_task_per_class[task_label][class_label] = 0.0
                print(f"\t{eval_label}Task: {taskCategories._fields[task_label]}, class {class_label}, mean accuracy: "
                      f"{original_task_accurcy_per_task_per_class[task_label][class_label]}")
            print(f"\t\t{eval_label}Task: {taskCategories._fields[task_label]}, Marginal accuracy: "
                  f"{np.mean(overall_accuracy_per_task) if len(overall_accuracy_per_task) > 0 else 0.0}")
        for task_label in range(len([taskCategories.sentiment, taskCategories.factcheck])):
            print(f"\t{eval_label}Task: {taskCategories._fields[task_label]}, parsing errors: {json_parse_error_by_task[task_label]}")

    return {"lines": lines,
            "generated_lines": generated_lines,
            "original_labels": original_labels,  # the original task labels
            "labels": labels,
            "taskCategoryInts": taskCategoryInts,
            "shifted_token_labels": shifted_token_labels,  # expanded (i.e., vocab * 2) token ids, shifted for next-token prediction
            "embeddings": embeddings,
            "embedding_size": embedding_size,
            "global_embedding_size": global_embedding_size,
            "composition_attributes_size": composition_attributes_size,
            "uuids": uuids,
            "end_of_document_indicators": end_of_document_indicators,
            "construct_negative_indicators": construct_negative_indicators,
            "prompts_no_assistant": prompts_no_assistant,
            "uuid2idx": uuid2idx,
            "original_task_accurcy_per_task_per_class": original_task_accurcy_per_task_per_class,
            "json_parse_error_by_task": json_parse_error_by_task,
            "task_predictions": task_predictions}, summary_stats


def get_gen_ai_model_lm_head_weights_file(gen_ai_model_lm_head_weights_file):
    try:
        return torch.load(gen_ai_model_lm_head_weights_file, weights_only=True, map_location=torch.device("cpu"))
    except:
        print(f"Gen AI LM head linear weights are missing. Exiting.")
        exit()

def _format_and_save_cached_embeddings(cache_directory, filename, meta_data, original_json_list):
    dataset_size = len(meta_data["labels"])
    assert len(meta_data["lines"]) == dataset_size
    assert meta_data["embeddings"].shape[0] == dataset_size
    json_list = []
    embedding_dimension = 0
    for instance_i in range(dataset_size):
        line_id = meta_data["uuid2idx"][meta_data["uuids"][instance_i]]
        json_obj = original_json_list[line_id]
        json_obj["embedding"] = meta_data["embeddings"][instance_i].detach().numpy().tolist()
        json_list.append(json_obj)
        if embedding_dimension == 0:
            embedding_dimension = len(json_obj['embedding'])
        else:
            assert embedding_dimension == len(json_obj['embedding'])
    filename_save_path_as_string = str(Path(cache_directory, filename).as_posix())
    utils_model.save_json_lines(filename_save_path_as_string, json_list)
    print(f"{filename} (with {dataset_size} lines) saved to {filename_save_path_as_string} "
          f"with embeddings of dimension {embedding_dimension}")


def _get_json_lines(filepath_with_name):
    json_list = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list


def _process_and_save_cached_embeddings(options, gen_ai_model, tokenizer, modelCategory, taskCategory, llmType,
                                        model=None, is_training=False, file_path=None):
    start_time = time.time()
    meta_data, _ = \
        get_metadata_lines_for_gen(options, gen_ai_model, tokenizer, options.max_length, str(file_path.as_posix()),
                                   calculate_summary_stats=False, is_training=is_training,
                                   taskCategory=None,
                                   modelCategory=modelCategory,
                                   top_logits_k=options.top_logits_k,
                                   model=model,
                                   return_text=True,
                                   llmType=llmType
                                   )
    json_list = _get_json_lines(str(file_path.as_posix()))
    _format_and_save_cached_embeddings(options.cache_directory, file_path.name, meta_data, json_list)
    print("Processing time", time.time() - start_time)


def cache_embeddings_for_classification(options, gen_ai_model, tokenizer, modelCategory, taskCategory, llmType,
                                        model=None):

    assert Path(options.cache_directory).exists()

    train_path = Path(options.input_training_set_file)
    calibration_path = Path(options.input_calibration_set_file)
    eval_path = Path(options.input_eval_set_file)
    if eval_path.exists():
        _process_and_save_cached_embeddings(options, gen_ai_model, tokenizer, modelCategory, taskCategory, llmType,
                                            model=model, is_training=False, file_path=eval_path)
    if not options.only_cache_eval:
        if train_path.exists():
            _process_and_save_cached_embeddings(options, gen_ai_model, tokenizer, modelCategory, taskCategory, llmType,
                                                model=model, is_training=True, file_path=train_path)
        if calibration_path.exists():
            _process_and_save_cached_embeddings(options, gen_ai_model, tokenizer, modelCategory, taskCategory, llmType,
                                                model=model, is_training=False, file_path=calibration_path)
