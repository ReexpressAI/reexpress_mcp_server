# Copyright Reexpress AI, Inc. All rights reserved.

from typing import Any, Callable, List, Tuple
import json
import os
import numpy as np

import argparse
import time
from pathlib import Path
import codecs

import torch

import data_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


model_path = "ibm-granite/granite-3.3-8b-instruct"
# device = "mps"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)
set_seed(42)
kSLEEP_CONSTANT = 40
VERIFICATION_CLASSIFICATION_KEY = "verification_classification"
CONFIDENCE_IN_CLASSIFICATION_KEY = "confidence_in_classification"
SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY = "short_explanation_for_classification_confidence"

FIELD_PROBABILITIES_KEY = "field_probabilities"
LOG_PROB_MODEL_RESPONSE_KEY = "LOG_PROB_MODEL"
REASONING_MODEL_RESPONSE_KEY = "REASONING_MODEL"
GEMINI_2_5_PRO_MODEL_RESPONSE_KEY = "GEMINI_2_5_PRO_MODEL"

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 8194


def get_agreement_model_embedding(document_text: str): # -> list[float]:
    conv = [{"role": "user",
             "content": document_text}]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=False,
                                              return_dict=True, add_generation_prompt=True).to(device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    hidden_states = outputs.hidden_states
    scores = outputs.scores
    no_id = tokenizer.vocab["No"]
    yes_id = tokenizer.vocab["Yes"]
    probs = torch.softmax(scores[0], dim=-1)
    # average of all (across tokens) final hidden states :: final token hidden state (here this corresponds to the hidden state of the linear layer that determines the No/Yes classification) :: no_prob :: yes_prob
    embedding = torch.cat([
        torch.mean(hidden_states[0][-1][0], dim=0).unsqueeze(0),
        hidden_states[0][-1][0][-1, :].unsqueeze(0),
        probs[0:1, no_id].unsqueeze(0),
        probs[0:1, yes_id].unsqueeze(0)
    ], dim=-1)
    embedding = [float(x) for x in embedding[0].cpu().numpy().tolist()]
    assert len(embedding) == EXPECTED_EMBEDDING_SIZE
    agreement_classification = probs[0:1, no_id] < probs[0:1, yes_id]
    return embedding, agreement_classification.item()


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj["id"])
    return existing_ids


def construct_embedding_streaming(options):
    count_incomplete_responses = 0
    output_file = options.output_file
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    instance_i = -1
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            instance_i += 1
            if instance_i % 25000 == 0:
                print(f"Currently processing instance {instance_i}")
            if json_obj["id"] in existing_ids:
                continue
            try:
                embedding, _ = get_agreement_model_embedding(document_text=json_obj["agreement_prompt"])
                json_obj[REEXPRESS_EMBEDDING_KEY] = embedding
            except:
                json_obj[REEXPRESS_EMBEDDING_KEY] = [0.0] * EXPECTED_EMBEDDING_SIZE
                count_incomplete_responses += 1
            data_utils.save_by_appending_json_lines(output_file, [json_obj])
            existing_ids.add(json_obj["id"])
    print(f"Count of documents with embedding set to 0's: {count_incomplete_responses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--output_file", default="", help="")

    options = parser.parse_args()

    start_time = time.time()
    construct_embedding_streaming(options)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
