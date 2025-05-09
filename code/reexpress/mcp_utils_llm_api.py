# Copyright Reexpress AI, Inc. All rights reserved.

# LLM API calls and transformations for MCP server

import torch
import numpy as np
from pydantic import BaseModel
import time
import os

import constants

# env variables
USE_AZURE_01 = int(os.getenv("USE_AZURE_01", "1"))
if USE_AZURE_01 == 1:
    from openai import AzureOpenAI
    kAPI_VERSION = "2024-12-01-preview"
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=kAPI_VERSION,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    LOG_PROB_MODEL = os.getenv("GPT41_2025_04_14_AZURE_DEPLOYMENT_NAME")
    REASONING_MODEL = os.getenv("O4_MINI_2025_04_16_AZURE_DEPLOYMENT_NAME")
    EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_3_LARGE_AZURE_DEPLOYMENT_NAME")
else:
    from openai import OpenAI
    client = OpenAI()
    LOG_PROB_MODEL = "gpt-4.1-2025-04-14"
    REASONING_MODEL = "o4-mini-2025-04-16"
    EMBEDDING_MODEL = "text-embedding-3-large"


class ResponseVerificationWithConfidenceAndExplanation(BaseModel):
    verification_classification: bool
    confidence_in_classification: float
    short_explanation_for_classification_confidence: str


def get_logit_attributes(logprobs_content, probability_space=True, verbose=0) -> list[float]:
    field_checks = ["verification_classification\":", "confidence_in_classification\":", "short_explanation_for_classification_confidence\":"]
    next_indicators = ["\"confidence", "\"short"]
    running_string = ""
    field_probabilities = []
    for _ in range(len(field_checks)):
        field_probabilities.append([])
    field_i = 0
    for position_id, completion_token_position_value in enumerate(logprobs_content):
        for top_token_k, top_token in enumerate(completion_token_position_value.top_logprobs):
            if top_token_k == 0:
                if probability_space:
                    token_prob = np.exp(top_token.logprob)
                else:
                    token_prob = top_token.logprob
                if verbose >= 1:
                    print(position_id, top_token_k, top_token.token, token_prob)
                running_string += top_token.token
                valid_parse = False
                if field_i == 0:
                    valid_parse = field_checks[field_i] in running_string and field_checks[
                        field_i + 1] not in running_string and top_token.token.strip() in ["true", "false"]
                elif field_i == 1:
                    valid_parse = field_checks[field_i] in running_string and field_checks[
                        field_i + 1] not in running_string and (top_token.token.strip().isdigit() or top_token.token.strip() in ["."])
                elif field_i == len(field_checks) - 1:
                    valid_parse = field_checks[field_i] in running_string and top_token.token.strip() not in [""]  # ignore space characters
                if valid_parse:
                    if verbose >= 2:
                        print(f"\t\t###len(running_string): {len(running_string)}")
                    field_probabilities[field_i].append(token_prob)
                    if verbose >= 2:
                        print(f"\t\t\t///field_probabilities[field_i]: {field_probabilities[field_i]}")
                if field_i < len(next_indicators):
                    if next_indicators[field_i] in running_string:
                        field_i += 1
                if verbose >= 2:
                    print(f"field_i: {field_i}")
                    print(running_string)
    field_probability_averages = []
    for field_i in range(len(field_checks)):
        if field_i == 0:
            field_probability_averages.append(field_probabilities[field_i][0])
        elif field_i == 1:
            field_probability_averages.append(field_probabilities[field_i][0])
        elif field_i == len(field_checks) - 1:
            field_probability_averages.append(np.mean(field_probabilities[field_i][1:-1]).item())
    # also include an average final 10
    field_i = len(field_checks) - 1
    final_explanation_field_values = field_probabilities[field_i][1:-1]
    field_probability_averages.append(np.mean(final_explanation_field_values[-10:]).item())
    # also add quantiles:
    quantiles = np.quantile(final_explanation_field_values, [0.25, 0.5, 0.75])
    field_probability_averages.extend([float(x) for x in quantiles])
    # also add std:
    field_probability_averages.append(np.std(final_explanation_field_values).item())
    # final 3
    field_probability_averages.append(np.mean(final_explanation_field_values[-3:]).item())
    # final 1
    field_probability_averages.append(final_explanation_field_values[-1])
    return field_probability_averages


def get_document_attributes_from_reasoning(previous_query_and_response_to_verify_string: str) -> \
        dict[str, float | bool]:
    time.sleep(torch.abs(torch.randn(1)).item() / constants.SLEEP_CONSTANT)
    try:
        max_tokens = 25000
        messages_structure = [
                {"role": "system", "content": f"{constants.SYSTEM_MESSAGE_WITH_EXPLANATION}"},
                {"role": "user",
                 "content": f"{previous_query_and_response_to_verify_string}"}
            ]
        completion = client.beta.chat.completions.parse(
            model=REASONING_MODEL,
            messages=messages_structure,
            response_format=ResponseVerificationWithConfidenceAndExplanation,
            max_completion_tokens=max_tokens,
            reasoning_effort="medium",
            user="sdm_llm_reasoning_branching_v1",
            seed=0
        )
        verification_object = completion.choices[0].message.parsed
        verification_dict = {constants.VERIFICATION_CLASSIFICATION_KEY: verification_object.verification_classification,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: verification_object.confidence_in_classification,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: verification_object.short_explanation_for_classification_confidence}
        return verification_dict
    except:
        verification_dict = {constants.VERIFICATION_CLASSIFICATION_KEY: False,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: 0.01,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR}
        return verification_dict


def get_document_attributes(previous_query_and_response_to_verify_string: str) -> dict[str, float | bool]:
    time.sleep(torch.abs(torch.randn(1)).item() / constants.SLEEP_CONSTANT)
    try:
        max_tokens = 4096
        total_logprobs_to_consider = 1
        messages_structure = [
                {"role": "system", "content": f"{constants.SYSTEM_MESSAGE_WITH_EXPLANATION}"},
                {"role": "user",
                 "content": f"{previous_query_and_response_to_verify_string}"}
            ]
        completion = client.beta.chat.completions.parse(
            model=LOG_PROB_MODEL,
            messages=messages_structure,
            response_format=ResponseVerificationWithConfidenceAndExplanation,
            max_completion_tokens=max_tokens,
            logprobs=True,
            top_logprobs=total_logprobs_to_consider,
            temperature=0.0,
            user="sdm_llm_branching_v1",
            seed=0
        )
        verification_object = completion.choices[0].message.parsed
        attributes = get_logit_attributes(completion.choices[0].logprobs.content)
        verification_dict = {constants.VERIFICATION_CLASSIFICATION_KEY: verification_object.verification_classification,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: verification_object.confidence_in_classification,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: verification_object.short_explanation_for_classification_confidence,
                             constants.REEXPRESS_ATTRIBUTES_KEY: attributes}
        return verification_dict
    except:
        verification_dict = {constants.VERIFICATION_CLASSIFICATION_KEY: False,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: 0.01,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                             constants.REEXPRESS_ATTRIBUTES_KEY: None}
        return verification_dict


def get_embedding(document_text: str) -> list[float]:
    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=document_text,
        encoding_format="float",
        user="embed_llm_branching_v1",
    )
    full_embedding = embedding_response.data[0].embedding
    assert len(full_embedding) == 3072
    compressed_embedding = []
    start_index = 1344  # start index for compression
    compressed_embedding.extend(full_embedding[0:start_index])
    for offset in [64, 128, 256, 512, 768]:
        end_index = start_index+offset
        compressed_embedding.append(float(np.mean(full_embedding[start_index:end_index]).item()))
        start_index = end_index
    return compressed_embedding


def llm_api_controller(log_prob_model_explanation: str, reasoning_model_explanation: str):
    call_schedule = 0
    while True:
        try:
            log_prob_model_embedding = get_embedding(document_text=log_prob_model_explanation)
            time.sleep(torch.abs(torch.randn(1)).item() / 40)
            reasoning_model_embedding = get_embedding(document_text=reasoning_model_explanation)
            return log_prob_model_embedding, reasoning_model_embedding
        except:
            if call_schedule == 2:
                # additional final wait
                #call_schedule += 2
                #exception_backoff = 2 ** call_schedule + (torch.abs(torch.randn(1)).item()) # * 30)
                #exception_backoff = time.sleep(torch.abs(torch.randn(1)).item() / constants.SLEEP_CONSTANT)
                #time.sleep(exception_backoff)
                return None, None
            exception_backoff = 2 ** call_schedule + torch.abs(torch.randn(1)).item()
            time.sleep(exception_backoff)
            call_schedule += 1


def get_model_explanations(log_prob_model_verification_dict, reasoning_model_verification_dict):
    return log_prob_model_verification_dict[constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip(), \
        reasoning_model_verification_dict[constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip()
