# Copyright Reexpress AI, Inc. All rights reserved.

# utility functions for MCP server

import constants

import torch
import numpy as np

import json
import codecs

def get_confidence_soft_one_hot_list(is_verified, verbalized_confidence):
    assert 0.0 <= verbalized_confidence <= 1.0
    if is_verified:
        return [0.0, float(verbalized_confidence)]
    else:
        return [float(verbalized_confidence), 0.0]


def construct_document_attributes_and_embedding(log_prob_model_verification_dict, reasoning_model_verification_dict):
    # log probability model processed log probabilities | log probability model soft one hot by verbalized uncertainty | reasoning model soft one hot by verbalized uncertainty
    # (negative/unverified | positive/verified)
    is_verified_log_prob = log_prob_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY]
    confidence_log_prob = log_prob_model_verification_dict[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_soft_one_hot_list_log_prob = get_confidence_soft_one_hot_list(is_verified_log_prob, confidence_log_prob)
    unprocessed_attributes = log_prob_model_verification_dict[constants.REEXPRESS_ATTRIBUTES_KEY]
    assert len(unprocessed_attributes) == constants.EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH
    attributes = np.zeros(constants.EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH * 2)
    if is_verified_log_prob:
        attributes[constants.EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH:] = unprocessed_attributes
    else:
        attributes[0:constants.EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH] = unprocessed_attributes
    reexpression_attributes = [float(x) for x in attributes.tolist()]
    reexpression_attributes.extend(confidence_soft_one_hot_list_log_prob)
    is_verified_reasoning = reasoning_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY]
    confidence_reasoning = reasoning_model_verification_dict[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_soft_one_hot_list_reasoning = get_confidence_soft_one_hot_list(is_verified_reasoning, confidence_reasoning)
    reexpression_attributes.extend(confidence_soft_one_hot_list_reasoning)

    assert len(reexpression_attributes) == constants.EXPECTED_ATTRIBUTES_LENGTH

    assert len(log_prob_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY]) == constants.EXPECTED_EMBEDDING_SIZE
    assert len(reasoning_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY]) == constants.EXPECTED_EMBEDDING_SIZE
    embedding = log_prob_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY] + reasoning_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY]

    reexpression_input = torch.tensor(embedding + reexpression_attributes).unsqueeze(0)
    return reexpression_input