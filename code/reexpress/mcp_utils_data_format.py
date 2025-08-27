# Copyright Reexpress AI, Inc. All rights reserved.

# utility functions for MCP server

import constants

import torch
import numpy as np

import json
import codecs

def get_confidence_soft_one_hot_list(is_verified, verbalized_confidence):
    # assert 0.0 <= verbalized_confidence <= 1.0
    if is_verified:
        return [0.0, float(verbalized_confidence)]
    else:
        return [float(verbalized_confidence), 0.0]


def construct_document_attributes_and_embedding(gpt5_model_verification_dict,
                                                gemini_model_verification_dict,
                                                model_embedding):
    # | GPT-5 model soft one hot by verbalized uncertainty | gemini model soft one hot by verbalized uncertainty
    attributes = [0.0, 0.0, 0.0, 0.0]  # [GPT-5 class 0; GPT-5 class 1; Gemini class 0; Gemini class 1]
    is_verified_gpt5 = gpt5_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY]
    confidence_gpt5 = gpt5_model_verification_dict[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_soft_one_hot_list_gpt5 = get_confidence_soft_one_hot_list(is_verified_gpt5, confidence_gpt5)
    attributes[0:2] = confidence_soft_one_hot_list_gpt5

    is_verified_gemini = gemini_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY]
    confidence_gemini = gemini_model_verification_dict[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_soft_one_hot_list_gemini = get_confidence_soft_one_hot_list(is_verified_gemini, confidence_gemini)
    attributes[2:4] = confidence_soft_one_hot_list_gemini

    assert len(attributes) == constants.EXPECTED_ATTRIBUTES_LENGTH
    assert len(model_embedding) == constants.EXPECTED_EMBEDDING_SIZE

    reexpression_input = torch.tensor(model_embedding + attributes).unsqueeze(0)
    return reexpression_input
