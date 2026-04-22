# Copyright Reexpress AI, Inc. All rights reserved.

# utility functions for MCP server

import constants

import torch


def get_confidence_signed_indicator_list(is_verified, verbalized_confidence):
    # v2.3.0 format. Note that verbalized_confidence <= 0 for an activated index, which is rare, remains 0.
    if is_verified:
        return [0.0, 1.0] if float(verbalized_confidence) > 0 else [0.0, 0.0]
    else:
        return [-1.0, 0.0] if float(verbalized_confidence) > 0 else [0.0, 0.0]


def construct_document_attributes_and_embedding(gpt5_model_verification_dict,
                                                gemini_model_verification_dict,
                                                model_embedding):
    # | GPT-5 model soft one hot by verbalized uncertainty | gemini model soft one hot by verbalized uncertainty
    attributes = [0.0, 0.0, 0.0, 0.0]  # [GPT-5 class 0; GPT-5 class 1; Gemini class 0; Gemini class 1]
    is_verified_gpt5 = gpt5_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY]
    confidence_gpt5 = gpt5_model_verification_dict[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_signed_indicator_list_gpt5 = \
        get_confidence_signed_indicator_list(is_verified_gpt5, confidence_gpt5)
    attributes[0:2] = confidence_signed_indicator_list_gpt5

    is_verified_gemini = gemini_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY]
    confidence_gemini = gemini_model_verification_dict[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_signed_indicator_list_gemini = \
        get_confidence_signed_indicator_list(is_verified_gemini, confidence_gemini)
    attributes[2:4] = confidence_signed_indicator_list_gemini

    assert len(attributes) == constants.EXPECTED_ATTRIBUTES_LENGTH
    assert len(model_embedding) == constants.EXPECTED_EMBEDDING_SIZE

    reexpression_input = torch.tensor(model_embedding + attributes).unsqueeze(0)
    return reexpression_input
