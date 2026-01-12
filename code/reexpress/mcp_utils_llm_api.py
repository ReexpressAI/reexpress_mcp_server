# Copyright Reexpress AI, Inc. All rights reserved.

# LLM API calls and transformations for MCP server

import torch
import numpy as np
from pydantic import BaseModel
import time
import os

import constants
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types

model_path = "ibm-granite/granite-3.3-8b-instruct"

try:
    device = str(os.getenv("MCP_SERVER_AGREEMENT_MODEL_DEVICE",
                           default=constants.MCP_SERVER_AGREEMENT_MODEL_DEVICE__DEFAULT))
    MCP_SERVER_AGREEMENT_MODEL_MAX_CHARACTER_LENGTH = int(
        os.getenv("MCP_SERVER_AGREEMENT_MODEL_MAX_CHARACTER_LENGTH",
                  default=constants.MCP_SERVER_AGREEMENT_MODEL_MAX_CHARACTER_LENGTH__DEFAULT))
except:
    device = constants.MCP_SERVER_AGREEMENT_MODEL_DEVICE__DEFAULT
    MCP_SERVER_AGREEMENT_MODEL_MAX_CHARACTER_LENGTH = \
        constants.MCP_SERVER_AGREEMENT_MODEL_MAX_CHARACTER_LENGTH__DEFAULT

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)


# env variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
google_client = genai.Client(api_key=GEMINI_API_KEY)  # can alternatively replace with a Vertex AI deployment

GEMINI_MODEL="gemini-3-pro-preview"

USE_AZURE_01 = int(os.getenv("USE_AZURE_01", "1"))
if USE_AZURE_01 == 1:
    from openai import AzureOpenAI
    kAPI_VERSION = "2024-12-01-preview"
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=kAPI_VERSION,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    GPT5_MODEL = os.getenv("GPT_5_2_MODEL_2025_12_11_AZURE_DEPLOYMENT")
else:
    from openai import OpenAI
    client = OpenAI()
    GPT5_MODEL = "gpt-5.2-2025-12-11"


class ResponseVerificationWithConfidenceAndExplanationAndSummary(BaseModel):
    short_summary_of_original_question_and_response: str
    verification_classification: bool
    confidence_in_classification: float
    short_explanation_for_classification_confidence: str


def get_document_attributes_from_gpt5(previous_query_and_response_to_verify_string: str) -> \
        dict[str, float | bool]:
    time.sleep(torch.abs(torch.randn(1)).item() / constants.SLEEP_CONSTANT)
    try:
        max_tokens=100000
        messages_structure = [
            {"role": "developer", "content": f"{constants.GPT_5_SYSTEM_MESSAGE.strip()}"},
            {"role": "user",
             "content": f"{previous_query_and_response_to_verify_string}"}
        ]
        completion = client.beta.chat.completions.parse(
            model=GPT5_MODEL,
            messages=messages_structure,
            response_format=ResponseVerificationWithConfidenceAndExplanationAndSummary,
            max_completion_tokens=max_tokens,
            reasoning_effort="high",
            user="sdm_llm_reasoning_branching_v1",
            seed=0
        )
        verification_object = completion.choices[0].message.parsed
        verification_dict = {constants.SHORT_SUMMARY_KEY: verification_object.short_summary_of_original_question_and_response,
                             constants.VERIFICATION_CLASSIFICATION_KEY: verification_object.verification_classification,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: verification_object.confidence_in_classification,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: verification_object.short_explanation_for_classification_confidence,
                             constants.LLM_API_ERROR_KEY: False}
    except:
        verification_dict = {constants.SHORT_SUMMARY_KEY: "",
                             constants.VERIFICATION_CLASSIFICATION_KEY: False,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: 0.01,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                             constants.LLM_API_ERROR_KEY: True}
        return verification_dict
    return verification_dict


def get_document_attributes_from_gemini_reasoning(previous_query_and_response_to_verify_string: str) -> \
        dict[str, float | bool]:
    time.sleep(torch.abs(torch.randn(1)).item() / constants.SLEEP_CONSTANT)
    try:
        max_tokens=65535
        response = google_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=previous_query_and_response_to_verify_string,
            config=types.GenerateContentConfig(
                system_instruction=constants.SYSTEM_MESSAGE_WITH_EXPLANATION.strip(),
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.HIGH,
                    include_thoughts=False
                ),
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch(),
                        code_execution=types.ToolCodeExecution(),
                        url_context=types.UrlContext(),
                    )
                ],
                response_mime_type="application/json",
                response_schema={"type": "OBJECT", "properties": {"verification_classification": {"type": "BOOLEAN"},
                                                                  "confidence_in_classification": {"type": "NUMBER"},
                                                                  "short_explanation_for_classification_confidence": {
                                                                      "type": "STRING"}},
                                 "required": ["verification_classification", "confidence_in_classification",
                                              "short_explanation_for_classification_confidence"]},
                response_modalities=["TEXT"],
                temperature=1.0,
                max_output_tokens=max_tokens,
                seed=0)
        )
        verification_object = response.parsed
        verification_dict = {constants.VERIFICATION_CLASSIFICATION_KEY:
                                 bool(verification_object[constants.VERIFICATION_CLASSIFICATION_KEY]),
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY:
                                 float(verification_object[constants.CONFIDENCE_IN_CLASSIFICATION_KEY]),
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY:
                                 str(verification_object[constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY]),
                             constants.LLM_API_ERROR_KEY: False}
    except:
        verification_dict = {constants.VERIFICATION_CLASSIFICATION_KEY: False,
                             constants.CONFIDENCE_IN_CLASSIFICATION_KEY: 0.01,
                             constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY: constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                             constants.LLM_API_ERROR_KEY: True}
        return verification_dict
    return verification_dict


def get_agreement_model_embedding(document_text: str):
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
    assert len(embedding) == constants.EXPECTED_EMBEDDING_SIZE
    agreement_classification = probs[0:1, no_id] < probs[0:1, yes_id]
    return embedding, agreement_classification.item()


def get_model_explanations_formatted_as_binary_agreement_prompt(gpt5_model_summary,
                                                                gpt5_model_explanation,
                                                                gemini_model_explanation) -> str:
    if gpt5_model_summary != "":
        topic_string = f"<topic> {gpt5_model_summary} </topic> "
    else:
        topic_string = ""
    formatted_output_string = f"{topic_string}Do the following model explanations agree that the response is correct? <model1_explanation> {gpt5_model_explanation} </model1_explanation> <model2_explanation> {gemini_model_explanation} </model2_explanation> Yes or No?"
    return formatted_output_string



def llm_api_controller(gpt5_model_summary: str, gpt5_model_explanation: str,
                       gemini_model_explanation: str):
    try:
        # Hard truncate by max allowed character count, with strict priority:
        # gpt5_model_explanation first, then gemini_model_explanation, then summary.
        # This is intended to put a hard constraint on memory use of the on-device model. Adjust as applicable
        # via the corresponding environment variable.
        remaining_max_length_counter = MCP_SERVER_AGREEMENT_MODEL_MAX_CHARACTER_LENGTH
        gpt5_model_explanation_filtered = gpt5_model_explanation[0:max(0, remaining_max_length_counter)]
        remaining_max_length_counter -= len(gpt5_model_explanation_filtered)
        gemini_model_explanation_filtered = gemini_model_explanation[0:max(0, remaining_max_length_counter)]
        remaining_max_length_counter -= len(gemini_model_explanation_filtered)
        gpt5_model_summary_filtered = gpt5_model_summary[0:max(0, remaining_max_length_counter)]

        prompt = get_model_explanations_formatted_as_binary_agreement_prompt(gpt5_model_summary_filtered,
                                                                             gpt5_model_explanation_filtered,
                                                                             gemini_model_explanation_filtered)
        agreement_model_embedding, agreement_model_classification = \
            get_agreement_model_embedding(document_text=prompt)
        return agreement_model_embedding, agreement_model_classification
    except:
        return None, None


def get_model_explanations(gpt5_model_verification_dict,
                           gemini_model_verification_dict):
    return gpt5_model_verification_dict[constants.SHORT_SUMMARY_KEY].strip(), \
        gpt5_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY], \
        gpt5_model_verification_dict[constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip(), \
        gemini_model_verification_dict[constants.VERIFICATION_CLASSIFICATION_KEY], \
        gemini_model_verification_dict[constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip()
