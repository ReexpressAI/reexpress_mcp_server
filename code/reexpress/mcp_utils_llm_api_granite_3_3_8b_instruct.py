# Copyright Reexpress AI, Inc. All rights reserved.

# Local Embedding from "ibm-granite/granite-3.3-8b-instruct" for MCP server

import torch
import os

import constants
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def get_agreement_model_embedding(document_text: str):
    with torch.no_grad():
        conv = [{"role": "user", "content": document_text}]
        input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=False,
                                                  return_dict=True, add_generation_prompt=True).to(device)

        # Capture only the final layer's (post-norm) hidden state via a forward hook on the
        # model's final norm module. This replaces output_hidden_states=True, which would
        # otherwise retain the hidden states of every transformer block even though we only
        # consume the last one. For a many-layer multi-billion-parameter model this reduces
        # peak activation memory roughly by a factor of the layer count.
        captured_final_hidden = {}

        def _capture_final_hidden(_module, _inputs, output):
            # output shape: (batch, seq_len, hidden_size). This is equivalent to
            # hidden_states[-1] when output_hidden_states=True is requested.
            captured_final_hidden["value"] = output.detach()

        hook_handle = model.model.norm.register_forward_hook(_capture_final_hidden)
        try:
            outputs = model.generate(
                **input_ids,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        finally:
            hook_handle.remove()

        # With max_new_tokens=1, the hook fires exactly once during the prompt forward pass.
        # Select batch index 0 to match the original hidden_states[0][-1][0] indexing.
        final_hidden_state = captured_final_hidden["value"][0]  # (seq_len, hidden_size)

        # ## Start Reference
        # # The above assumes max_new_tokens=1. With max_new_tokens=2, to, for example, capture the mean over all
        # # hidden states and the final hidden state corresponding to the final token classification, use:
        # captured_final_hidden = []
        # def _capture_final_hidden(_module, _inputs, output):
        #     captured_final_hidden.append(output.detach())
        # handle = model.model.norm.register_forward_hook(_capture_final_hidden)
        # # ...
        # final_hidden_step1 = captured_final_hidden[1][0]
        # embedding = torch.cat([
        #     torch.mean(final_hidden_step1, dim=0).unsqueeze(0),
        #     final_hidden_step1[-1, :].unsqueeze(0)
        # ], dim=-1).to(torch.float32)
        # ## End Reference

        scores = outputs.scores
        no_id = tokenizer.vocab["No"]
        yes_id = tokenizer.vocab["Yes"]
        probs = torch.softmax(scores[0], dim=-1)
        # max of all (across tokens) final hidden states ::
        # average of all (across tokens) final hidden states ::
        # final token hidden state (here this corresponds to the hidden state of the linear layer that determines
        # the No/Yes classification) ::
        # no logit :: yes logit ::
        # no_prob (normalized over the vocabulary) :: yes_prob (normalized over the vocabulary)
        embedding = torch.cat([
            torch.max(final_hidden_state, dim=0).values.unsqueeze(0),
            torch.mean(final_hidden_state, dim=0).unsqueeze(0),
            final_hidden_state[-1, :].unsqueeze(0),
            scores[0][0:1, no_id].unsqueeze(0),
            scores[0][0:1, yes_id].unsqueeze(0),
            probs[0:1, no_id].unsqueeze(0),
            probs[0:1, yes_id].unsqueeze(0)
        ], dim=-1)
        embedding = [float(x) for x in embedding[0].cpu().numpy().tolist()]
        assert len(embedding) == constants.EXPECTED_LOCAL_LM_EMBEDDING_SIZE
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


def get_local_embedding_for_agreement_prompt(gpt5_model_summary: str, gpt5_model_explanation: str,
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
