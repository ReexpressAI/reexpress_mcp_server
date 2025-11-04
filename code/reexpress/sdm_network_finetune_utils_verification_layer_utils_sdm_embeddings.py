# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import sdm_network_constants
import re


def get_verification_embedding(model, input_ids): #, tokenizer):
    EXPECTED_EMBEDDING_SIZE = 6144
    # Ensure input_ids is on the model's device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,  # Ensure batch dimension
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=False  # This avoids the caching/gradient checkpointing conflict
        )
        hidden_states = outputs.hidden_states
        scores = outputs.scores
        # # Here, the starting no has a prefix underscore. Other models may differ. tokenizer.convert_ids_to_tokens([1939])
        # no_id = tokenizer.vocab['▁No']
        # yes_id = tokenizer.vocab['▁Yes']  # tokenizer.convert_ids_to_tokens([3869])
        # In this case, there is no prefix underscore, since there is no space after the closing XML tag.
        # no_id = tokenizer.vocab['No']
        # yes_id = tokenizer.vocab['Yes']
        probs = torch.softmax(scores[0], dim=-1)
        # For reference, this reproduces the final output as input to the softmax:
        # model.lm_head(hidden_states[0][-1][0][-1, :])

        # The embeddings are as follows:
        # average of all (across tokens) final hidden states ::
        # final token hidden state (here this corresponds to the hidden state taken as input to the linear layer
        # that determines the vocabulary probabilities)
        embedding = torch.cat([
            torch.mean(hidden_states[0][-1][0], dim=0).unsqueeze(0),
            hidden_states[0][-1][0][-1, :].unsqueeze(0)
        ], dim=-1).to(torch.float32)
        # embedding = [float(x) for x in embedding[0].cpu().numpy().tolist()]
        assert embedding.shape[1] == EXPECTED_EMBEDDING_SIZE
        assert embedding.shape[0] == 1
        # return embedding.detach().cpu()

    return embedding.detach().cpu()
        # # The attributes are as follows:
        # # no_prob :: yes_prob
        # attributes = torch.cat([probs[0:1, no_id].unsqueeze(0),
        #                         probs[0:1, yes_id].unsqueeze(0)], dim=-1)
        # attributes = [float(x) for x in attributes[0].cpu().numpy().tolist()]
        # assert len(attributes) == 2
        # logits = torch.cat([scores[0][0:1, no_id].unsqueeze(0),
        #                     scores[0][0:1, yes_id].unsqueeze(0)], dim=-1)
        # # We also save the raw logits which we will use for comparisons to post-hoc classification methods. (E.g.,
        # # for temperature scaling, we need the raw logits before the softmax.)
        # logits = [float(x) for x in logits[0].cpu().numpy().tolist()]
        # assert len(logits) == 2
        # llm_classification = probs[0:1, no_id] < probs[0:1, yes_id]
        # return embedding, attributes, logits, int(llm_classification.item())


def extract_sentences_from_generation(generated_text):
    """Extract sentences and verification status from generated text."""
    sentence_pattern = r'<sentence>(.*?)</sentence>'
    verified_pattern = r'<verified>(.*?)</verified>'

    sentences = re.findall(sentence_pattern, generated_text, re.DOTALL)
    verifications = re.findall(verified_pattern, generated_text, re.DOTALL)

    valid_sentences = []
    valid_verifications = []

    for i in range(len(sentences)):
        if i < len(verifications):
            valid_sentences.append(sentences[i].strip())
            is_verified = verifications[i].strip().lower() == "yes"
            valid_verifications.append(is_verified)
            if is_verified:
                break  # Stop at first "Yes"

    return valid_sentences, valid_verifications


def generate_formatted_assistant_response_from_output(sentences: list[str], verifications: list[bool]):
    assert len(sentences) == len(verifications)
    response = ""
    verification_response = ""
    for i, sentence in enumerate(sentences):
        is_verified = verifications[i]
        answer_string = "Yes" if is_verified else "No"
        new_full_response = f"<sentence>{sentence}</sentence>\n<verified>{answer_string}</verified>\n\n"
        response += new_full_response
        if i == len(sentences) - 1:
            # Leave off final classification
            verification_response += f"<sentence>{sentence}</sentence>\n<verified>"
        else:
            verification_response += new_full_response
    return response, verification_response


def generate_formatted_assistant_response_from_malformed_output(generated_text: str):
    # Malformed output always gets a trailing No. This can also be used with well-formed responses for which the
    # generated sentence is known to be incorrect. This should not, however, be used to CONSTRUCT RESPONSES
    # with well-formed responses for which the generated sentence is correct, because the resulting classification
    # label will be incorrect. HOWEVER, the verification_response does not include the final label, so this IS used
    # as the general formatting for the VERIFICATION RESPONSES, regardless of the final control tokens.
    generated_text = generated_text.strip()
    negative_verification_string = f"<verified>No</verified>"
    positive_verification_string = f"<verified>Yes</verified>"
    if generated_text.endswith(negative_verification_string):
        response = f"{generated_text}\n\n"
        verification_response = generated_text[:-len(f"No</verified>")]
    elif generated_text.endswith(positive_verification_string):
        response = f"{generated_text[:-len(positive_verification_string)] + negative_verification_string}\n\n"
        verification_response = generated_text[:-len(f"Yes</verified>")]
    else:
        response = f"{generated_text}" + "\n" + negative_verification_string + "\n\n"
        verification_response = generated_text + "\n" + f"<verified>"
    return response, verification_response


def get_ids_from_prompt_text_and_assistant_response(tokenizer, prompt_text, assistant_response, model_max_length=2048):
    conv = sdm_network_constants.get_conv(prompt_text)
    conv.append({"role": "assistant", "content": assistant_response})

    # Tokenize the conversation
    encoding = tokenizer.apply_chat_template(
        conv,
        return_tensors="pt",
        max_length=model_max_length,
        padding=False,
        truncation=True,
        return_dict=True,
        add_generation_prompt=False  # We have the assistant response
    )

    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    return input_ids, attention_mask

