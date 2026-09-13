# Copyright Reexpress AI, Inc. All rights reserved.

# Local embedding from the 4-bit MLX conversion of google/gemma-4-31B-it for the MCP server (Apple silicon only).
# MLX port of mcp_utils_llm_api_gemma_4_e4b_it.py, fixed to one checkpoint:
#   mlx-community/gemma-4-31b-it-4bit   (18.4 GB; 4-bit affine, group size 64; converted from google/gemma-4-31B-it)
# Same prompt, truncation policy, and embedding layout ([max] :: mean :: last, 3 x 5376 = 16128 dims) as the PyTorch
# module, so the two are interchangeable at the API level but NOT at the vector level: 4-bit weights give different
# hidden states than bf16, so anything fit on one must be served from the same one.
#
# Requires: pip install -U mlx-lm   (Gemma 4 support landed in mlx-lm 0.31.x; use a current release)

import functools
import logging
import types

import mlx.core as mx
from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "mlx-community/gemma-4-31b-it-4bit"

# Standard system prompt from the Gemma 4 model card. Thinking is toggled by a `<|think|>` token at the start of the
# system prompt, so it must not appear here; enable_thinking=False is also passed to the chat template.
SYSTEM_PROMPT = "You are a helpful assistant."

# With thinking disabled, the 31B still opens the model turn with an empty thought block before the final answer:
#     <|turn>model\n<|channel>thought\n<channel|>Yes
# The checkpoint's chat template emits this block at the end of the generation prompt, so the last prompt position is
# the one that predicts Yes/No. _load checks the rendered prompt and appends the block itself if a template revision
# does not.
EMPTY_THOUGHT_BLOCK = "<|channel>thought\n<channel|>"

LABELS = ("Yes", "No")

# Load-time probe: an explanation that plainly agrees, so the expected first generated token is Yes.
PROBE_SUMMARY = "The response says that 2 + 2 = 4."
PROBE_EXPLANATION = "The arithmetic is right, so the response is correct."

# _resolve_head reproduces the full model's next-token logits from decoder hidden states; this is the largest
# bf16 discrepancy accepted (logits are softcapped to +-30, where a bf16 ulp is 0.125-0.25).
HEAD_CHECK_TOLERANCE = 1.0

# Token-level truncation policy (see _truncate_fields).
MIN_SUMMARY_TOKENS = 512          # reserved for the summary when the explanation would otherwise crowd it out
TRUNCATION_HEAD_FRACTION = 0.25   # keep this fraction of a truncated field from its start, the rest from its end
TRUNCATION_MARKER = " [...] "
TOKEN_MARGIN = 16                 # slack for the marker and decode/re-encode boundary shifts


def get_model_explanations_formatted_as_binary_agreement_prompt(lm_model_summary,
                                                                lm_model_explanation) -> str:
    formatted_output_string = f"<topic> {lm_model_summary} </topic> Does the following model explanation agree that the response is correct? <model_explanation> {lm_model_explanation} </model_explanation> Answer with a single word: Yes or No?"
    return formatted_output_string


def _conversation(document_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": document_text},
    ]


def _hf(tokenizer):
    """The underlying Hugging Face tokenizer: mlx_lm.load returns it inside a TokenizerWrapper."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


def _render_prompt(tokenizer, document_text: str, answer_prefix: str = "") -> str:
    """Chat-templated prompt (with generation prompt) followed by the answer prefix, if one is needed."""
    return _hf(tokenizer).apply_chat_template(
        _conversation(document_text), tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) + answer_prefix


def _prompt_ids(tokenizer, document_text: str, answer_prefix: str = "") -> list:
    """Token ids of the rendered prompt (the template already includes <bos>)."""
    return _hf(tokenizer).encode(_render_prompt(tokenizer, document_text, answer_prefix), add_special_tokens=False)


def _resolve_label_token_ids(tokenizer, prompt_text: str, label: str) -> list:
    """Ids of `label` tokenized exactly as it would be generated at the start of the model's answer.

    Resolved in context (rendered prompt + label) rather than via the vocab, so tokenizer-specific prefix variants
    (e.g. '▁Yes' vs 'Yes') and multi-token spellings are picked up automatically.
    """
    hf = _hf(tokenizer)
    prompt_ids = hf.encode(prompt_text, add_special_tokens=False)
    full_ids = hf.encode(prompt_text + label, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids, f"Appending {label!r} changed the prompt tokenization."
    label_ids = full_ids[len(prompt_ids):]
    assert label_ids, f"{label!r} produced no tokens in context."
    return label_ids


def _softcap(logits, cap):
    return mx.tanh(logits / cap) * cap if cap else logits


def _forward(lm, ids: list):
    """One prompt forward pass. Returns the post-final-norm hidden states (seq_len, hidden_size) -- the input to
    the LM head, equivalent to hidden_states[-1] in the PyTorch module -- and the next-token logits (vocab_size,)
    in float32. The LM head is applied only at the last position, so the 262K-vocab logits are never materialized
    for the whole prompt."""
    hidden = lm.decoder(mx.array(ids)[None])
    logits = _softcap(lm.head(hidden[:, -1:]), lm.softcap)[0, -1].astype(mx.float32)
    mx.eval(hidden, logits)
    return hidden[0], logits


def _detect_softcap(model, text_lm):
    for holder in (text_lm, getattr(text_lm, "args", None)):
        cap = getattr(holder, "final_logit_softcapping", None)
        if cap:
            return float(cap)
    text_config = getattr(getattr(model, "args", None), "text_config", None) or {}
    return text_config.get("final_logit_softcapping")


def _resolve_head(model, text_lm, decoder, probe_ids: list):
    """Find the (LM head, softcap) pair that reproduces the full model's next-token logits from the decoder's
    output. mlx-lm has no forward hooks, so the decoder is called directly; this check confirms that its output is
    the post-final-norm state and that the head/softcap wiring matches, and it raises rather than silently
    returning the wrong vectors if mlx-lm's gemma4_text layout ever changes."""
    x = mx.array(probe_ids)[None]
    reference = model(x)[0, -1].astype(mx.float32)  # full model, short probe: full logits are cheap here
    hidden = decoder(x)
    mx.eval(reference, hidden)
    heads = []
    if hasattr(text_lm, "lm_head"):
        heads.append(("lm_head", text_lm.lm_head))
    heads.append(("embed_tokens.as_linear", decoder.embed_tokens.as_linear))  # tie_word_embeddings
    attempts = []
    for cap in dict.fromkeys([_detect_softcap(model, text_lm), None]):
        for name, head in heads:
            logits = _softcap(head(hidden[:, -1:]), cap)[0, -1].astype(mx.float32)
            diff = float(mx.max(mx.abs(logits - reference)))
            attempts.append(f"{name}/softcap={cap}: max|dlogit|={diff:.3f}")
            if diff <= HEAD_CHECK_TOLERANCE and int(mx.argmax(logits)) == int(mx.argmax(reference)):
                logger.info("LM head resolved: %s, softcap=%s (max |dlogit| vs full model %.3f)", name, cap, diff)
                return head, cap
    raise RuntimeError("Could not reproduce the model's next-token logits from decoder hidden states; mlx-lm's "
                       "gemma4_text layout may have changed. Attempts: " + "; ".join(attempts))


@functools.lru_cache(maxsize=1)
def _load(model_path: str):
    """Load once per model_path and cache; subsequent calls are free. MLX runs on the unified-memory GPU."""
    model, tokenizer = mlx_load(model_path)
    hf = _hf(tokenizer)
    assert getattr(hf, "chat_template", None), (
        f"{model_path} loaded without a chat template; update mlx-lm/transformers so chat_template.jinja is picked up."
    )
    # mlx-lm's gemma4 module wraps the text model as .language_model (a text-only gemma4_text checkpoint has no
    # wrapper); the text model holds the decoder as .model, whose output is the final RMSNorm's.
    text_lm = getattr(model, "language_model", model)
    decoder = text_lm.model

    probe_prompt = get_model_explanations_formatted_as_binary_agreement_prompt(PROBE_SUMMARY, PROBE_EXPLANATION)
    if _render_prompt(tokenizer, probe_prompt).endswith(EMPTY_THOUGHT_BLOCK):
        answer_prefix = ""
    else:
        answer_prefix = EMPTY_THOUGHT_BLOCK
        logger.warning("Chat template does not emit %r in the generation prompt; appending it to every prompt.",
                       EMPTY_THOUGHT_BLOCK)
    probe_ids = _prompt_ids(tokenizer, probe_prompt, answer_prefix)
    head, softcap = _resolve_head(model, text_lm, decoder, probe_ids)

    prompt_text = _render_prompt(tokenizer, probe_prompt, answer_prefix)
    label_ids = {label: _resolve_label_token_ids(tokenizer, prompt_text, label) for label in LABELS}
    for label, ids in label_ids.items():
        if len(ids) > 1:
            logger.warning("%r is %d tokens in context (%s); its first token is used for classification.",
                           label, len(ids), hf.convert_ids_to_tokens(ids))
    yes_id, no_id = label_ids["Yes"][0], label_ids["No"][0]
    assert yes_id != no_id, "Yes and No share their first token in this tokenizer, so they cannot be compared."

    lm = types.SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        head=head,
        softcap=softcap,
        answer_prefix=answer_prefix,
        yes_id=yes_id,
        no_id=no_id,
        # Exact token cost of the chat template, answer prefix, and prompt boilerplate with both fields empty.
        prompt_overhead_tokens=len(_prompt_ids(
            tokenizer, get_model_explanations_formatted_as_binary_agreement_prompt("", ""), answer_prefix)),
    )

    # Load-time sanity check: on the probe, the first generated token should be the exact Yes token.
    hidden, logits = _forward(lm, probe_ids)
    top1 = int(mx.argmax(logits))
    tok = hf.convert_ids_to_tokens
    if top1 not in (yes_id, no_id):
        logger.warning("Probe prompt: top-1 next token is %r, not Yes/No. Run check_label_tokens() to inspect.",
                       tok(top1))
    else:
        logger.info("%s: Yes=%r No=%r, answer prefix=%r, probe answer=%r, hidden size=%d",
                    model_path, tok(yes_id), tok(no_id), answer_prefix, tok(top1), hidden.shape[-1])
    return lm


def _truncate(tokenizer, text: str, ids, budget: int) -> str:
    """Keep the head and tail of `text` within `budget` tokens (the verdict usually sits at the end)."""
    if len(ids) <= budget:
        return text
    if budget <= 0:
        return ""
    head = int(budget * TRUNCATION_HEAD_FRACTION)
    tail = budget - head
    hf = _hf(tokenizer)
    return hf.decode(ids[:head]) + TRUNCATION_MARKER + hf.decode(ids[len(ids) - tail:])


def _truncate_fields(tokenizer, summary: str, explanation: str, budget: int):
    """Split `budget` tokens between the two fields: the explanation has priority, but the summary is
    guaranteed min(MIN_SUMMARY_TOKENS, its own length) so the <topic> block is never emptied."""
    hf = _hf(tokenizer)
    summary_ids = hf.encode(summary, add_special_tokens=False)
    explanation_ids = hf.encode(explanation, add_special_tokens=False)
    if len(summary_ids) + len(explanation_ids) <= budget:
        return summary, explanation
    summary_budget = min(len(summary_ids), max(MIN_SUMMARY_TOKENS, budget - len(explanation_ids)), budget)
    explanation_budget = budget - summary_budget
    return (_truncate(tokenizer, summary, summary_ids, summary_budget),
            _truncate(tokenizer, explanation, explanation_ids, explanation_budget))


def get_agreement_model_embedding(lm, document_text: str, device: str = None, include_max_pool: bool = True,
                                  include_yes_no_logits: bool = False):
    """`device` is accepted for signature compatibility with the PyTorch module and ignored."""
    assert not include_yes_no_logits, "ERROR: The current convention is to not use the output logits."
    hidden, next_token_logits = _forward(lm, _prompt_ids(lm.tokenizer, document_text, lm.answer_prefix))
    hidden = hidden.astype(mx.float32)  # (seq_len, hidden_size)

    top1 = int(mx.argmax(next_token_logits))
    if top1 not in (lm.yes_id, lm.no_id):
        logger.warning("Top-1 next token is %r, not Yes/No.", _hf(lm.tokenizer).convert_ids_to_tokens(top1))

    # [max over tokens] :: mean over tokens :: final-token hidden state (input to the LM head that
    # determines No/Yes) :: [no logit :: yes logit]
    parts = []
    if include_max_pool:
        parts.append(mx.max(hidden, axis=0))
    parts.append(mx.mean(hidden, axis=0))
    parts.append(hidden[-1])
    if include_yes_no_logits:
        parts.append(next_token_logits[mx.array([lm.no_id, lm.yes_id])])
    embedding = mx.concatenate(parts).tolist()
    agreement_classification = bool((next_token_logits[lm.yes_id] > next_token_logits[lm.no_id]).item())
    return embedding, agreement_classification


def get_local_embedding_for_agreement_prompt(lm_model_summary: str, lm_model_explanation: str,
                                             model_path: str = DEFAULT_MODEL_PATH, device: str = None,
                                             max_token_length: int = 8192,
                                             include_max_pool: bool = True, include_yes_no_logits: bool = False):
    """Returns (embedding, agreement_classification), or (None, None) on failure.

    `device` is accepted for signature compatibility with the PyTorch module and ignored (MLX has one device).
    `max_token_length` bounds the full templated prompt; it is enforced at the token level by truncating the
    explanation (head + tail) and, if necessary, the summary.
    """
    return get_local_embedding_for_agreement_prompt_with_prompt(
        lm_model_summary, lm_model_explanation,
        model_path=model_path, device=device,
        max_token_length=max_token_length,
        include_max_pool=include_max_pool, include_yes_no_logits=include_yes_no_logits,
    )[:2]


def get_local_embedding_for_agreement_prompt_with_prompt(lm_model_summary: str, lm_model_explanation: str,
                                                         model_path: str = DEFAULT_MODEL_PATH, device: str = None,
                                                         max_token_length: int = 8192,
                                                         include_max_pool: bool = True,
                                                         include_yes_no_logits: bool = False):
    """Returns (embedding, agreement_classification, [possibly truncated] prompt), or (None, None, None) on failure.

    `device` is accepted for signature compatibility with the PyTorch module and ignored (MLX has one device).
    `max_token_length` bounds the full templated prompt; it is enforced at the token level by truncating the
    explanation (head + tail) and, if necessary, the summary.
    """
    try:
        lm = _load(model_path)
        budget = max_token_length - lm.prompt_overhead_tokens - TOKEN_MARGIN
        summary, explanation = _truncate_fields(lm.tokenizer, lm_model_summary, lm_model_explanation, budget)
        prompt = get_model_explanations_formatted_as_binary_agreement_prompt(summary, explanation)
        embedding, agreement_classification = get_agreement_model_embedding(lm, prompt, device, include_max_pool,
                                                                            include_yes_no_logits)
        return embedding, agreement_classification, prompt
    except Exception:
        logger.exception("Agreement model embedding failed.")
        return None, None, None


def check_label_tokens(document_text: str, model_path: str = DEFAULT_MODEL_PATH, device: str = None,
                       top_k: int = 5, max_new_tokens: int = 8) -> None:
    """Diagnostic: print how the prompt ends, the top-k next-token candidates at the last prompt position, and a
    short greedy continuation produced by mlx-lm's own generate loop. The candidates verify the exact Yes/No
    tokens resolved in _load; the continuation is an independent check that the checkpoint decodes coherently
    (a broken quantized load shows up here as repeated or nonsense tokens)."""
    lm = _load(model_path)
    hf = _hf(lm.tokenizer)
    ids = _prompt_ids(lm.tokenizer, document_text, lm.answer_prefix)
    print(f"prompt ends with: {hf.decode(ids[-8:])!r} "
          f"(answer prefix {'appended by _load' if lm.answer_prefix else 'not needed'})")
    _, logits = _forward(lm, ids)
    probs = mx.softmax(logits)
    tok = hf.convert_ids_to_tokens
    print(f"Yes -> id {lm.yes_id} {tok(lm.yes_id)!r} p={probs[lm.yes_id].item():.4f}; "
          f"No -> id {lm.no_id} {tok(lm.no_id)!r} p={probs[lm.no_id].item():.4f}")
    for i in mx.argsort(-probs)[:top_k].tolist():
        print(f"  p={probs[i].item():.4f}  id={i}  token={tok(i)!r}")
    continuation = mlx_generate(lm.model, lm.tokenizer, prompt=ids, max_tokens=max_new_tokens, verbose=False)
    print(f"greedy continuation: {continuation!r}")
