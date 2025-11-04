# Copyright Reexpress AI, Inc. All rights reserved.
"""
Data collator for the word ordering task
"""

import logging

import sdm_network_constants

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch


class CustomDataCollator:
    # assumes DocumentOrderingDataset
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Extract document IDs before padding
        document_ids = [f[sdm_network_constants.REEXPRESS_ID_KEY] for f in features]

        # Use tokenizer's built-in padding for input_ids and attention_mask
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"],
              "attention_mask": f["attention_mask"]} for f in features],
            padding=True,  # Pad to longest in batch
            return_tensors="pt"
        )

        # Also pad prompt_input_ids and prompt_attention_mask
        prompt_batch = self.tokenizer.pad(
            [{"input_ids": f["prompt_input_ids"],
              "attention_mask": f["prompt_attention_mask"]} for f in features],
            padding=True,
            return_tensors="pt"
        )

        # Handle labels separately
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            labels = f["labels"]
            padding_length = max_length - len(labels)
            padded = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=labels.dtype)
            ])
            padded_labels.append(padded)

        batch["labels"] = torch.stack(padded_labels)
        batch["document_id"] = document_ids  # Pass through as list
        batch["prompt_input_ids"] = prompt_batch["input_ids"]
        batch["prompt_attention_mask"] = prompt_batch["attention_mask"]
        batch[sdm_network_constants.DEFAULT_NEGATIVE_KEY] = \
            [f[sdm_network_constants.DEFAULT_NEGATIVE_KEY] for f in features]
        batch[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY] = \
            [f[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY] for f in features]
        batch[sdm_network_constants.CACHED_PROMPT_KEY] = \
            [f[sdm_network_constants.CACHED_PROMPT_KEY] for f in features]
        batch[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] = \
            [f[sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY] for f in features]
        return batch