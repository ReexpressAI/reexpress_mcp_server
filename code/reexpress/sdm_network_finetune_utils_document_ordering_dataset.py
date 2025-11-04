# Copyright Reexpress AI, Inc. All rights reserved.
"""
Dataset subclass for the word ordering task
"""

import logging

from torch.utils.data import Dataset, DataLoader

import sdm_network_constants

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
import json


class DocumentOrderingDataset(Dataset):
    """Dataset for document ordering task with selective masking."""

    def __init__(
            self,
            data_file: str,
            tokenizer,
            max_length: int = 2048,
            mask_prefix: bool = True,
            mask_until_pattern: str = "No</verified>"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prefix = mask_prefix
        self.mask_until_pattern = mask_until_pattern

        # Load data
        self.examples = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        document_id = example.get(sdm_network_constants.REEXPRESS_ID_KEY, f"doc_{idx}")  # Fallback to idx if no ID

        # Extract the document (multiset) and the assistant response
        # document = example.get(sdm_network_constants.MULTISET_KEY, "")
        assistant_response = example.get(sdm_network_constants.REEXPRESS_GENAI_DOCUMENT_KEY, "")
        #
        # conv = sdm_network_constants.get_ordering_prompt_v1(document)
        prompt_text = example.get(sdm_network_constants.CACHED_PROMPT_KEY, "")
        conv = sdm_network_constants.get_conv(prompt_text)

        # Create prompt-only encoding for generation
        prompt_encoding = self.tokenizer.apply_chat_template(
            conv,  # Without assistant response
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_dict=True,
            padding=False,
            add_generation_prompt=True  # Add the generation prompt
        )

        conv.append({"role": "assistant", "content": assistant_response})

        # Tokenize the conversation
        encoding = self.tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            max_length=self.max_length,
            padding=False, #"max_length",
            truncation=True,
            return_dict=True,
            add_generation_prompt=False  # We have the assistant response
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels with selective masking
        # labels = self.create_masked_labels(input_ids.tolist(), assistant_response) #, attention_mask.tolist())
        labels = \
            sdm_network_constants.create_masked_labels(self.mask_prefix, self.tokenizer, self.mask_until_pattern,
                                                       input_ids.tolist(), assistant_response)
        labels = torch.tensor(labels, dtype=torch.long)

        # If this changes, CustomDataCollator must also be updated.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            sdm_network_constants.REEXPRESS_ID_KEY: document_id,
            "prompt_input_ids": prompt_encoding["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_encoding["attention_mask"].squeeze(0),
            sdm_network_constants.DEFAULT_NEGATIVE_KEY: example.get(sdm_network_constants.DEFAULT_NEGATIVE_KEY, ""),
            sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY:
                example.get(sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY, ""),
            sdm_network_constants.CACHED_PROMPT_KEY: prompt_text,
            sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY:
                example.get(sdm_network_constants.IS_SDM_TRAINING_SPLIT_KEY, -1),
        }
