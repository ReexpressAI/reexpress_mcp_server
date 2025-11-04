# Copyright Reexpress AI, Inc. All rights reserved.

"""
Minimal interactive chat script - just the essentials.
Usage: python simple_chat.py <model_path>
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# Get model path from command line
if len(sys.argv) < 3:
    print("Usage: python simple_chat.py <model_path> 0 (0 for greedy search, 1 for sample)")
    print("Example: python simple_chat.py microsoft/Phi-3.5-mini-instruct 0 (0 for greedy search, 1 for sample)")
    print("Example: python simple_chat.py ./models/phi35-finetuned 0 (0 for greedy search, 1 for sample)")
    sys.exit(1)

model_path = sys.argv[1]
do_sample = int(sys.argv[2]) == 1

temperature = 0.7
# Load model and tokenizer
print(f"Loading {model_path}...")
if do_sample:
    print(f"Sampling with {temperature}")
else:
    print(f"Greedy decoding")
tokenizer = AutoTokenizer.from_pretrained(model_path) #, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False  # Avoid compatibility issues
).eval()

print("Ready! Type 'quit' to exit.\n")

# Chat loop
while True:
    # Get user input
    user_input = input("You: ")

    if user_input.lower() in ['quit', 'exit']:
        break

    # Create prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_input}
    ]

    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate
    print("Bot: ", end="", flush=True)
    with torch.no_grad():
        if do_sample:
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            # Separate to avoid parameter warning:
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id
            )

    # Decode and print
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(response)
    print(f"Count of generated tokens: {inputs['input_ids'].shape[1]}")
    print()  # Blank line for readability
