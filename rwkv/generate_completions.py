# Generates completions from RWKV model based on a prompt.

import argparse
import os
import pathlib
import time
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library
import numpy as np


# ======================================== Script settings ========================================

# prompt: str = """# rwkv.cpp
#
# This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).
#
# Besides usual **FP32**, it supports **FP16** and **quantized INT4** inference on CPU. This project is **CPU only**."""

#prompt: str = "This is a simple demonstration of linguistic ambiguity. In the sentence \"Bob collapsed on the sidewalk. Soon he saw Carl coming to help. He was very ill.\", the person who was very ill was"

# with open("rwkv/prompt.txt", "r") as f:
#     prompt = f.read()
#

prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

# How many completions to generate.
generation_count: int = 1
# Token count per single completion.
tokens_per_generation: int = 300

# How much prompt to print out
prompt_len = 0

# Sampling settings.
temperature: float = 1.0
top_p: float = 0.6

# Contrastive search:
# Determines strength of degradation factor in sampling - used for a lerp between model's output and its similarity to earlier model states
alpha: float = 0.4  # between 0 and 1, where 0 is greedy search and thus disables CS, leaving nucleus sampling with top_p and temp defined above - high values lead to nonsense
# Temperature for the similarities array - similar to alpha, however is non-linear and is more likely to respect when the model is 90% confident in a token - this is probably better to change than alpha
beta: float = 1.2 # min value: 1.0, where higher values decrease chance of model repeating itself
top_k: int = 5 # Number of samples to contrast with context and then pick from - higher values directly influence processing time and could potentially decrease quality

typical_sampling = False
temperature = temperature
# tau = 0.2

# =================================================================================================

parser = argparse.ArgumentParser(description='Generate completions from RWKV model based on a prompt')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
args = parser.parse_args()

assert prompt != '', 'Prompt must not be empty'

print('Loading 20B tokenizer')
tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, args.model_path)
print(f'n_layer: {model.layer_count}')

prompt_tokens = tokenizer.encode(prompt).ids
prompt_token_count = len(prompt_tokens)
print(f'{prompt_token_count} tokens in prompt')

init_logits, init_state = None, None
init_state_history = []

if alpha == 0.0:
    for i in range(len(prompt_tokens) - 1):
        init_logits, init_state = model.eval(prompt_tokens[i], init_state, init_state, init_logits)
else:
    for i in range(len(prompt_tokens) - 1):
        init_logits, init_state = model.eval(prompt_tokens[i], init_state, init_state, init_logits)
        init_state_history.append(np.reshape(init_state.clone(), (model.layer_count * 5, model.embedding_size))) # TODO: optimise so that only the required layers are retained

for GENERATION in range(generation_count):
    print(f'\n--- Generation {GENERATION} ---\n')
    print(prompt[prompt_len * -1:], end='[')
    start = time.time()

    token = prompt_tokens[-1:][0] # last token
    logits, state = init_logits.clone(), init_state.clone()
    state_history = list(init_state_history)

    for i in range(tokens_per_generation):
        logits, state = model.eval(token, state, state, logits)

        if alpha == 0.0:
            if typical_sampling:
                token = sampling.sample_logits_typical(logits, temperature, tau)
            else:
                token = sampling.sample_logits(logits, temperature, top_p, top_k)
        else:
            state_history.append(np.reshape(state.clone(), (model.layer_count * 5, model.embedding_size)))
            token = sampling.sample_logits_contrastive_state(model, logits, state_history, top_k, alpha, beta)

        print(tokenizer.decode([token]), end='')


    delay = time.time() - start
    print(']\n\nTook %.3f sec, %d ms per token' % (delay, delay / tokens_per_generation * 1000))
