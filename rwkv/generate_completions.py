# Generates completions from RWKV model based on a prompt.

import argparse
import os
import pathlib
import time

from torch.nn import init
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library
import numpy as np


# ======================================== Script settings ========================================


prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

# How many completions to generate.
generation_count: int = 1
# Token count per single completion.
tokens_per_generation: int = 300

## Variables for sampling methods, in order of how they will be applied:
# Typical Sampling:
tau: float = 0.2 # Used for typical sampling; 1.0 disables

top_p: float = 0.8 # Both 1.0 and 0.0 disable

top_k: int = 0 # If using contrastive search (CS), this cannot be too high or 0 (which disables it)

temperature: float = 1.1 # If not using CS and not using typical sampling, setting this to 0.0 will force greedy search

# Contrastive Search:
# Determines strength of degradation factor in sampling - used for a lerp between model's output and its similarity to earlier model states
alpha: float = 0.6  # between 0 and 1, where 0 disables CS and 1 uses only the similarities, leading to nonsense
# Temperature for the similarities array - similar to alpha, however is non-linear and is more likely to respect when the model is 90% confident in a token - this is probably better to change than alpha
beta: float = 1.2 # min value: 0, where higher values increase the range of values in the similarities array

argmax=False # If True, the most likely token is chosen after all the various sampling strategies. Otherwise, a random token is chosen based on the modified probabilities
# To achieve greedy sampling, set argmax=True, top_k=1, and disable typical sampling

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

do_contrastive = alpha != 0.0

init_logits, init_state = None, None
init_state_history = []

if not do_contrastive:
    for i in range(len(prompt_tokens) - 1):
        init_logits, init_state = model.eval(prompt_tokens[i], init_state, init_state, init_logits)
else:
    for i in range(len(prompt_tokens) - 1):
        init_logits, init_state = model.eval(prompt_tokens[i], init_state, init_state, init_logits)
        init_state_history.append(np.reshape(init_state.clone(), (model.layer_count * 5, model.embedding_size))[:2*5]) # Only keep first 2 layers - the same value as num_partial_layers in rwkv.cpp

for GENERATION in range(generation_count):
    print(f'\n--- Generation {GENERATION} ---\n')
    print(prompt, end='[')
    start = time.time()

    token = prompt_tokens[-1:][0] # last token
    logits, state = init_logits.clone(), init_state.clone()
    state_history = list(init_state_history)

    for i in range(tokens_per_generation):
        logits, state = model.eval(token, state, state, logits)

        if do_contrastive:
            state_history.append(np.reshape(state.clone(), (model.layer_count * 5, model.embedding_size))[:2*5]) # Same as above
        if len(state_history) > 124:
            state_history.pop(0) # TODO: investigate dequeue?
        token = sampling.sample_logits(logits, temperature, top_p, top_k, alpha, beta, state_history, model, tau, None, argmax=argmax)

        print(tokenizer.decode([token]), end='')


    delay = time.time() - start
    print(']\n\nTook %.3f sec, %d ms per token' % (delay, delay / tokens_per_generation * 1000))
