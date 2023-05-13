import math
import numpy as np
import torch
from typing import Dict
from torch.nn import functional as F

import tokenizers
import os, pathlib

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8, top_k: int = 0, logit_bias: Dict[int, float] = None) -> int:
    probs = F.softmax(out.cpu(), dim=-1).numpy()
    
    return sample_probs(probs, temperature, top_p, top_k, logit_bias)


def sample_logits_contrastive_state(model, out: torch.Tensor, state_history, top_k: int = 0, alpha: float = 0.4, beta: float = 1.2, logit_bias: Dict[int, float] = None) -> int:
    probs = F.softmax(out.cpu(), dim=-1).numpy()

    return sample_probs_contrastive_state(model, probs, state_history, top_k, alpha, beta, logit_bias)


def sample_probs(probs: np.ndarray, temperature: float = 1.0, top_p: float = 0.8, top_k: int = 0, logit_bias: Dict[int, float] = None) -> int:
    assert 0.0 <= temperature, 'temperature'
    assert 0 <= top_k, 'top_k'
    assert 0.0 <= top_p <= 1.0, 'top_p'

    if top_p == 0.0:
        top_p = 1.0

    if logit_bias is not None:
        logits = np.log(probs)

        for token in logit_bias.keys():
            logits[token] += logit_bias[token]

        probs = np.exp(logits) / np.sum(np.exp(logits))

    if temperature == 0.0:
        return np.argmax(probs).item()

    if top_k != 0:
        indices = np.argpartition(probs, -top_k)[:-top_k]
        probs[indices] = 0

    if top_p < 1.0:
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    probs = probs / np.sum(probs)

    return np.random.choice(a=len(probs), p=probs)

def sample_probs_contrastive_state(model, probs: np.ndarray, state_history, top_k: int = 0, alpha = 0.0, beta: float = 1.2, logit_bias: Dict[int, float] = None) -> int:
    assert 0 < top_k <= 50, 'top_k' # Very unwise to run with no top_k or too high
    # assert 0 < top_p <= 1.0, 'top_p'
    assert 0.0 <= alpha <= 1.0, 'alpha'
    assert 1.0 <= beta, 'beta' # Below 1.0 would probably work to make the degradation array less varied, but that's generally not desirable

    if logit_bias is not None:
        # Copied from the other sampling method in rwkv.cpp
        logits = np.log(probs)

        for token in logit_bias.keys():
            logits[token] += logit_bias[token]

        probs = np.exp(logits) / np.sum(np.exp(logits))

    # Am mostly using top_p as a speedup: if the probability for a token was very high (say, "ists" after "Scien"),
    # then I suspect the degradation penalty would still allow this, even if it's repetitive, but there's no point checking other tokens
    #
    # I have disabled top_p as it often led to degradation penalty being ignored, since the model would get in a loop of producing really high probability outputs and so top_p would lead to only 1 being possible, degrading to greedy sampling
    # if top_p < 1.0:
    #     sorted_probs = np.sort(probs)[::-1]
    #     cumulative_probs = np.cumsum(sorted_probs)
    #     cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    #     probs[probs < cutoff] = 0
    # top_p_indices = np.nonzero(probs)[0]
    #
    # if len(top_p_indices) > top_k:
    #     indices = np.argpartition(probs, -top_k)[-top_k:]
    # else:
    #     indices = top_p_indices

    indices = np.argpartition(probs, -top_k)[-top_k:]


    # Notes:
    # So far, I've only really been testing sampling layer 0 because it would be the fastest if I can modify rwkv.cpp to allow limiting the number of layers computed
    # LAYER_TO_SAMPLE + 0 is last_x for channel mixing, LAYER_TO_SAMPLE + 1 is last_x for time mixing, and the other 3 are internal bits of time mixing
    # Experiements with single layer of state:
    # Generally, I only worked with one part of each layer's state at a time (eg channel mixing's last_x, aa or bb from time mixing, etc)
    # I think the 3 time mixing ones (state[layer+2] and 3 or 4) do actually reflect somewhat how repetitive it is, since using the Greedy: debug metric from below, they are higher when it's repeating itself (eg from 0.77 -> 0.82)
    # However, they change really slowly and with a very small range over the output, meaning they do a bad job of selecting which next token is repetitive and which is new, since they give basically the same output for each token possibility
    # I've previously gotten fairly good results with [layer+0] and [layer+1] (last_x for channel and time mixing), but I don't see as obvious a change (particularly in the later layers)
    # The only way I've reliably gotten a large range of different similarities over possible next tokens (which I assume means the function is more
    # able to distinguish repetitive and non-repetitive tokens) is to use layer+0 or +1 from early in the model, but I worry this is just because those variables 
    # are a weighted sum of the token itself with some state, so the variety just reflects the fact there are different tokens
    # Code for single layer single vector from the layer:
    LAYER_TO_SAMPLE = 0
    assert(LAYER_TO_SAMPLE < model.layer_count)
    MODEL_VEC = 0
    assert(MODEL_VEC < 5)
    # sim = lambda a, b: F.cosine_similarity(a[5*LAYER_TO_SAMPLE+MODEL_VEC], b[5*LAYER_TO_SAMPLE+MODEL_VEC], dim=0)
    # Use both the time mixing and channel mixing last_x vectors:
    sim = lambda a, b: torch.mean(F.cosine_similarity(a[5*LAYER_TO_SAMPLE:5*LAYER_TO_SAMPLE+2], b[5*LAYER_TO_SAMPLE:5*LAYER_TO_SAMPLE+2], dim=1))
    # Entire layer:
    # sim = lambda a, b: 1 + math.log(torch.mean(F.cosine_similarity(a[5*LAYER_TO_SAMPLE:5*LAYER_TO_SAMPLE+5], b[5*LAYER_TO_SAMPLE:5*LAYER_TO_SAMPLE+5], dim=1))) / math.log(2)
    # Results from testing with greedy: before first repetition, it gives 0.788 but before the second it gives 0.977, with a range of around 0.1 for the next token - quite good, but quite high and clustered together
    # I got similar results from other earlier layers, and later layers were worse in every aspect but not by too much
    # With a longer prompt, the similarity goes up to 0.999 for repeated results, but was already fairly high at the start (~0.95, with occasional dips to 0.75)
    # Maybe I could use log? (eg similarity = 1+math.log(prev_sim), to give 0.999 -> 0.999 but 0.8 -> 0.77 and 0.7 -> 0.64 by default, with potential for scaling by dividing the log term by log(2) or something else smaller than e)
    # Also limited testing shows that using just the first two vectors is slightly better in every way
    # I tried using log and it's great but could do with some adjustment, so I added a hyperparameter called beta:
    sim = lambda a, b: 1 + math.log(torch.mean(F.cosine_similarity(a[5*LAYER_TO_SAMPLE:5*LAYER_TO_SAMPLE+2], b[5*LAYER_TO_SAMPLE:5*LAYER_TO_SAMPLE+2], dim=1))) ** 1/beta

    # Experiments with entire state:
    # Trying to do cosine_similarity on the entire state vector generally doesn't give very good results, since the state as a whole doesn't change much
    # I tried some cursed thing where I computed the similarity between the *channels* of the prev states and the potential, and I couldn't get a good result - I wasn't sure what do do with the similarity vector
    # Here's that cursed channels thing:
    # split_state = lambda a: torch.stack((torch.cat([a[i*5:i*5+1] for i in range(model.layer_count)], dim=0), torch.cat([a[i*5+1:i*5+2] for i in range(model.layer_count)], dim=0)))
    # sim = lambda a, b: torch.amax(torch.mean(F.cosine_similarity(split_state(a), split_state(b), dim=1), dim=1))
    # Just a naive comparison between state vectors (not very good - always gives a small range of outputs and doesn't reflect how repetitive it is)
    # sim = lambda a, b: torch.mean(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


    # All of this could probably be optimised to use matrices, based on the code for the contrastive search paper
    similarities = np.zeros(len(probs))
    for i in range(len(indices)):
        potential_state = model.eval(indices[i], state_history[-1].flatten())[1].reshape(model.layer_count * 5, model.embedding_size)
        similarities[indices[i]] = np.amax([sim(potential_state, state_history[j]) for j in range(len(state_history))])

    weighted = np.zeros(len(probs)) - 1    # 0 > -1, and the lowest value of the below formula is -1.0
    for i in range(len(indices)):
        weighted[indices[i]] = (1 - alpha) * probs[indices[i]] - alpha * similarities[indices[i]]

    # tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
    # tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    # print([tokenizer.decode([i]) for i in indices])
    # print([probs[indices[i]] for i in range(len(indices))])
    # print([similarities[indices[i]] for i in range(len(indices))])
    # print([weighted[indices[i]] for i in range(len(indices))])

    # I'm mostly using this to evaluate the different similarity functions
    # It just tells me how much variation there is, which I interpret as how much information has been captured by the function
    # Plus, I can evaluate how useful similarity functions actually are by forcing greedy sampling (changing the return line to be in terms of probs instead) and
    # looking at the Greedy: value, since that corresponds to what greedy sampling will pick, meaning I can see how well the function punishes repetition and rewards other ideas
    # So far, this type of analysis has lead to me realising that most comparison functions I've tried aren't actually that good at all
    # sims = [similarities[indices[i]] for i in range(len(indices))]
    # print("     Max: {}, Min: {}, Avg: {}, Range: {}, Greedy: {}".format(round(max(sims), 3), round(min(sims), 3), round(sum(sims) / len(sims), 3), round(max(sims) - min(sims), 3), round(similarities[np.argmax(probs).item()], 3)))

    return np.argmax(weighted).item()

def sample_logits_typical(logits, temp=1.0, tau=0.2):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
        shifted_logits = torch.abs(logits - ent)
        sorted_ids = torch.argsort(shifted_logits)
        sorted_logits = shifted_logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < tau)
        probs[shifted_logits > sorted_logits[cutoff]] = 0
        if temp != 1.0:
            probs = probs ** (1.0 / temp)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
