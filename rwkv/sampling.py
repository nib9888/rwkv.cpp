import numpy as np
import torch
from typing import Dict, Optional, List
from torch.nn import functional as F

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8, top_k: int = 0, alpha: float = 0.0, beta: float = 1.2, state_history: List[torch.Tensor] = None, model = None, tau: float = 1.0, logit_bias: Optional[Dict[int, float]] = None, argmax=False) -> int:
    assert 0.0 <= temperature, 'temperature'
    assert 0 <= top_k, 'top_k'
    assert 0.0 <= top_p <= 1.0, 'top_p'
    assert 0.0 <= tau <= 1.0, 'tau'
    assert 0.0 <= alpha <= 1.0, 'alpha'

    do_typical = tau != 1.0
    do_contrastive = alpha != 0.0
    if do_contrastive:
        assert state_history is not None, 'state_history'
        assert 0.0 <= beta, 'beta'
        assert model is not None, 'model'
        assert top_k != 0, 'Contrastive search needs top_k'

    probs = F.softmax(out.cpu(), dim=-1).numpy()

    if temperature == 0.0 and alpha == 0.0 and tau == 1.0:
        return np.argmax(probs).item()

    if logit_bias is not None:
        probs = apply_logit_bias(probs, logit_bias)

    if do_typical:
        probs = apply_typical(torch.from_numpy(probs), tau).numpy()

    if top_p != 0.0 and top_p != 1.0:
        probs = apply_top_p(probs, top_p)

    if top_k != 0:
        indices = np.argpartition(probs, -top_k)[:-top_k]
        probs[indices] = 0

    indices = np.nonzero(probs)[0]
    if len(indices) == 1:
        return indices[0]

    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    if do_contrastive:
        probs = apply_contrastive(probs, model, state_history, alpha, beta)

    if not argmax:
        probs = probs / np.sum(probs)
        return np.random.choice(a=len(probs), p=probs)
    else:
        return np.argmax(probs).item()

def apply_logit_bias(probs: np.ndarray, logit_bias: Dict[int, float]):
    assert logit_bias is not None, 'logit_bias'

    logits = np.log(probs)

    for token in logit_bias.keys():
        logits[token] += logit_bias[token]

    return np.exp(logits) / np.sum(np.exp(logits))

def apply_typical(probs, tau):
    logits = -torch.log(probs)
    ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
    shifted_logits = torch.abs(logits - ent)
    sorted_ids = torch.argsort(shifted_logits)
    sorted_logits = shifted_logits[sorted_ids]
    sorted_probs = probs[sorted_ids]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
    cutoff = np.sum(cumulative_probs < tau)
    probs[shifted_logits > sorted_logits[cutoff]] = 0
    return probs

def apply_top_p(probs: np.ndarray, top_p: float):
    assert 0.0 < top_p < 1.0, 'top_p'

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    return probs

def apply_contrastive(probs: np.ndarray, model, state_history: list[torch.Tensor], alpha: float, beta: float):

    LAYER_TO_SAMPLE = 1
    assert(LAYER_TO_SAMPLE < 2)
    MODEL_VEC = 0 # Use channel mixing's last_x
    assert(MODEL_VEC < 5)

    indices = np.nonzero(probs)[0]

    similarities = np.zeros(len(probs))
    prev_states = np.concatenate(state_history, axis=0).reshape((len(state_history), -1, model.embedding_size))
    prev_vecs = torch.from_numpy(prev_states[:, LAYER_TO_SAMPLE*5+MODEL_VEC])
    for i in indices:
        potential_state = model.partial_eval(i, torch.from_numpy(np.resize(state_history[-1], (model.layer_count * 5, model.embedding_size)).flatten())).reshape(model.layer_count * 5, model.embedding_size)
        potential_vec = potential_state[LAYER_TO_SAMPLE*5+MODEL_VEC]
        similarities[i] = torch.amax(((F.cosine_similarity(potential_vec, prev_vecs) + 1) / 2) ** beta) # the ( + 1) / 2 is required to keep everything positive

    weighted = np.zeros(len(probs))
    for i in indices:
        weighted[i] = (1 - alpha) * probs[i] - alpha * similarities[i] + 1

    # Debug/analysis stuff
    # import tokenizers
    # import os, pathlib
    # tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
    # tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    # print("\n" + str([tokenizer.decode([i]) for i in indices]))

    # print([probs[indices[i]] for i in range(len(indices))])
    # print([similarities[indices[i]] for i in range(len(indices))])
    # print([weighted[indices[i]] for i in range(len(indices))])

    # sims = [similarities[indices[i]] for i in range(len(indices))]
    # print("\nMax: {:.3f}, Min: {:.3f}, Avg: {:.3f}, Range: {:.3f}, Greedy: {:.3f}".format(round(max(sims), 3), round(min(sims), 3), round(sum(sims) / len(sims), 3), round(max(sims) - min(sims), 3), round(similarities[np.argmax(probs).item()], 3)), end="")
    # print("\nAvg: {:.3f}, Range: {:.3f}, Greedy: {:.3f}".format(round(sum(sims) / len(sims), 3), round(max(sims) - min(sims), 3), round(similarities[np.argmax(probs).item()], 3)), end="")
    # print("\nAvg: {:.3f}, Range: {:.3f}, Chosen: {:.3f}".format(round(sum(sims) / len(sims), 3), round(max(sims) - min(sims), 3), round(similarities[np.argmax(weighted).item()], 3)), end="")

    return weighted

