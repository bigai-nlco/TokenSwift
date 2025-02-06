import torch
from torch.nn import functional as F

from typing import List

from arguments import SampleArgs


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(logits: torch.Tensor, sample_args: SampleArgs, input_ids: List) -> torch.Tensor:

    if sample_args.assistant_token_id is not None:
        assistant_token_id_list = [int(x) for x in sample_args.assistant_token_id.split(",")]
        for token_id in assistant_token_id_list:
            logits[..., token_id] = -torch.inf

    if sample_args.do_sample:

        logits = logits / sample_args.temperature

        if sample_args.penalty > 1.0:

            # count = Counter(input_ids[-2048:])
            # input_ids = [id_ for id_, freq in count.items() if freq >= sample_args.min_freq]
            input_ids = torch.tensor(input_ids[-sample_args.penalty_length :], device=logits.device).unsqueeze(0)

            if len(logits.shape) == 3:
                input_ids = input_ids.unsqueeze(0)

            input_logit = torch.gather(logits, -1, input_ids)
            # if input_logit < 0 then repetition penalty has to be multiplied to reduce the token probabilities
            input_logit = torch.where(
                input_logit < 0, input_logit * sample_args.penalty, input_logit / sample_args.penalty
            )
            logits = logits.scatter(-1, input_ids, input_logit)

        if sample_args.min_p > 0:
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            # Get the probability of the top token for each sequence in the batch
            top_probs, _ = probs.max(dim=-1, keepdim=True)
            # Calculate the actual min_p threshold by scaling min_p with the top token's probability
            scaled_min_p = sample_args.min_p * top_probs
            # Create a mask for tokens that have a probability less than the scaled min_p
            tokens_to_remove = probs < scaled_min_p

            sorted_indices = torch.argsort(logits, descending=True, dim=-1)
            sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : sample_args.min_tokens_to_keep] = False

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if 0 < sample_args.epsilon < 1:
            epsilon = torch.tensor(sample_args.epsilon, device=logits.device)
            probs = torch.softmax(logits, dim=-1)
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            eta = torch.min(epsilon, torch.sqrt(epsilon) * torch.exp(-entropy))[..., None]
            indices_to_remove = probs < eta

            # Keep the words with the 'min_tokens_to_keep'-highest probabilities
            top_k = min(sample_args.min_tokens_to_keep, logits.size(-1))  # Safety check
            indices_to_remove = indices_to_remove & (logits < torch.topk(logits, top_k)[0][..., -1, None])

            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if (sample_args.top_p > 0) or (sample_args.top_k > 0):
            logits = top_k_top_p_filter(logits, top_k=sample_args.top_k, top_p=sample_args.top_p)

    probs = F.softmax(logits, dim=-1)

    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    try:
        idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    except Exception as e:
        print(e)
        print(torch.isnan(probs).all())
        idx_next = probs[torch.isnan(probs)] = 0
        idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)

    return idx_next


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum
