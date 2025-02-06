import torch
import random

from typing import List, Dict
from utils.sampling import sample

TOPK = 10


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.

    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".

    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[: c + 1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1 : start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i - 1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[: c + 1]))
                retrieve_paths.append(cur_medusa_choice[: c + 1])
        retrieve_indices_nest.append(retrieve_indice)

    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1
    )

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers


def generate_candidates(combined_prob, tree_indices, retrieve_indices, root_token):
    """
    Generate candidates based on provided logits and indices.

    Parameters:
    - combined_prob (torch.Tensor): Spec infer probs [gamma, vocab_size]
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.
    - root_token (torch.Tensor): Last resample token [1, 1]

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """
    # Extract the TOPK candidates from combined robs.
    candidates = torch.topk(combined_prob, TOPK, dim=-1).indices
    # [gamma, topk] => [gamma * topk]
    candidates = candidates.view(-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    candidates = torch.cat([root_token.squeeze(0), candidates], dim=0)
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    # To substitute -1, the -1 is the last position
    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=tree_candidates.dtype, device=tree_candidates.device)], dim=0
    )

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)

    return cart_candidates, tree_candidates


def update_buffer_ngram(
    medusa_buffers: Dict[str, torch.Tensor],
    tree_candidates: torch.Tensor,
    candidates: torch.Tensor,
    N: int,
    retrieved_ngrams: List,
):
    device = medusa_buffers["medusa_position_ids"].device
    ngram_num = len(retrieved_ngrams)

    retrieved_ngrams = torch.tensor(retrieved_ngrams, dtype=candidates.dtype, device=device)
    new_medusa_position_ids = torch.cat(
        [
            medusa_buffers["medusa_position_ids"],
            torch.tensor(
                [i for _ in range(ngram_num) for i in range(1, N)],
                dtype=medusa_buffers["medusa_position_ids"].dtype,
                device=device,
            ),
        ],
        dim=-1,
    )

    bsz, _, candidate_len, _ = medusa_buffers["medusa_attn_mask"].size()
    new_medusa_attn_mask = torch.zeros(
        (bsz, 1, candidate_len + (N - 1) * ngram_num, candidate_len + (N - 1) * ngram_num),
        dtype=medusa_buffers["medusa_attn_mask"].dtype,
        device=device,
    )
    new_medusa_attn_mask[:, :, :, 0] = 1
    new_medusa_attn_mask[:, :, :candidate_len, :candidate_len] = medusa_buffers["medusa_attn_mask"]
    small_causal_mask = torch.tril(torch.ones((N - 1, N - 1), device=device, dtype=new_medusa_attn_mask.dtype))
    for i in range(ngram_num):
        start_idx = candidate_len + i * (N - 1)
        end_idx = candidate_len + (i + 1) * (N - 1)
        new_medusa_attn_mask[:, :, start_idx:end_idx, start_idx:end_idx] = small_causal_mask

    new_retrieve_indices = torch.zeros((ngram_num, N), dtype=medusa_buffers["retrieve_indices"].dtype, device=device)
    start_position = tree_candidates.shape[1]
    new_retrieve_indices[:, 1:] = torch.tensor(
        [i for i in range(start_position, start_position + (N - 1) * ngram_num)], dtype=torch.long, device=device
    ).reshape([ngram_num, N - 1])
    new_retrieve_indices = torch.cat([medusa_buffers["retrieve_indices"], new_retrieve_indices], dim=0)

    new_tree_candidates = torch.cat([tree_candidates, retrieved_ngrams[:, 1:].reshape(-1).unsqueeze(0)], dim=-1)

    new_candidates = torch.cat([candidates, retrieved_ngrams], dim=0)

    new_medusa_buffers = {
        "medusa_position_ids": new_medusa_position_ids,
        "medusa_attn_mask": new_medusa_attn_mask,
        "retrieve_indices": new_retrieve_indices,
    }

    return new_medusa_buffers, new_tree_candidates, new_candidates


def decode_tree_candidates(
    graph_engine,
    tree_candidates,
    medusa_position_ids,
    cur_kv_len,
    retrieve_indices,
    medusa_attn_mask,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    originally tree_decoding from Medusa, changed name to avoid conflict

    Parameters:
    - graph_engine
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - cur_kv_len
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    - medusa_attn_mask

    Returns:
    - torch.Tensor: Returns logits.
    """

    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    position_ids = medusa_position_ids + cur_kv_len

    # Use the model to decode the tree candidates.
    tree_logits = graph_engine.engine.model(
        input_ids=tree_candidates,
        position_ids=position_ids.unsqueeze(0),
        kv_cache=graph_engine.engine.kv_cache,
        retri_cache=graph_engine.engine.retri_cache,
        medusa_mask=medusa_attn_mask,
    )

    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    return logits


def exactly_match_sampling(candidates: torch.Tensor, full_verify_probs: torch.Tensor, retrieved_ngram_num: int = 0):
    assert len(candidates) == len(full_verify_probs)

    # the first token in candidates is accepted by default
    # [leaf_nodes, gamma + 1] => [leaf_nodes, gamma]
    medusa_candidates = candidates[:, 1:]  # draft tokens

    # [leaf_nodes, gamma + 1, vocab_size]
    _, gamma_plus_1, vocab_size = full_verify_probs.shape

    # extract candidate tokens' probs
    # [leaf_nodes, gamma + 1, vocab_size] => [leaf_nodes, gamma + 1]
    full_candidates = sample(full_verify_probs.reshape(-1, vocab_size)).reshape(-1, gamma_plus_1)
    verify_candidates = full_candidates[:, :-1]

    # find max accept length
    # mask = 1: selected [leaf_nodes, gamma]
    mask = (medusa_candidates == verify_candidates).to(torch.int8)
    # [leaf_nodes]
    candidates_accept_length = (torch.cumprod(mask, dim=1)).sum(dim=1)
    accept_len = candidates_accept_length.max()

    if (retrieved_ngram_num > 0) and (candidates_accept_length[-retrieved_ngram_num:].max() == accept_len):
        max_acc_ngram_len = accept_len
    else:
        max_acc_ngram_len = 0

    max_accept_length_list = torch.nonzero(candidates_accept_length == accept_len).squeeze(1)
    accept_index = max_accept_length_list[random.sample((0, max_accept_length_list.shape[0] - 1), k=1)].item()

    accept_seq = candidates[accept_index, : accept_len + 1]  # plus 1 for root token

    resample_token = full_candidates[accept_index, accept_len].unsqueeze(0)

    return {
        "accept_tokens": accept_seq,
        "resample_token": resample_token,
        "index": accept_index,
        "all_accept": (accept_len == verify_candidates.shape[1]),
        "max_acc_ngram_len": max_acc_ngram_len,
    }
