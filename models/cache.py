import torch

from typing import Tuple
from einops import rearrange, einsum
from transformers import PreTrainedModel


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


class FlashSimpleCache(Cache):
    def __init__(self, model: PreTrainedModel, target_max_budget: int) -> None:
        self.seq_len = 0
        self.target_max_budget = target_max_budget

        self.hidden_size = model.config.hidden_size
        self.num_key_value_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros(
            [self.layers, 1, self.num_key_value_heads, self.target_max_budget, self.head_dim], dtype=dtype
        ).to(model.device)
        self.value_cache = torch.zeros(
            [self.layers, 1, self.num_key_value_heads, self.target_max_budget, self.head_dim], dtype=dtype
        ).to(model.device)

    def print_status(self):
        print("[Full Cache] Cached:", self.seq_len, "| Budget:", self.target_max_budget)

    def reset(self):
        self.seq_len = 0
        self.key_cache.zero_()
        self.value_cache.zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.seq_len + key_states.shape[-2] > self.target_max_budget:
            raise IndexError(
                f"seq_len is {self.seq_len}, incoming is {key_states.shape[-2]}, target_max_budget is {self.target_max_budget}"
            )

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, : self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, : self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers - 1:
            self.seq_len += key_states.shape[-2]

        return key, value

    def get_full_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.seq_len + key_states.shape[-2] > self.target_max_budget:
            raise IndexError(
                f"seq_len is {self.seq_len}, incoming is {key_states.shape[-2]}, target_max_budget is {self.target_max_budget}"
            )
        key = torch.cat([self.key_cache[layer_idx][:, :, : self.seq_len], key_states], dim=2)
        value = torch.cat([self.value_cache[layer_idx][:, :, : self.seq_len], value_states], dim=2)
        return key, value


class RetrievalCache(Cache):
    def __init__(
        self,
        model: PreTrainedModel,
        retri_max_budget: int,
        prefill_len: int,
        gamma: int,
    ) -> None:

        self.origin_prefill_len = prefill_len
        self.prefill_len = prefill_len
        self.gamma = gamma

        self.retri_max_budget = retri_max_budget
        self.real_budget = retri_max_budget + gamma + 1

        self.hidden_size = model.config.hidden_size
        self.num_key_value_heads = model.config.num_key_value_heads
        self.num_key_value_groups = model.config.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros(
            [self.layers, 1, self.num_key_value_heads, self.real_budget, self.head_dim], dtype=dtype
        ).to(model.device)
        self.value_cache = torch.zeros(
            [self.layers, 1, self.num_key_value_heads, self.real_budget, self.head_dim], dtype=dtype
        ).to(model.device)

        self.update_prefill = True
        self.start_size = 16

    def print_status(self):
        print("[Retrieval Cache] Budget:", self.retri_max_budget, " | PreFill:", self.prefill_len)

    def init_retri_cache(self, kv_cache: FlashSimpleCache, query_states: torch.Tensor, layer_idx: int) -> None:

        query_states = query_states[:, :, -1:]

        # (bsz, num_kv_heads, kv_len, head_dim)
        k_ = kv_cache.key_cache[layer_idx, :, :, : self.prefill_len]
        v_ = kv_cache.value_cache[layer_idx, :, :, : self.prefill_len]

        if k_.shape[1] != query_states.shape[1]:
            query_states = rearrange(query_states, "b (n g) qs d -> b n g qs d", g=self.num_key_value_groups)
            # (bsz, num_kv_heads, num_kv_groups q_len, kv_len) --> (bsz, num_kv_heads, kv_len)
            attn_score = einsum(query_states, k_, "b n g qs d, b n kvs d -> b n g qs kvs").sum(2).squeeze(2)
        else:
            attn_score = einsum(query_states, k_, "b n qs d, b n kvs d -> b n qs kvs").squeeze(-2)

        _, topk_idx = torch.topk(attn_score[:, :, self.start_size :], k=self.retri_max_budget - self.start_size, dim=-1)
        topk_idx += self.start_size
        # [bsz, num_kv_heads, select_len, head_dim]
        topk_idx = topk_idx[:, :, :, None].repeat(1, 1, 1, self.head_dim)

        select_key = torch.gather(k_, dim=2, index=topk_idx)
        select_value = torch.gather(v_, dim=2, index=topk_idx)

        self.key_cache[layer_idx][:, :, : self.retri_max_budget] = torch.cat(
            [k_[:, :, : self.start_size], select_key], dim=2
        )
        self.value_cache[layer_idx][:, :, : self.retri_max_budget] = torch.cat(
            [v_[:, :, : self.start_size], select_value], dim=2
        )

    def update_retri_cache(self, kv_cache: FlashSimpleCache) -> None:

        start = self.retri_max_budget - (kv_cache.seq_len - self.prefill_len)
        if start < self.start_size:
            # make generate length infinite
            start = self.start_size
            self.update_prefill = True
            self.prefill_len = kv_cache.seq_len

        kv_start = kv_cache.seq_len - (self.retri_max_budget - start)
        self.key_cache[:, :, :, start : self.retri_max_budget] = kv_cache.key_cache[
            :, :, :, kv_start : kv_cache.seq_len
        ].clone()
        self.value_cache[:, :, :, start : self.retri_max_budget] = kv_cache.value_cache[
            :, :, :, kv_start : kv_cache.seq_len
        ].clone()

    def spec_update(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        start = self.retri_max_budget
        end = self.retri_max_budget + new_k_cache.shape[-2]

        self.key_cache[layer_idx][:, :, start:end] = new_k_cache.clone()
        self.value_cache[layer_idx][:, :, start:end] = new_v_cache.clone()

        return self.key_cache[layer_idx][:, :, :end], self.value_cache[layer_idx][:, :, :end]

    def reset(self):
        self.update_prefill = True
        self.prefill_len = self.origin_prefill_len
        self.key_cache.zero_()
        self.value_cache.zero_()
