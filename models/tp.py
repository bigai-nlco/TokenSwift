import os
import torch
import torch.distributed as dist

from torch import nn
from itertools import accumulate

from models.modeling_qwen2_5 import Qwen2Attention2, Qwen2MLP, Qwen2ForCausalLM


def _get_global_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def init_dist():
    global_rank = _get_global_rank()
    world_size = _get_world_size()
    torch.cuda.set_device(global_rank)
    dist.init_process_group(
        backend="nccl", rank=global_rank, world_size=world_size, device_id=torch.device(f"cuda:{global_rank}")
    )
    global_group = dist.group.WORLD
    return global_rank, global_group


def _select_kv_heads(num_kv_heads, rank_group: list):
    global_rank = _get_global_rank()
    rank = rank_group.index(global_rank)
    world_size = len(rank_group)
    base_heads = num_kv_heads // world_size
    remainder = num_kv_heads % world_size
    distribution = [base_heads] * world_size
    for i in range(remainder):
        distribution[i] += 1
    cumulative_distribution = list(accumulate(distribution))
    if rank == 0:
        start = 0
        end = cumulative_distribution[0]
    else:
        start = cumulative_distribution[rank - 1]
        end = cumulative_distribution[rank]
    return start, end


def _apply_tp_linear(
    linear: nn.Linear,
    style: str,
    rank_group=None,
    num_kv_heads=None,
    num_heads=None,
    head_dim=None,
    is_kv=False,
    bias=True,
) -> None:

    num_group = num_heads // num_kv_heads
    kv_start, kv_end = _select_kv_heads(num_kv_heads, rank_group)
    q_start = kv_start * num_group * head_dim
    q_end = kv_end * num_group * head_dim
    kv_start = kv_start * head_dim
    kv_end = kv_end * head_dim

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    def shard(x, dim, start, end):
        if dim == 0:
            return x[start:end]
        elif dim == 1:
            return x[:, start:end]

    # ensure we can shard evenly
    def shard_qkv(x, dim, is_kv):
        if is_kv:
            return shard(x, dim, kv_start, kv_end)
        else:
            return shard(x, dim, q_start, q_end)

    sharded_weight = shard_qkv(linear.weight, shard_dim, is_kv)
    if hasattr(linear, "scales") and style == "colwise":
        linear.scales = shard_qkv(linear.scales, 0, is_kv)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    if bias:
        sharded_bias = shard_qkv(linear.bias, shard_dim, is_kv)
        linear.bias = nn.Parameter(sharded_bias, requires_grad=False)
    setattr(linear, size_attr, linear.weight.shape[shard_dim])


def _apply_tp_linear_mlp(linear: nn.Linear, style: str, rank_group=None) -> None:
    global_rank = _get_global_rank()
    rank = rank_group.index(global_rank)
    world_size = len(rank_group)

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    def shard(x, dim):
        return torch.chunk(x, world_size, dim=dim)[rank]

    # shard
    sharded_weight = shard(linear.weight, shard_dim)
    if hasattr(linear, "scales") and style == "colwise":
        linear.scales = shard(linear.scales, 0)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, linear.weight.shape[shard_dim])


def _apply_tp_ffn(mlp: Qwen2MLP, rank_group, group) -> None:
    assert hasattr(mlp, "gate_proj")
    assert hasattr(mlp, "up_proj")
    assert hasattr(mlp, "down_proj")

    _apply_tp_linear_mlp(mlp.gate_proj, "colwise", rank_group=rank_group)
    _apply_tp_linear_mlp(mlp.up_proj, "colwise", rank_group=rank_group)
    _apply_tp_linear_mlp(mlp.down_proj, "rowwise", rank_group=rank_group)
    mlp.process_group = group


def _apply_tp_attn(attn: Qwen2Attention2, rank_group, config, group) -> None:
    assert hasattr(attn, "q_proj")
    assert hasattr(attn, "k_proj")
    assert hasattr(attn, "v_proj")
    assert hasattr(attn, "o_proj")

    _apply_tp_linear(
        attn.q_proj,
        "colwise",
        rank_group=rank_group,
        num_kv_heads=attn.num_key_value_heads,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
    )
    _apply_tp_linear(
        attn.k_proj,
        "colwise",
        rank_group=rank_group,
        num_kv_heads=attn.num_key_value_heads,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
        is_kv=True,
    )
    _apply_tp_linear(
        attn.v_proj,
        "colwise",
        rank_group=rank_group,
        num_kv_heads=attn.num_key_value_heads,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
        is_kv=True,
    )
    _apply_tp_linear(
        attn.o_proj,
        "rowwise",
        rank_group=rank_group,
        num_kv_heads=attn.num_key_value_heads,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
        bias=False,
    )

    # overwrite
    attn.num_heads = config.num_attention_heads
    attn.hidden_size = config.hidden_size
    attn.head_dim = attn.hidden_size // attn.num_heads
    attn.num_key_value_heads = config.num_key_value_heads
    attn.process_group = group


def _apply_tp_Transformer(Transformer: Qwen2ForCausalLM, rank_group, process_group) -> None:
    # overwrite config before Transformer.setup_cache is called
    num_heads = Transformer.config.num_attention_heads
    num_kv_heads = Transformer.config.num_key_value_heads
    num_group = num_heads // num_kv_heads

    start, end = _select_kv_heads(num_kv_heads, rank_group)
    local_num_kv_heads = end - start
    local_num_heads = local_num_kv_heads * num_group
    local_dim = Transformer.config.hidden_size * local_num_kv_heads // num_kv_heads

    Transformer.config.num_attention_heads = local_num_heads
    Transformer.config.hidden_size = local_dim
    Transformer.config.num_key_value_heads = local_num_kv_heads

    _apply_tp_linear_mlp(Transformer.model.embed_tokens, "rowwise", rank_group=rank_group)
    _apply_tp_linear_mlp(Transformer.lm_head, "colwise", rank_group=rank_group)
    for block in Transformer.model.medusa_layers:
        block.process_group = process_group
        _apply_tp_linear_mlp(block.linear, "colwise", rank_group=rank_group)

    Transformer.process_group = process_group
    Transformer.model.process_group = process_group


def apply_tp(model: Qwen2ForCausalLM, rank_group, group) -> None:
    _apply_tp_Transformer(model, rank_group, group)
    for block in model.model.layers:
        _apply_tp_ffn(block.mlp, rank_group, group)
        _apply_tp_attn(block.self_attn, rank_group, model.config, group)
