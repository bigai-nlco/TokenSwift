from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import HfArgumentParser


@dataclass
class Arguments:
    target: str
    ckpt_path: str

    gen_len: int
    gamma: int
    prefill_len: int

    model_type: str
    model_path_prefix: Optional[str] = field(default="/scratch2/nlp/plm")

    prompt_id: Optional[int] = field(default=-1)
    verbose: Optional[bool] = field(default=False)
    debug: Optional[bool] = field(default=False)
    compile: Optional[bool] = field(default=True)

    retrival_max_budget: Optional[int] = field(default=4096)

    num_heads: int = field(default=3)
    tree_decoding: Optional[bool] = field(default=False)
    ngram_topk: Optional[int] = field(default=0)


@dataclass
class SampleArgs:
    do_sample: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=1.0)

    penalty: Optional[float] = field(default=1.0)
    penalty_length: Optional[int] = field(default=0)

    top_k: Optional[int] = field(default=-1)  # 0.9 to 0.99
    top_p: Optional[float] = field(default=-1)

    min_p: Optional[float] = field(default=-1)  # 0.1 to 0.01
    epsilon: Optional[float] = field(default=-1)  # 3e-4 to 4e-3
    min_tokens_to_keep: Optional[int] = field(default=20)

    assistant_token_id: Optional[str] = field(default=None)


def get_args() -> Tuple[Arguments, SampleArgs]:

    parser = HfArgumentParser((Arguments, SampleArgs))
    args, sample_args = parser.parse_args_into_dataclasses()

    return args, sample_args
