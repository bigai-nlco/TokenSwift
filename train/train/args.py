from dataclasses import dataclass, field
from typing import Optional, Tuple

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llama_type: Optional[str] = field(default=None)

    medusa_heads: int = field(default=3)


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    max_length: int = field(default=8 * 1024)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)


def get_args() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args
