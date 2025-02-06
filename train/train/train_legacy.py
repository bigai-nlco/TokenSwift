from dataclasses import dataclass
import datasets
import pathlib
from typing import Dict, Sequence

import torch
import transformers
from transformers import Trainer
from torch.nn import CrossEntropyLoss

import sys

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from train.args import get_args
from model.llama_config import LlamaConfig
from model.qwen2_5_config import Qwen2Config


class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs: torch.Tensor, num_items_in_batch: int):

        if hasattr(model, "module"):
            base_model = model.module.model
            lm_head = model.module.lm_head
        else:
            base_model = model.model
            lm_head = model.lm_head

        hidden_states_list = base_model(input_ids=inputs["input_ids"])

        m_logits = []
        for item in hidden_states_list:
            m_logits.append(lm_head(item))

        ce_loss_fct = CrossEntropyLoss()

        loss = 0
        log = {}

        for i in range(len(m_logits)):

            # CE
            medusa_logits = m_logits[i][:, : -(2 + i)].contiguous()
            actual_labels = inputs["labels"][:, 2 + i :].contiguous()

            medusa_logits = medusa_logits.view(-1, medusa_logits.shape[-1])
            actual_labels = actual_labels.view(-1)

            ce_loss_i = ce_loss_fct(medusa_logits, actual_labels)

            log[f"ce_loss_{i + 1}"] = ce_loss_i.item()

            # Add top-k accuracy
            for k in range(1, 4):
                _, top_k = medusa_logits.topk(k, dim=-1)
                correct = top_k.eq(actual_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i + 1}_top{k}"] = correct.float().mean().item()

            loss += ce_loss_i

        if self.state.global_step % self.state.logging_steps == 0:
            self.log(log)

        return loss


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        # each data larger than 8k length
        input_ids = [torch.tensor(x["input_ids"][: self.max_length]) for x in instances]

        input_ids = torch.vstack(input_ids)

        return dict(input_ids=input_ids, labels=input_ids)


def train():
    global local_rank

    transformers.set_seed(42)

    model_args, data_args, training_args = get_args()
    local_rank = training_args.local_rank

    if model_args.llama_type == "qwen2_5":
        config = Qwen2Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        config = LlamaConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

    config.use_cache = False
    config.medusa_heads = model_args.medusa_heads
    rank0_print(config)

    # Load model
    if model_args.llama_type == "llama2":
        from model.modeling_llama2 import LlamaForCausalLM as CausalLM
    elif model_args.llama_type == "llama3_1":
        from model.modeling_llama3_1 import LlamaForCausalLM as CausalLM
    elif model_args.llama_type == "qwen2_5":
        from model.modeling_qwen2_5 import Qwen2ForCausalLM as CausalLM
    else:
        raise NotImplementedError

    model = CausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    if model.config.tie_word_embeddings:
        rank0_print(">> tie_word_embeddings")
        model.lm_head.weight = model.model.embed_tokens.weight

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=True,
    )

    # Freeze the model
    for name, param in model.named_parameters():
        if "medusa" in name:
            rank0_print(name)
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}/kv_lr_{training_args.learning_rate}"

    # Load data
    rank0_print("Loading data...")
    train_dataset = datasets.load_from_disk(data_args.data_path).shuffle(seed=42)
    data_collator = DataCollatorForSupervisedDataset(max_length=data_args.max_length)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # Start trainner
    rank0_print("Starting train...")
    trainer = CustomizedTrainer(model=model, args=training_args, **data_module)
    trainer.train()


if __name__ == "__main__":
    train()
