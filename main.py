import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import json
import torch

from tqdm import tqdm
from datasets import load_dataset
from termcolor import colored
from datetime import datetime
from transformers import AutoTokenizer, set_seed

from arguments import get_args
from models.tp import _get_global_rank
from models.cache import FlashSimpleCache, RetrievalCache
from utils.decoding import Autoregressive, SpecLong
from utils.misc import print_config, rank0_print
from utils.graph_infer import GraphInferenceEngine
from utils.medusa_utils import generate_medusa_buffers
from utils.n_gram import N_Gram
from utils.medusa_choices import decoding_tree


if __name__ == "__main__":

    set_seed(42)

    args, sample_args = get_args()
    assert args.gamma == args.num_heads + 1

    if args.model_type == "llama2":
        from models.modeling_llama2 import LlamaForCausalLM as CausalLM
    elif args.model_type == "llama3_1":
        from models.modeling_llama3_1 import LlamaForCausalLM as CausalLM

        sample_args.assistant_token_id = "18328,78191"
    elif args.model_type == "qwen2_5":
        from models.modeling_qwen2_5 import Qwen2ForCausalLM as CausalLM
    else:
        raise NotImplementedError

    ######## model initialization ########
    target = CausalLM.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    if args.tp_size > 1:
        from models.tp import apply_tp, init_dist

        rank0_print("Applying tensor parallel to model ...")
        target.config.tp_size = args.tp_size
        rank, global_group = init_dist()
        apply_tp(target, [i for i in range(args.tp_size)], group=global_group)
    target = target.to("cuda").eval()

    if target.config.tie_word_embeddings:
        rank0_print(">> tie_word_embeddings")
        target.lm_head.weight = target.model.embed_tokens.weight

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path_prefix}/{args.target}", use_fast=True, legacy=False)

    ######## load dataset ########
    datasetparent = "data/pg19/"
    dataset = load_dataset("json", data_files=[datasetparent + "pg19-test.json"], split="train")
    tokenized_prompts = []
    for i in tqdm(range(10)):
        prompt = dataset[i]["text"][5000:]
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
        assert tokenized_prompt.shape[-1] > args.prefill_len
        tokenized_prompts.append(tokenized_prompt)
    if args.prompt_id >= len(tokenized_prompts):
        raise ValueError

    ######## file initialization ########
    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_path_root = "output" if not args.debug else "debug"
    file_path = os.path.join(file_path_root, time_string + str(os.environ.get("LOCAL_RANK", "0")))
    if _get_global_rank == 0:
        os.makedirs(file_path)
        with open(f"{file_path}/config.json", "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
            json.dump(vars(sample_args), json_file, indent=4)

    print_config(
        target,
        args.prefill_len,
        args.gen_len,
        args.gamma,
        file_path=file_path,
        method="SpecLong",
        sample_args=sample_args,
        spec_args={
            "retrival_cache_budget": args.retrival_max_budget,
            "num_heads": args.num_heads,
        },
    )

    ######## ngram & tree init ########
    ngram_retriever = N_Gram(args.gamma)
    medusa_buffers = generate_medusa_buffers(decoding_tree, device=target.device)
    max_ngram_group = max([path[0] for path in decoding_tree]) + 1

    ######## cache init ########
    cache = FlashSimpleCache(
        target,
        args.prefill_len
        + args.gen_len
        + len(medusa_buffers["tree_indices"])
        + max_ngram_group * args.ngram_topk * ngram_retriever.n
        + 1,
    )
    retri_cache = RetrievalCache(
        target, retri_max_budget=args.retrival_max_budget, gamma=args.gamma, prefill_len=args.prefill_len
    )

    graph_engine = GraphInferenceEngine(target, cache, retri_cache)
    graph_engine.initialize_cuda_graph(args.gamma)

    ######## Warm up for baseline ########
    if args.debug:
        if args.model_type == "llama3_1":
            # llama3.1-8b
            ar_latency = 173
            ar_latency_record = {
                10240: 41,
                20480: 55,
                30720: 68,
                40960: 83,
                51200: 97,
                61440: 113,
                71680: 128,
                81920: 143,
                92160: 158,
            }
        elif args.model_type == "llama2":
            ar_latency = 114
            ar_latency_record = {
                10240: 35.4,
                20480: 44.1,
                30720: 52.8,
                40960: 61.55,
                51200: 70.3,
                61440: 79.1,
                71680: 87.8,
                81920: 96.6,
                92160: 105.3,
            }
        elif args.target == "Qwen2.5-7B":
            ar_latency = 142.94
            ar_latency_record = {
                10240: 35.56411,
                20480: 46.57908,
                30720: 57.83444,
                40960: 69.30128,
                51200: 81.2599,
                61440: 93.67075,
                71680: 106.12493,
                81920: 118.47555,
                92160: 130.7438,
            }
        elif args.target == "Qwen2.5-14B":
            ar_latency = 278.34
            ar_latency_record = {
                10240: 65.83,
                20480: 87.25,
                30720: 109.21,
                40960: 131.742,
                51200: 155.457,
                61440: 180.197,
                71680: 205.04,
                81920: 229.65,
                92160: 254.07,
            }
        else:
            ar_latency = 0
            ar_latency_record = None
    else:
        n_warmups = 1
        input_ids = tokenized_prompts[0].to(target.device)[:, : args.prefill_len]
        for i in tqdm(range(n_warmups), desc="Autoregressive Warmup"):
            _ = Autoregressive(
                tokenizer, graph_engine, input_ids, gen_len=100, sample_args=sample_args, verbose=args.verbose
            )

        for input_ids in tqdm(tokenized_prompts[:1], desc="Autoregressive Test"):
            input_ids = input_ids.to(target.device)[:, : args.prefill_len]
            ar_latency, ar_latency_record = Autoregressive(
                tokenizer, graph_engine, input_ids, gen_len=args.gen_len, sample_args=sample_args, verbose=args.verbose
            )

        rank0_print(colored(f"[Autoregressive] average latency: {ar_latency} ms", "red"))

    ######## Warm up for our method ########
    n_warmups = 1
    input_ids = tokenized_prompts[0].to(target.device)[:, : args.prefill_len]
    for i in tqdm(range(n_warmups), desc="SpecLong Warmup"):
        _ = SpecLong(
            tokenizer,
            graph_engine,
            input_ids,
            gamma=args.gamma,
            gen_len=100,
            sample_args=sample_args,
            verbose=args.verbose,
            tree_decoding=args.tree_decoding,
            medusa_buffers=medusa_buffers,
        )

    all_acceptance_rate, all_latency = [], []

    if args.prompt_id >= 0:
        tokenized_prompts = tokenized_prompts[args.prompt_id : args.prompt_id + 1]

    for input_ids in tqdm(tokenized_prompts, desc="SpecLong Test"):
        input_ids = input_ids.to(target.device)[:, : args.prefill_len]

        acceptance_rate, latency = SpecLong(
            tokenizer,
            graph_engine,
            input_ids,
            gamma=args.gamma,
            gen_len=args.gen_len,
            sample_args=sample_args,
            verbose=args.verbose,
            ar_latency_record=ar_latency_record,
            file_path=file_path,
            record_args={"budget": args.retrival_max_budget, "gamma": args.gamma, "baseline": ar_latency},
            tree_decoding=args.tree_decoding,
            ngram_topk=args.ngram_topk,
            medusa_buffers=medusa_buffers,
            ngram_retriever=ngram_retriever,
        )

        all_acceptance_rate.append(acceptance_rate)
        all_latency.append(latency)

        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_path_root = "output" if not args.debug else "debug"
        file_path = os.path.join(file_path_root, time_string)
        if _get_global_rank == 0:
            os.makedirs(file_path)
            with open(f"{file_path}/config.json", "w") as json_file:
                json.dump(vars(args), json_file, indent=4)
                json.dump(vars(sample_args), json_file, indent=4)

    method_latency = sum(all_latency) / len(all_latency)

    rank0_print(
        colored(
            f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"
        )
    )
    rank0_print(colored(f"[SpecLong] average latency: {method_latency} ms", "red"))
    rank0_print(colored(f"[E2E Speedup]: {ar_latency / method_latency}", "red"))
