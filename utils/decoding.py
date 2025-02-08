import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import torch
import time

from typing import Dict, Any
from transformers import PreTrainedTokenizer

from arguments import SampleArgs
from utils.misc import spec_stream, log_csv, rank0_print
from utils.sampling import sample, norm_logits, max_fn
from utils.graph_infer import GraphInferenceEngine
from utils.medusa_utils import (
    generate_candidates,
    decode_tree_candidates,
    exactly_match_sampling,
    update_buffer_ngram,
)
from utils.n_gram import N_Gram


@torch.inference_mode()
def Autoregressive(
    tokenizer: PreTrainedTokenizer,
    graph_engine: GraphInferenceEngine,
    input_ids: torch.Tensor,
    gen_len: int,
    sample_args: SampleArgs,
    verbose: bool = False,
):
    # reset all cache
    graph_engine.engine.kv_cache.reset()

    logits = graph_engine.engine.model_prefill(input_ids=input_ids)

    if verbose:
        graph_engine.engine.kv_cache.print_status()

    next_token = sample(norm_logits(logits[:, -1, :], sample_args=sample_args, input_ids=input_ids.squeeze().tolist()))
    output_ids = torch.cat([input_ids.squeeze(0), next_token.squeeze(0)]).tolist()

    if verbose:
        spec_stream(next_token[0], tokenizer, "cyan")

    n = 0
    ar_latency_record = {}
    length_list = [i * 10240 for i in range(1, 11)]

    torch.cuda.synchronize()
    time1 = time.time()
    while n < gen_len:
        logits = graph_engine.engine.model(
            input_ids=next_token, kv_cache=graph_engine.engine.kv_cache, retri_cache=None
        )
        next_token = sample(norm_logits(logits[:, -1, :], sample_args=sample_args, input_ids=output_ids))
        output_ids.append(next_token.item())
        n += 1

        if n in length_list:
            torch.cuda.synchronize()
            time2 = time.time()
            ar_latency_record[n] = (time2 - time1) / n * 1000

        if verbose:
            spec_stream(next_token[0], tokenizer, "cyan")
    torch.cuda.synchronize()
    time2 = time.time()

    return (time2 - time1) / n * 1000, ar_latency_record


@torch.inference_mode()
def SpecLong(
    tokenizer: PreTrainedTokenizer,
    graph_engine: GraphInferenceEngine,
    input_ids: torch.LongTensor,
    gamma: int,
    gen_len: int,
    sample_args: SampleArgs,
    ar_latency_record: Dict[str, Any] = None,
    file_path: str = None,
    record_args: Dict[str, Any] = None,
    verbose: bool = False,
    tree_decoding: bool = False,
    ngram_topk: int = 0,
    medusa_buffers: dict = None,
    ngram_retriever: N_Gram = None,
):

    # reset all cache
    graph_engine.engine.clear_kv()

    _ = graph_engine.engine.model_prefill(input_ids=input_ids[:, :-1])
    logits = graph_engine.engine.model_prefill(input_ids=input_ids[:, -1:])  # [1, 1, vocab_size]

    if verbose:
        graph_engine.engine.kv_cache.print_status()
        graph_engine.engine.retri_cache.print_status()

    resample_count = 0
    accepted_count = 0
    ngram_accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    input_ids_list = input_ids.squeeze().tolist()
    next_token = sample(norm_logits(logits[:, -1, :], sample_args=sample_args, input_ids=input_ids_list))

    if ngram_topk > 0:
        ngram_retriever.preload(input_ids_list[: -sample_args.penalty_length])

    if verbose:
        spec_stream(next_token[0], tokenizer, "cyan")

    total_input_ids = input_ids_list + [next_token.item()]

    cur_gen_len, factor = 0, 1
    total_time = 0

    if ar_latency_record is not None:
        ar_token_num_list = list(ar_latency_record.keys())
        record_idx = 0

    while cur_gen_len < gen_len:

        if (
            (file_path is not None)
            and (int(os.environ.get("LOCAL_RANK", "0")) == 0)
            and (ar_latency_record is not None)
            and ((record_idx < len(ar_token_num_list)) and (cur_gen_len >= ar_token_num_list[record_idx]))
        ):
            ar_token_num = ar_token_num_list[record_idx]
            ar_latency = ar_latency_record[ar_token_num]
            sd_latency = total_time / cur_gen_len * 1000
            record_idx += 1

            header = "ar_token_num,sd_token_num,ar_latency,sd_latency,speed_up\n"
            entry = f"{ar_token_num},{cur_gen_len},{ar_latency},{sd_latency},{ar_latency / sd_latency}\n"
            log_csv(f"{file_path}/latency_res.csv", header, entry)

        if cur_gen_len > 512 * factor:
            factor += 1
            rank0_print("\n", "*" * 50, flush=True)
            rank0_print(f"Generate {cur_gen_len} Tokens Already", flush=True)
            acceptance_rate = accepted_count / draft_count
            ngram_acceptance_rate = ngram_accepted_count / draft_count
            rank0_print(f"accepted rate {acceptance_rate} ngram accepted rate {ngram_acceptance_rate}", flush=True)
            rank0_print("*" * 50, flush=True)

            if (file_path is not None) and (int(os.environ.get("LOCAL_RANK", "0")) == 0):
                header = "sd_gen_tokens,accept_rate,ngram_acc\n"
                entry = f"{cur_gen_len},{acceptance_rate},{ngram_acceptance_rate}\n"
                log_csv(f"{file_path}/acc_res.csv", header, entry)

        torch.cuda.synchronize()
        time1 = time.time()

        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)

        if graph_engine.engine.kv_cache.seq_len < graph_engine.engine.retri_cache.retri_max_budget:
            raise NotImplementedError("seq_len must large than retri_max_budget")

        elif tree_decoding:
            assert medusa_buffers is not None, f"medusa_buffers is None"

            logits = graph_engine.engine.spec_infer(input_ids=next_token)  # [gamma, 1, vocab_size]
            draft_count += logits.shape[0]

            combined_prob = norm_logits(logits[:, -1], sample_args, input_ids=total_input_ids)

            # candidates: [leaf_nodes, gamma + 1], tree_candidates: [1, tree_nodes]
            candidates, tree_candidates = generate_candidates(
                combined_prob=combined_prob,
                tree_indices=medusa_buffers["tree_indices"],
                retrieve_indices=medusa_buffers["retrieve_indices"],
                root_token=next_token,
            )

            new_medusa_buffers = medusa_buffers
            if ngram_topk > 0:
                retrieved_ngrams = []
                backbone_head_pred_list = torch.unique(candidates[:, 1]).tolist()
                for backbone_head_pred in backbone_head_pred_list:
                    retrieved_ngrams.extend(ngram_retriever.topk(backbone_head_pred, ngram_topk))
                retrieved_ngrams = [(next_token.item(),) + ngram for ngram in retrieved_ngrams]

                if len(retrieved_ngrams) > 0:
                    new_medusa_buffers, tree_candidates, candidates = update_buffer_ngram(
                        medusa_buffers=medusa_buffers,
                        tree_candidates=tree_candidates,
                        candidates=candidates,
                        N=ngram_retriever.n + 1,
                        retrieved_ngrams=retrieved_ngrams,
                    )

            # Use tree attention to verify the candidates and get predictions
            # [leaf_nodes, gamma + 1, vocab_size]
            verify_logits = decode_tree_candidates(
                graph_engine=graph_engine,
                tree_candidates=tree_candidates,
                medusa_position_ids=new_medusa_buffers["medusa_position_ids"],  # batch size is 1
                cur_kv_len=graph_engine.engine.kv_cache.seq_len,
                retrieve_indices=new_medusa_buffers["retrieve_indices"],
                medusa_attn_mask=new_medusa_buffers["medusa_attn_mask"],
            )

            verify_probs = norm_logits(verify_logits, sample_args, input_ids=total_input_ids)

            # [leaf_nodes, gamma + 1]
            sampling_result = exactly_match_sampling(
                candidates=candidates,
                full_verify_probs=verify_probs,
                retrieved_ngram_num=len(retrieved_ngrams) if ngram_topk > 0 else 0,
            )

            cur_seq_len = graph_engine.engine.kv_cache.seq_len - new_medusa_buffers["medusa_attn_mask"].shape[-1]

            preserve_indices = cur_seq_len + new_medusa_buffers["retrieve_indices"][sampling_result["index"]]
            preserve_indices = preserve_indices[: len(sampling_result["accept_tokens"])]

            key_cache_to_preserve = graph_engine.engine.kv_cache.key_cache[:, :, :, preserve_indices, :]
            value_cache_to_preserve = graph_engine.engine.kv_cache.value_cache[:, :, :, preserve_indices, :]

            graph_engine.engine.kv_cache.key_cache[
                :, :, :, cur_seq_len : cur_seq_len + len(sampling_result["accept_tokens"]), :
            ] = key_cache_to_preserve
            graph_engine.engine.kv_cache.value_cache[
                :, :, :, cur_seq_len : cur_seq_len + len(sampling_result["accept_tokens"]), :
            ] = value_cache_to_preserve

            graph_engine.engine.kv_cache.seq_len -= new_medusa_buffers["medusa_attn_mask"].shape[-1]
            graph_engine.engine.kv_cache.seq_len += len(sampling_result["accept_tokens"])
            graph_engine.engine.retri_cache.update_retri_cache(graph_engine.engine.kv_cache)

            if verbose:
                spec_stream(sampling_result["accept_tokens"][1:], tokenizer, "green")
                if sampling_result["all_accept"]:
                    spec_stream(sampling_result["resample_token"].squeeze(0), tokenizer, "blue")
                else:
                    spec_stream(sampling_result["resample_token"].squeeze(0), tokenizer, "red")

            accepted_count += len(sampling_result["accept_tokens"]) - 1  # delete resample token
            ngram_accepted_count += sampling_result["max_acc_ngram_len"]
            next_token = sampling_result["resample_token"]

            acc_token_list = sampling_result["accept_tokens"][1:].tolist()
            total_input_ids.extend(acc_token_list)
            total_input_ids.append(sampling_result["resample_token"].item())

            if ngram_topk > 0:
                end_idx = len(total_input_ids) - ngram_retriever.n - sample_args.penalty_length
                for i in range(end_idx - len(sampling_result["accept_tokens"]), end_idx):
                    ngram_retriever.update(total_input_ids[i : i + ngram_retriever.n])

            cur_gen_len += len(sampling_result["accept_tokens"])

        else:
            logits = graph_engine.engine.spec_infer(input_ids=next_token)  # [gamma, 1, vocab_size]

            # [gamma, vocab_size]
            speculation_prob = norm_logits(logits[:, -1], sample_args, input_ids=total_input_ids).squeeze()
            speculation_probs = list(speculation_prob)

            # [gamma, vocab_size] => [gamma, 1] => [1, gamma]
            pred_token_idx = sample(speculation_prob).transpose(0, 1)
            verify_tokens = torch.cat([next_token, pred_token_idx], dim=-1)

            draft_count += logits.shape[0]

            logits = graph_engine.engine.model_verify(input_ids=verify_tokens)  # [1, gamma + 1, vocab_size]

            if isinstance(verify_tokens, torch.Tensor):
                verify_tokens = verify_tokens.squeeze().tolist()

            count = 0
            verify_probs = []

            probs = norm_logits(logits[0], sample_args=sample_args, input_ids=total_input_ids)
            for i in range(gamma + 1):
                verify_probs.append(probs[i])

            pass_tokens = torch.full((1, gamma + 2), -100, device=graph_engine.engine.model.device)
            pass_tokens[:, 0] = next_token
            generated_ids = verify_tokens[1:]

            for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
                r = torch.rand(1, device=graph_engine.engine.model.device)
                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                    count += 1
                    accepted_count += 1
                    cur_gen_len += 1
                    pred_token_idx = torch.tensor([[i]]).to(graph_engine.engine.model.device)
                    pass_tokens[:, count] = pred_token_idx
                    if verbose:
                        spec_stream(i, tokenizer, "green")
                    total_input_ids.append(i)

                else:
                    resample_count += 1
                    cur_gen_len += 1
                    pred_token_idx = sample(max_fn(verify_prob - speculation_prob))
                    pass_tokens[:, count + 1] = pred_token_idx
                    if verbose:
                        spec_stream(pred_token_idx, tokenizer, "red")
                    total_input_ids.append(pred_token_idx.item())
                    break

            # update cache
            graph_engine.engine.kv_cache.seq_len -= len(generated_ids) - count
            graph_engine.engine.retri_cache.update_retri_cache(graph_engine.engine.kv_cache)

            if count == len(generated_ids):
                target_sample_count += 1
                cur_gen_len += 1
                pred_token_idx = sample(verify_probs[-1])
                pass_tokens[:, count + 1] = pred_token_idx
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, "blue")
                total_input_ids.append(pred_token_idx.item())
                count += 1

            next_token = pred_token_idx

        torch.cuda.synchronize()
        total_time += time.time() - time1

    acceptance_rate = accepted_count / draft_count
    ngram_acceptance_rate = ngram_accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        rank0_print(
            f"Use {total_time} sec to generate {cur_gen_len} tokens (now {graph_engine.engine.kv_cache.seq_len} tokens), Tokens/s: {cur_gen_len / (total_time)}",
            flush=True,
        )
        rank0_print(
            f"accepted rate {acceptance_rate}, ngram accepted rate {ngram_acceptance_rate}, avg generated tokens {avg_tokens}",
            flush=True,
        )

    if file_path is not None:

        final_output_text = tokenizer.decode(
            total_input_ids[input_ids.shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        with open(f"{file_path}/out.txt", "w", encoding="utf-8") as wp:
            wp.write(final_output_text)

        sd_latency = total_time / cur_gen_len * 1000
        header = "acceptance_rate,ngram_acc,token/s,avg_tokens,prefill,gen_len,latency,speed_up\n"
        entry = f"{acceptance_rate},{ngram_acceptance_rate},{cur_gen_len / total_time},{avg_tokens},{input_ids.shape[1]},{cur_gen_len},{sd_latency},{record_args['baseline'] / sd_latency}\n"

        if record_args is not None:
            for k, v in record_args.items():
                header = header.replace("\n", f",{k}\n")
                entry = entry.replace("\n", f",{v}\n")

        log_csv(f"{file_path}/res.csv", header, entry)

    if ngram_retriever is not None:
        ngram_retriever.clear()

    return acceptance_rate, total_time / cur_gen_len * 1000
