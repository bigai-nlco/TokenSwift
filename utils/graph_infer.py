import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import gc
import math
import torch

from typing import Optional
from transformers import PreTrainedModel

from models.cache import FlashSimpleCache, RetrievalCache


class InferenceEngine:
    def __init__(
        self,
        model: PreTrainedModel,
        cache: FlashSimpleCache,
        retri_cache: Optional[RetrievalCache] = None,
    ) -> None:

        self.model = model
        self.model.eval()

        self.kv_cache = cache
        self.retri_cache = retri_cache

    @torch.inference_mode()
    def model_prefill(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        print(">> start prefilling", flush=True)

        if input_ids.shape[-1] == 1:
            retri_cache = self.retri_cache
        else:
            retri_cache = None

        iter_prefill = math.ceil(input_ids.shape[1] / 256)
        for i in range(iter_prefill):
            logits = self.model(
                input_ids=input_ids[:, i * 256 : (i + 1) * 256],
                kv_cache=self.kv_cache,
                retri_cache=retri_cache,
            )

        return logits

    @torch.inference_mode()
    def spec_infer(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """darft for partial cache"""

        # [k, seq_len, vocab_size]
        hidden_states_list = self.model.model(
            input_ids=input_ids,
            retri_cache=self.retri_cache,
            spec=True,
            past_kv_len=self.kv_cache.seq_len,
        )

        hidden_states = torch.vstack(hidden_states_list)
        logits = self.model.lm_head(hidden_states)

        return logits

    # @torch.inference_mode()
    # def spec_infer(
    #     self,
    #     input_ids: torch.LongTensor,
    # ) -> torch.LongTensor:
    #     """darft for full cache"""

    #     hidden_states_list = self.model.model(
    #         input_ids=input_ids,
    #         kv_cache=self.kv_cache,
    #         spec=True,
    #     )

    #     hidden_states = torch.vstack(hidden_states_list)
    #     logits = self.model.lm_head(hidden_states)

    #     return logits

    @torch.inference_mode()
    def model_verify(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:

        logits = self.model(
            input_ids=input_ids,
            kv_cache=self.kv_cache,
            retri_cache=self.retri_cache,
        )

        return logits

    def clear_kv(self):
        self.kv_cache.reset()
        self.retri_cache.reset()


def spec_infer_capture_graph(
    engine: InferenceEngine,
    mempool=None,
):
    device = engine.model.device

    # spec infer is incremental decoding
    static_input_ids = torch.full((1, 1), 0, dtype=torch.long, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        static_logits = engine.spec_infer(input_ids=static_input_ids)
        s.synchronize()  # Waits for all operations in stream s to complete. This ensures that all previous CUDA operations have completed.
    torch.cuda.current_stream().wait_stream(s)

    print(f"[draft run] capturing graph for {static_logits.shape[0]}...")
    graph = (
        torch.cuda.CUDAGraph()
    )  # CUDA graphs are used to optimize the execution of a set of operations, significantly improving performance.
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.spec_infer(input_ids=static_input_ids)

    def run(input_ids):
        static_input_ids.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    return run


def model_verify_capture_graph(
    engine: InferenceEngine,
    gamma: int,
    mempool=None,
):
    device = engine.model.device

    # model_verify is verifying gamma tokens
    static_input_ids = torch.full((1, gamma + 1), 0, dtype=torch.long, device=device)

    s = (
        torch.cuda.Stream()
    )  # CUDA streams allow operations to be executed in parallel on the GPU to improve performance.
    # Makes the stream s wait for the current stream to complete. This ensures that subsequent operations do not start while the current stream is still in progress.
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        static_logits = engine.model_verify(input_ids=static_input_ids)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    print(f"[model verify] capturing graph for spec len {gamma}...\n\n")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.model_verify(input_ids=static_input_ids)

    def run(input_ids):
        static_input_ids.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    return run


class GraphInferenceEngine:
    def __init__(
        self,
        model: PreTrainedModel,
        cache: FlashSimpleCache,
        retri_cache: Optional[RetrievalCache] = None,
    ) -> None:

        self.engine = InferenceEngine(model, cache, retri_cache)
        self.callables = {}
        self.mempool = None

    @torch.inference_mode()
    def initialize_cuda_graph(self, gamma: int):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()

        self.callable_spec_infer = spec_infer_capture_graph(
            engine=self.engine,
            mempool=self.mempool,
        )

        self.callable_model_verify = model_verify_capture_graph(
            engine=self.engine,
            gamma=gamma,
            mempool=self.mempool,
        )

        self.engine.clear_kv()

    def clear_kv(self):
        self.engine.clear_kv()

    @torch.inference_mode()
    def sepc_infer(self, input_ids: torch.LongTensor):
        # spec infer
        return self.callable_spec_infer(input_ids)

    @torch.inference_mode()
    def model_verify(self, input_ids: torch.LongTensor):
        # model verify
        return self.callable_model_verify(input_ids)
