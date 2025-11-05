"""End-to-end profiler for Djinn using a representative synthetic workload."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import requests
import torch
from pathlib import Path

from genie import capture
from djinn.core.lazy_tensor import LazyTensor
from djinn.core.subgraph_builder import SubgraphBuilder, RemoteSubgraph

logger = logging.getLogger(__name__)


@dataclass
class RemoteTiming:
    client_graph_serialize_ms: float
    client_tensor_serialize_ms: float
    client_request_ms: float
    client_deserialize_ms: float
    server_deserialize_ms: float
    server_execution_ms: float
    server_serialize_ms: float
    server_total_ms: float
    total_client_ms: float
    client_network_overhead_ms: float
    graph_size_bytes: int
    tensor_size_bytes: int
    result_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileSummary:
    warm_remote: RemoteTiming
    local_total_ms: float
    slowdown_factor: float
    overhead_ms: float
    cold_remote: Optional[RemoteTiming] = None

    @property
    def remote(self) -> RemoteTiming:
        """Backward-compatible alias for warm_remote."""
        return self.warm_remote

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["warm_remote"] = self.warm_remote.to_dict()
        if self.cold_remote is not None:
            payload["cold_remote"] = self.cold_remote.to_dict()
        else:
            payload["cold_remote"] = None
        payload["remote"] = payload["warm_remote"]
        return payload


class ComprehensiveProfiler:
    """Profiles Djinn by executing a synthetic subgraph remotely vs locally."""

    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: int = 8888,
        spawn_server: bool = True,
        request_timeout: float = 120.0,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        self.request_timeout = request_timeout
        self.spawn_server = spawn_server
        self.session = requests.Session()
        self.server_process: Optional[subprocess.Popen] = None
        self.server_log_file: Optional[Any] = None
        self.server_log_path = Path(
            os.environ.get("GENIE_PROFILER_SERVER_LOG", "profiling_server.log")
        ).resolve()
        self._gpt2_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

        if spawn_server:
            self._start_server()

    def close(self) -> None:
        if self.spawn_server:
            self._stop_server()
        self.session.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def profile(
        self,
        batch_size: int = 256,
        hidden_size: int = 1024,
        warmup_runs: int = 1,
        workload: str = "synthetic",
        seq_length: int = 64,
        model_name: str = "sshleifer/tiny-gpt2",
    ) -> ProfileSummary:
        """Profile Djinn remote execution using a representative workload."""

        if warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")

        if workload not in {"synthetic", "gpt2"}:
            raise ValueError(f"Unsupported workload '{workload}'")

        if workload == "synthetic":
            subgraph_dict, input_data, _ = self._build_synthetic_workload(batch_size, hidden_size)
            local_executor = lambda: self._execute_local_synthetic(input_data, hidden_size)
        else:
            subgraph_dict, input_data, local_executor = self._build_gpt2_workload(
                model_name=model_name,
                batch_size=batch_size,
                seq_length=seq_length,
            )

        cold_remote: Optional[RemoteTiming] = None
        measured_remote: Optional[RemoteTiming] = None
        measured_output: Optional[torch.Tensor] = None

        total_runs = warmup_runs + 1
        for run_idx in range(total_runs):
            timing, output = self._execute_remote(subgraph_dict, input_data)

            if run_idx == 0 and warmup_runs > 0:
                cold_remote = timing

            if run_idx == warmup_runs:
                measured_remote = timing
                measured_output = output

        if measured_remote is None or measured_output is None:
            raise RuntimeError("Failed to collect measured remote timing")

        local_total_ms, local_output = local_executor()

        slowdown = measured_remote.total_client_ms / local_total_ms if local_total_ms > 0 else float("inf")
        overhead = measured_remote.total_client_ms - local_total_ms

        # SANITY CHECK 1: Results must match
        max_diff = torch.max(torch.abs(measured_output - local_output)).item()
        if not torch.allclose(measured_output, local_output, atol=1e-4, rtol=1e-3):
            logger.error(
                "❌ CRITICAL: Remote and local results don't match!\n"
                "   Max difference: %.6f\n"
                "   Remote shape: %s\n"
                "   Local shape: %s\n"
                "   This indicates a correctness bug in the remote executor.",
                max_diff,
                measured_output.shape,
                local_output.shape,
            )
        else:
            logger.info("✅ Result verification passed (max diff: %.2e)", max_diff)

        # SANITY CHECK 2: Remote should not be faster than local
        if slowdown < 1.0:
            logger.warning(
                "⚠️  WARNING: Remote is faster than local (%.2fx speedup)!\n"
                "   Remote: %.2f ms\n"
                "   Local:  %.2f ms\n"
                "   This may indicate:\n"
                "   - Local baseline not using GPU\n"
                "   - Missing synchronization in local timing\n"
                "   - Remote is caching results\n"
                "   - Different computation paths\n"
                "   Please verify measurement correctness.",
                1.0 / slowdown,
                measured_remote.total_client_ms,
                local_total_ms,
            )

        return ProfileSummary(
            warm_remote=measured_remote,
            local_total_ms=local_total_ms,
            slowdown_factor=slowdown,
            overhead_ms=overhead,
            cold_remote=cold_remote,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_synthetic_workload(
        self,
        batch_size: int,
        hidden_size: int,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Any]:
        subgraph, input_data = self._build_synthetic_subgraph(batch_size, hidden_size)
        subgraph["model_id"] = f"synthetic_b{batch_size}_h{hidden_size}"
        return subgraph, input_data, None

    def _execute_local_synthetic(
        self,
        input_data: Dict[str, torch.Tensor],
        hidden_size: int,
    ) -> Tuple[float, torch.Tensor]:
        return self._execute_local(input_data, hidden_size)
    def _start_server(self) -> None:
        if self.server_process is not None:
            return

        env = os.environ.copy()
        cmd = [
            sys.executable,
            "-m",
            "genie.runtime.simple_server",
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
        ]
        self.server_log_file = self.server_log_path.open("w", encoding="utf-8")
        self.server_process = subprocess.Popen(  # noqa: S603, S607
            cmd,
            stdout=self.server_log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logger.info(
            "Started profiling server PID=%s (logs -> %s)",
            self.server_process.pid,
            self.server_log_path,
        )
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        health_url = f"{self.server_url}/health"

        while time.time() < deadline:
            try:
                response = self.session.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    logger.info("Profiling server ready at %s", health_url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)

        raise RuntimeError(f"Profiling server not reachable at {health_url}")

    def _stop_server(self) -> None:
        if self.server_process is None:
            return
        self.server_process.terminate()
        try:
            self.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
            self.server_process.wait(timeout=5)
        logger.info("Stopped profiling server")
        self.server_process = None
        if self.server_log_file is not None:
            self.server_log_file.close()
            self.server_log_file = None

    # ------------------------------------------------------------------
    # Workload construction
    # ------------------------------------------------------------------
    def _build_synthetic_subgraph(
        self,
        batch_size: int,
        hidden_size: int,
    ) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Construct a representative semantic subgraph for profiling."""

        input_tensor = torch.randn(batch_size, hidden_size)
        scale_tensor = torch.randn(batch_size, hidden_size)
        bias_tensor = torch.randn(batch_size, hidden_size)

        subgraph = {
            "operations": [
                {
                    "op_id": 3,
                    "operation": "aten::mul",
                    "inputs": [0, 1],
                    "kwargs": {},
                },
                {
                    "op_id": 4,
                    "operation": "aten::add",
                    "inputs": [3, 2],
                    "kwargs": {},
                },
                {
                    "op_id": 5,
                    "operation": "aten::relu",
                    "inputs": [4],
                    "kwargs": {},
                },
                {
                    "op_id": 6,
                    "operation": "aten::softmax",
                    "inputs": [5],
                    "kwargs": {"dim": -1},
                },
            ],
            "input_tensors": {
                "0": {"shape": list(input_tensor.shape), "dtype": str(input_tensor.dtype)},
                "1": {"shape": list(scale_tensor.shape), "dtype": str(scale_tensor.dtype)},
                "2": {"shape": list(bias_tensor.shape), "dtype": str(bias_tensor.dtype)},
            },
            "output_id": 6,
        }

        input_data = {
            "0": input_tensor,
            "1": scale_tensor,
            "2": bias_tensor,
        }

        return subgraph, input_data

    # ------------------------------------------------------------------
    # Remote execution path
    # ------------------------------------------------------------------
    def _execute_remote(
        self,
        subgraph: Dict[str, Any],
        input_data: Dict[str, torch.Tensor],
    ) -> tuple[RemoteTiming, torch.Tensor]:

        graph_buffer = io.BytesIO()
        graph_serialize_start = time.perf_counter()
        torch.save(subgraph, graph_buffer)
        client_graph_serialize_ms = (time.perf_counter() - graph_serialize_start) * 1000
        graph_bytes = graph_buffer.getvalue()

        tensor_buffer = io.BytesIO()
        tensor_serialize_start = time.perf_counter()
        torch.save(input_data, tensor_buffer)
        client_tensor_serialize_ms = (time.perf_counter() - tensor_serialize_start) * 1000

        tensor_bytes = tensor_buffer.getvalue()

        files = {
            "request": ("request.pt", io.BytesIO(graph_bytes), "application/octet-stream"),
            "tensors": ("tensors.pt", io.BytesIO(tensor_bytes), "application/octet-stream"),
        }

        request_start = time.perf_counter()
        response = self.session.post(
            f"{self.server_url}/execute_subgraph",
            files=files,
            timeout=self.request_timeout,
        )
        client_request_ms = (time.perf_counter() - request_start) * 1000
        response.raise_for_status()

        server_timing_header = response.headers.get("X-Djinn-Server-Timing")
        if server_timing_header:
            try:
                server_timing = json.loads(server_timing_header)
            except json.JSONDecodeError:
                server_timing = {}
        else:
            server_timing = {}

        deserialize_start = time.perf_counter()
        try:
            if response.content.startswith(b"NUMPY001"):
                from ..core.serialization import deserialize_tensor

                result_tensor = deserialize_tensor(response.content)
            else:
                result_tensor = torch.load(io.BytesIO(response.content))
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.error("Failed to deserialize server response: %s", exc)
            logger.error("Server response (truncated): %s", response.content[:200])
            raise
        client_deserialize_ms = (time.perf_counter() - deserialize_start) * 1000

        server_total_ms = float(server_timing.get("total_ms", 0.0))
        client_network_overhead_ms = (
            max(client_request_ms - server_total_ms, 0.0)
            if server_total_ms > 0
            else client_request_ms
        )

        total_client_ms = (
            client_graph_serialize_ms
            + client_tensor_serialize_ms
            + client_request_ms
            + client_deserialize_ms
        )

        timing = RemoteTiming(
            client_graph_serialize_ms=client_graph_serialize_ms,
            client_tensor_serialize_ms=client_tensor_serialize_ms,
            client_request_ms=client_request_ms,
            client_deserialize_ms=client_deserialize_ms,
            server_deserialize_ms=float(server_timing.get("deserialize_ms", 0.0)),
            server_execution_ms=float(server_timing.get("execution_ms", 0.0)),
            server_serialize_ms=float(server_timing.get("serialize_ms", 0.0)),
            server_total_ms=server_total_ms,
            total_client_ms=total_client_ms,
            client_network_overhead_ms=client_network_overhead_ms,
            graph_size_bytes=len(graph_bytes),
            tensor_size_bytes=len(tensor_bytes),
            result_size_bytes=len(response.content),
        )

        return timing, result_tensor.cpu()

    # ------------------------------------------------------------------
    # Local baseline
    # ------------------------------------------------------------------
    def _execute_local_synthetic(
        self,
        input_data: Dict[str, torch.Tensor],
        hidden_size: int,
    ) -> tuple[float, torch.Tensor]:
        """Execute synthetic workload locally with proper GPU synchronization."""
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        x = input_data["0"].to(device)
        scale = input_data["1"].to(device)
        bias = input_data["2"].to(device)

        # Warmup runs to prime GPU kernels (same as remote gets)
        for _ in range(3):
            with torch.no_grad():
                _ = torch.softmax(torch.relu((x * scale) + bias), dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Measured run
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            scaled = x * scale
            summed = scaled + bias
            activated = torch.relu(summed)
            output = torch.softmax(activated, dim=-1)

        # CRITICAL: Synchronize BEFORE .cpu() to measure actual GPU execution
        if device.type == "cuda":
            torch.cuda.synchronize()

        total_ms = (time.perf_counter() - start) * 1000
        output_cpu = output.detach().cpu()
        return total_ms, output_cpu

    # ------------------------------------------------------------------
    # GPT-2 workload helpers
    # ------------------------------------------------------------------
    def _build_gpt2_workload(
        self,
        model_name: str,
        batch_size: int,
        seq_length: int,
    ) -> tuple[Dict[str, Any], Dict[str, torch.Tensor], Any]:
        cache_key = (model_name, batch_size, seq_length)
        if cache_key in self._gpt2_cache:
            cached = self._gpt2_cache[cache_key]
            return cached["subgraph"], cached["input_data"], cached["local_executor"]

        logger.info(
            "Preparing GPT-2 workload: model=%s batch=%d seq=%d",
            model_name,
            batch_size,
            seq_length,
        )

        capture_device = torch.device("cpu")
        execution_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.use_cache = False
        model.config._attn_implementation = "eager"
        model.to(capture_device)
        model.eval()

        vocab_size = model.config.vocab_size
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        with capture():
            input_ids_lazy = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, seq_length),
                dtype=torch.long,
            )

            outputs = model(
                input_ids=input_ids_lazy,
                use_cache=False,
            )
            logits = outputs.logits
            target = logits.mean(dim=-1)

        builder = SubgraphBuilder()
        remote_subgraph = builder.build_remote_subgraph(target)

        input_ids_concrete = input_ids_lazy.materialize().detach()
        overrides = {
            id(input_ids_lazy): input_ids_concrete,
        }

        subgraph_dict, input_tensor_map = self._serialize_remote_subgraph(
            remote_subgraph,
            materialized_overrides=overrides,
        )

        input_tensor_cpu = {
            key: tensor.detach().cpu() if tensor.device.type != "cpu" else tensor.detach()
            for key, tensor in input_tensor_map.items()
        }

        if execution_device != capture_device:
            model_runtime = model.to(execution_device)
        else:
            model_runtime = model

        def local_executor() -> Tuple[float, torch.Tensor]:
            with torch.no_grad():
                local_input_ids = input_ids_concrete.detach().clone().to(execution_device)
                
                # Warmup runs to prime GPU kernels (same as remote gets)
                for _ in range(3):
                    _ = model_runtime(input_ids=local_input_ids, use_cache=False)
                    if execution_device.type == "cuda":
                        torch.cuda.synchronize(execution_device)
                
                # Measured run
                if execution_device.type == "cuda":
                    torch.cuda.synchronize(execution_device)
                
                start = time.perf_counter()
                outputs_local = model_runtime(
                    input_ids=local_input_ids,
                    use_cache=False,
                )
                logits_local = outputs_local.logits
                summary = logits_local.mean(dim=-1).detach()
                # CRITICAL: Synchronize BEFORE .cpu() to measure actual GPU execution
                if execution_device.type == "cuda":
                    torch.cuda.synchronize(execution_device)
                total_ms = (time.perf_counter() - start) * 1000
                summary_cpu = summary.cpu()
            return total_ms, summary_cpu

        # Set model_id for GPU caching
        subgraph_dict["model_id"] = f"{model_name}_b{batch_size}_s{seq_length}"
        
        self._gpt2_cache[cache_key] = {
            "subgraph": subgraph_dict,
            "input_data": input_tensor_cpu,
            "local_executor": local_executor,
        }

        logger.info(
            "GPT-2 workload prepared: %d operations, %d external tensors, model_id=%s",
            len(subgraph_dict["operations"]),
            len(subgraph_dict["input_tensors"]),
            subgraph_dict["model_id"],
        )

        return subgraph_dict, input_tensor_cpu, local_executor

    def _serialize_remote_subgraph(
        self,
        subgraph: RemoteSubgraph,
        materialized_overrides: Optional[Dict[int, torch.Tensor]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        materialized_overrides = materialized_overrides or {}
        operation_ids = {id(op) for op in subgraph.operations}
        external_tensors: Dict[int, torch.Tensor] = {}
        factory_ops = {
            'aten::randn', 'aten::rand', 'aten::randint',
            'aten::zeros', 'aten::ones', 'aten::empty', 'aten::full',
            'aten::tensor', 'aten::as_tensor', 'aten::from_numpy',
            'aten::arange', 'aten::linspace', 'aten::logspace', 'aten::eye',
        }
        tuple_result_ops = {'aten::split', 'aten::chunk', 'aten::unbind'}
        tuple_result_counters: Dict[Any, int] = {}

        def ensure_external(tensor: Any) -> int:
            if isinstance(tensor, (int, float, bool)):
                tensor = torch.tensor(tensor)

            if isinstance(tensor, LazyTensor):
                tensor_id = id(tensor)
                if tensor_id in external_tensors:
                    return tensor_id
                if tensor_id in materialized_overrides:
                    concrete = materialized_overrides[tensor_id]
                else:
                    concrete = tensor.materialize()
                external_tensors[tensor_id] = concrete.detach()
                return tensor_id

            if isinstance(tensor, torch.Tensor):
                tensor_id = id(tensor)
                if tensor_id not in external_tensors:
                    external_tensors[tensor_id] = tensor.detach()
                return tensor_id

            raise TypeError(f"Unsupported tensor input type: {type(tensor)}")

        # Prime with builder-identified inputs
        for tensor_id, tensor in subgraph.input_tensors.items():
            ensure_external(tensor)

        serialized_operations = []
        for op in subgraph.operations:
            tuple_result_index: Optional[int] = None
            if op.operation in factory_ops:
                operation_ids.discard(id(op))
                continue
            if op.operation in tuple_result_ops:
                base_id = id(op.inputs[0]) if op.inputs else id(op)
                split_size = op.inputs[1] if len(op.inputs) > 1 else None
                dim = op.kwargs.get('dim') if hasattr(op, 'kwargs') else None
                key = (base_id, op.operation, split_size, dim, tuple(sorted(op.kwargs.items())))
                tuple_result_index = tuple_result_counters.get(key, 0)
                tuple_result_counters[key] = tuple_result_index + 1

            op_inputs: list[Any] = []
            for inp in op.inputs:
                if isinstance(inp, LazyTensor):
                    inp_id = id(inp)
                    if inp_id in operation_ids:
                        op_inputs.append(inp_id)
                    else:
                        op_inputs.append(ensure_external(inp))
                elif isinstance(inp, torch.Tensor):
                    op_inputs.append(ensure_external(inp))
                elif isinstance(inp, (int, float, bool)):
                    op_inputs.append({'type': 'scalar', 'value': inp})
                elif inp is None:
                    op_inputs.append({'type': 'none', 'value': None})
                elif isinstance(inp, tuple):
                    op_inputs.append({'type': 'tuple', 'value': [self._encode_literal_value(v) for v in inp]})
                elif isinstance(inp, list):
                    op_inputs.append({'type': 'list', 'value': [self._encode_literal_value(v) for v in inp]})
                elif isinstance(inp, torch.dtype):
                    op_inputs.append({'type': 'dtype', 'value': str(inp)})
                elif isinstance(inp, slice):
                    op_inputs.append(
                        {
                            'type': 'slice',
                            'value': [
                                self._encode_literal_value(inp.start),
                                self._encode_literal_value(inp.stop),
                                self._encode_literal_value(inp.step),
                            ],
                        }
                    )
                else:
                    raise TypeError(
                        f"Unexpected operation input type {type(inp)} for {op.operation}"
                    )

            op_entry = {
                "op_id": id(op),
                "operation": op.operation,
                "inputs": op_inputs,
                "kwargs": op.kwargs,
                "shape": list(op.shape) if op.shape else None,
                "dtype": str(op.dtype) if op.dtype else None,
            }
            if tuple_result_index is not None:
                op_entry["result_index"] = tuple_result_index

            serialized_operations.append(op_entry)

        input_metadata = {
            str(tensor_id): {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for tensor_id, tensor in external_tensors.items()
        }

        serialized = {
            "operations": serialized_operations,
            "input_tensors": input_metadata,
            "output_id": id(subgraph.output_tensor),
            "model_id": "default",  # Will be overridden by caller
        }

        return serialized, external_tensors

    @staticmethod
    def _encode_literal_value(value: Any) -> Any:
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        if isinstance(value, slice):
            return {
                'type': 'slice',
                'value': [
                    ComprehensiveProfiler._encode_literal_value(value.start),
                    ComprehensiveProfiler._encode_literal_value(value.stop),
                    ComprehensiveProfiler._encode_literal_value(value.step),
                ],
            }
        raise TypeError(f"Unsupported literal element type: {type(value)}")


