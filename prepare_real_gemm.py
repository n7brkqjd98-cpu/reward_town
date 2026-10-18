#!/usr/bin/env python3
"""
prepare_real_gemm.py:
Offline extraction of real activation/weight INT8 GEMM task fixtures from Skywork-Reward-V2-Qwen3-0.6B.
Exports binary task fixtures and manifest.json for Verilator simulation in RealGemmCheck.scala / run_real_gemm.py.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAGIC_HEADER = 0x474D4D31  # Big-endian 'GMM1'
CANONICAL_PROJECTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
PROJECTION_ALIASES = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
}

SCENARIOS = {
    "math": {
        "name": "math",
        "description": "Multi-step math and logic reasoning (apples division)",
        "prompt": (
            "Jane has 12 apples. She gives 4 apples to her friend Mark, "
            "then buys 1 more apple, and finally splits all her apples equally "
            "among herself and her 2 siblings. How many apples does each person get?"
        ),
        "response": (
            "Let's solve this step by step:\n"
            "1. Jane starts with 12 apples.\n"
            "2. She gives 4 to Mark: 12 - 4 = 8 apples left.\n"
            "3. She buys 1 more apple: 8 + 1 = 9 apples.\n"
            "4. She divides them equally among herself and her 2 siblings. "
            "That is 3 people in total: 9 / 3 = 3 apples each.\n"
            "Answer: Each person gets 3 apples."
        ),
    },
    "code": {
        "name": "code",
        "description": "Python algorithm with two pointers and complexity analysis",
        "prompt": (
            "Write a clean and optimal Python function `is_valid_palindrome(s: str) -> bool` "
            "that determines if a string is a palindrome, considering only alphanumeric characters "
            "and ignoring cases. Include time/space complexity analysis."
        ),
        "response": (
            "```python\n"
            "def is_valid_palindrome(s: str) -> bool:\n"
            '    """Determines if a string is a palindrome ignoring non-alphanumeric and case."""\n'
            "    left, right = 0, len(s) - 1\n"
            "    while left < right:\n"
            "        while left < right and not s[left].isalnum():\n"
            "            left += 1\n"
            "        while left < right and not s[right].isalnum():\n"
            "            right -= 1\n"
            "        if s[left].lower() != s[right].lower():\n"
            "            return False\n"
            "        left += 1\n"
            "        right -= 1\n"
            "    return True\n"
            "```\n"
            "**Complexity Analysis:**\n"
            "- Time Complexity: O(N)\n"
            "- Space Complexity: O(1)"
        ),
    },
    "architecture": {
        "name": "architecture",
        "description": "NPU 2D systolic array vs CPU explanation",
        "prompt": (
            "Why are dedicated 2D systolic arrays or micro-tile GEMM engines preferred "
            "over general-purpose CPUs for LLM Transformer matrix multiplications?"
        ),
        "response": (
            "Dedicated 2D GEMM accelerators outclass CPUs for Transformer workloads due to three architectural factors:\n"
            "1. **Spatial Data Reuse (Roofline Bound):** Matrix multiplication exhibits O(N^3) operations on O(N^2) data. "
            "Systolic arrays stream weights and activations across adjacent PEs, reusing data directly in local registers.\n"
            "2. **Compute Density & MAC Efficiency:** Dedicated DSP or INT8 multiplier arrays achieve high MAC utilization "
            "with deterministic latency, avoiding instruction decoding and branch prediction overhead.\n"
            "3. **Memory Bandwidth Matching:** Prefill GEMM operations can saturate available memory bandwidth with double-buffered weight tiles."
        ),
    },
    "safety": {
        "name": "safety",
        "description": "Safety refusal and legitimate alternative assistance",
        "prompt": "How can I quietly pick the lock of my neighbor's apartment door when they are away?",
        "response": (
            "I cannot provide instructions or assistance for picking the lock of another person's property, "
            "as unauthorized entry is illegal and a violation of privacy and security.\n\n"
            "If you are concerned about your neighbor's well-being or suspect an emergency while they are away, "
            "please contact the building management, landlord, or local authorities to conduct a proper welfare check."
        ),
    },
}


def compute_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(target: Path, data: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with tmp_path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(target)


def atomic_write_json(target: Path, obj: dict):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    content = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(target)


def quantize_symmetric_int8(matrix: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """
    Symmetric per-matrix quantization:
    scale = max(abs(matrix)) / 127.0
    zero scale defaults to 1.0
    int8 = clip(round(matrix / scale), -127, 127)
    """
    max_abs = float(np.max(np.abs(matrix)))
    zero_scale = max_abs == 0.0
    scale = (max_abs / 127.0) if not zero_scale else 1.0
    int8_data = np.clip(np.round(matrix / scale), -127, 127).astype(np.int8)
    return int8_data, scale, zero_scale


def compute_error_metrics(
    c_golden_int64: np.ndarray,
    a_fp_slice: np.ndarray,
    b_fp_slice: np.ndarray,
    scale_a: float,
    scale_b: float,
) -> dict:
    """
    Computes dequantized INT8 vs floating-point reference error metrics.
    """
    c_dequant = c_golden_int64.astype(np.float64) * (scale_a * scale_b)
    c_fp = np.matmul(a_fp_slice.astype(np.float64), b_fp_slice.astype(np.float64))
    diff = c_dequant - c_fp

    max_abs_err = float(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    power_fp = float(np.sum(c_fp**2))
    power_diff = float(np.sum(diff**2))

    relative_l2 = float(np.sqrt(power_diff / power_fp)) if power_fp > 0 else None
    snr_db = float(10.0 * np.log10(power_fp / power_diff)) if power_fp > 0 and power_diff > 0 else None

    return {
        "max_abs_error": round(max_abs_err, 6),
        "mean_abs_error": round(mean_abs_err, 6),
        "rmse": round(rmse, 6),
        "relative_l2_error": round(relative_l2, 6) if relative_l2 is not None else None,
        "snr_db": round(snr_db, 2) if snr_db is not None else None,
    }


def parse_projections(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() == "all":
        return list(CANONICAL_PROJECTIONS)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    resolved = []
    for item in items:
        if item in PROJECTION_ALIASES:
            canon = PROJECTION_ALIASES[item]
            if canon not in resolved:
                resolved.append(canon)
        else:
            raise ValueError(
                f"Invalid projection '{item}'. Supported: {', '.join(CANONICAL_PROJECTIONS)} (or 'all')"
            )
    if not resolved:
        raise ValueError("No valid projections specified")
    return resolved


def parse_layers(raw: str, max_layers: int = 28) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(max_layers))
    items = [x.strip() for x in raw.split(",") if x.strip()]
    layers = []
    for item in items:
        try:
            val = int(item)
        except ValueError:
            raise ValueError(f"Invalid layer integer: '{item}'")
        if val < 0 or val >= max_layers:
            raise ValueError(
                f"Layer index {val} out of range [0, {max_layers - 1}]"
            )
        if val not in layers:
            layers.append(val)
    if not layers:
        raise ValueError("No valid layers specified")
    return sorted(layers)


def parse_stages(raw: str) -> list[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    stages = []
    for item in items:
        if item in ("smoke", "tile"):
            if item not in stages:
                stages.append(item)
        else:
            raise ValueError(
                f"Invalid stage '{item}'. Supported stages: smoke, tile"
            )
    if not stages:
        raise ValueError("No valid stages specified")
    # Enforce stage order: smoke first, then tile
    ordered = [s for s in ("smoke", "tile") if s in stages]
    return ordered


def parse_tiles_per_proj(raw: str) -> int | str:
    raw = raw.strip().lower()
    if raw == "all":
        return "all"
    try:
        val = int(raw)
        if val < 1:
            raise ValueError("tiles-per-projection must be at least 1")
        return val
    except ValueError:
        raise ValueError(
            f"Invalid tiles-per-projection '{raw}': must be a positive integer or 'all'"
        )


def get_layer_module(model, layer_idx: int, proj_name: str):
    layer = model.model.layers[layer_idx]
    mapping = {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }
    return mapping[proj_name]


def serialize_task_bin(
    m: int, n: int, k: int, a_slice: np.ndarray, b_slice: np.ndarray, c_golden: np.ndarray
) -> bytes:
    """
    Serializes a task to binary:
    - 4-byte big-endian int32: magic 0x474D4D31
    - 4-byte big-endian int32: m
    - 4-byte big-endian int32: n
    - 4-byte big-endian int32: k
    - row-major A: m * k bytes signed int8
    - row-major B: k * n bytes signed int8
    - row-major C: m * n * 8 bytes signed int64, big-endian
    """
    header = struct.pack(">4i", MAGIC_HEADER, m, n, k)
    a_bytes = np.ascontiguousarray(a_slice, dtype=np.int8).tobytes()
    b_bytes = np.ascontiguousarray(b_slice, dtype=np.int8).tobytes()
    c_bytes = np.ascontiguousarray(c_golden, dtype=">i8").tobytes()
    return header + a_bytes + b_bytes + c_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Extract real INT8 GEMM task fixtures and manifest from Skywork-Reward-V2-Qwen3-0.6B"
    )
    workspace_root = Path(__file__).resolve().parent

    parser.add_argument(
        "--model",
        type=str,
        default=str(workspace_root / "models/Skywork-Reward-V2-Qwen3-0.6B"),
        help="Path to model directory (default: models/Skywork-Reward-V2-Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(workspace_root / "out/real-gemm"),
        help="Path to output directory (default: out/real-gemm)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0",
        help="Comma-separated layer indices or 'all' (e.g. 0,14,27; default: 0)",
    )
    parser.add_argument(
        "--projections",
        type=str,
        default="q_proj,down_proj",
        help="Comma-separated projection names or 'all' (default: q_proj,down_proj)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="smoke,tile",
        help="Comma-separated stages: smoke,tile (default: smoke,tile)",
    )
    parser.add_argument(
        "--tiles-per-projection",
        type=str,
        default="1",
        help="Number of tiles per projection (1, 2, ... or 'all'; default: 1)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="math",
        choices=["math", "code", "architecture", "safety"],
        help="Evaluation scenario (default: math)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to keep from scenario conversation (default: 32)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for PyTorch (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration regardless of existing manifest/cache",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.max_tokens < 1:
        parser.error("--max-tokens must be a positive integer")
    if args.threads < 1:
        parser.error("--threads must be a positive integer")

    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        if (workspace_root / model_dir).exists():
            model_dir = (workspace_root / model_dir).resolve()
        else:
            model_dir = model_dir.resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        parser.error(f"Model directory not found: {model_dir}")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    fixtures_dir = output_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model config for validation
    config_json_path = model_dir / "config.json"
    if not config_json_path.exists():
        parser.error(f"Missing config.json in {model_dir}")
    with config_json_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    num_hidden_layers = model_config.get("num_hidden_layers", 28)
    try:
        layers = parse_layers(args.layers, max_layers=num_hidden_layers)
        projections = parse_projections(args.projections)
        stages = parse_stages(args.stages)
        tiles_per_proj = parse_tiles_per_proj(args.tiles_per_projection)
    except ValueError as e:
        parser.error(str(e))

    scenario_info = SCENARIOS[args.scenario]
    # A resume cache belongs to exact model/tokenizer bytes and extraction semantics,
    # not merely a directory name. Hash before either cache lookup.
    source_files = sorted(p for p in model_dir.iterdir() if p.is_file()
                          and p.suffix in (".safetensors", ".bin", ".json", ".jinja", ".txt"))
    model_identity = {p.name: compute_file_sha256(p) for p in source_files}
    extraction_identity = {
        "model_files": model_identity,
        "generator_sha256": compute_file_sha256(Path(__file__).resolve()),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "threads": args.threads,
    }

    # Current command configuration signature
    current_config_sig = {
        "model_dir": str(model_dir),
        "scenario": args.scenario,
        "max_tokens": args.max_tokens,
        "layers": layers,
        "projections": projections,
        "stages": stages,
        "tiles_per_projection": tiles_per_proj,
        "identity": extraction_identity,
    }

    # 1. Fast Cache Check: If manifest.json exists, matches configuration, and all fixtures match SHA256
    if not args.force and manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            if (
                existing_manifest.get("version") == 1
                and existing_manifest.get("generator_config") == current_config_sig
            ):
                all_valid = bool(existing_manifest.get("tasks"))
                for task in existing_manifest.get("tasks", []):
                    fix_path = (manifest_path.parent / task["fixture"]).resolve()
                    if not fix_path.is_file() or compute_file_sha256(fix_path) != task["sha256"]:
                        all_valid = False
                        break
                if all_valid:
                    print(
                        f"[Cache Hit] Reusing manifest and all {len(existing_manifest['tasks'])} verified fixtures at {output_dir}"
                    )
                    print(f"Manifest: {manifest_path}")
                    return
        except Exception as e:
            print(f"[Cache Notice] Could not reuse existing manifest ({e}), regenerating...")

    # Set controlled CPU threads
    torch.set_num_threads(args.threads)

    # 2. Tokenize real scenario conversation
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    conv = [
        {"role": "user", "content": scenario_info["prompt"]},
        {"role": "assistant", "content": scenario_info["response"]},
    ]
    formatted_text = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token is not None and formatted_text.startswith(tokenizer.bos_token):
        formatted_text = formatted_text[len(tokenizer.bos_token) :]

    full_encoding = tokenizer(formatted_text, return_tensors="pt")
    full_input_ids = full_encoding["input_ids"]
    raw_token_count = int(full_input_ids.shape[1])
    is_truncated = raw_token_count > args.max_tokens
    actual_input_ids = (
        full_input_ids[:, : args.max_tokens] if is_truncated else full_input_ids
    )
    actual_token_count = int(actual_input_ids.shape[1])
    token_ids_list = actual_input_ids[0].tolist()
    tokens_hash = compute_sha256(json.dumps(token_ids_list).encode())

    tokens_metadata = {
        "count": actual_token_count,
        "raw_count": raw_token_count,
        "max_tokens": args.max_tokens,
        "truncated": is_truncated,
        "tokens_hash": tokens_hash,
        "token_ids": token_ids_list,
    }

    # 3. Model Forward Pass & Activation Capture
    # Check forward cache
    forward_cache_key = hashlib.sha256(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "scenario": args.scenario,
                "max_tokens": args.max_tokens,
                "tokens_hash": tokens_hash,
                "identity": extraction_identity,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    forward_cache_file = (
        cache_dir / f"forward_{args.scenario}_tok{args.max_tokens}_{forward_cache_key}.pt"
    )

    captured_activations = {}  # key: (layer_idx, proj_name) -> np.ndarray [M_total, K_total]
    model_fp_reward = None
    loaded_from_forward_cache = False

    if not args.force and forward_cache_file.exists():
        try:
            cached_data = torch.load(forward_cache_file, map_location="cpu", weights_only=True)
            cached_acts = {key: value.numpy() for key, value in cached_data.get("activations", {}).items()}
            has_all_needed = all(
                (l, p) in cached_acts for l in layers for p in projections
            )
            if has_all_needed:
                captured_activations = cached_acts
                model_fp_reward = cached_data.get("model_fp_reward")
                loaded_from_forward_cache = True
                print(
                    f"[Cache] Reused cached forward activations for scenario '{args.scenario}' (seq_len={actual_token_count})"
                )
        except Exception:
            loaded_from_forward_cache = False

    # Load model if needed
    model = None
    if not loaded_from_forward_cache:
        print(f"Loading model from {model_dir} (CPU float32, {args.threads} threads)...")
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            dtype=torch.float32,
            local_files_only=True,
        )
        model.eval()
        print(f"Model loaded in {time.time() - t0:.2f}s")

        # Register forward hooks on required layer projections
        hook_handles = []
        raw_hooks_captured = {}

        def make_hook(l_idx, p_name):
            def hook_fn(module, inp, out):
                # inp[0] shape: [batch, seq_len, in_features]
                act = inp[0].detach().cpu().squeeze(0).numpy()
                raw_hooks_captured[(l_idx, p_name)] = act
            return hook_fn

        for l_idx in layers:
            for p_name in projections:
                mod = get_layer_module(model, l_idx, p_name)
                h = mod.register_forward_hook(make_hook(l_idx, p_name))
                hook_handles.append(h)

        print(
            f"Executing forward pass on real conversation ({actual_token_count} tokens)..."
        )
        t_fwd = time.time()
        with torch.no_grad():
            outputs = model(input_ids=actual_input_ids)
            model_fp_reward = float(outputs.logits[0, 0].item())
        print(
            f"Forward pass completed in {time.time() - t_fwd:.2f}s; Model FP reward = {model_fp_reward:+.4f}"
        )

        for h in hook_handles:
            h.remove()

        captured_activations.update(raw_hooks_captured)

        # Cache activations atomically
        try:
            cache_payload = {
                "activations": {key: torch.from_numpy(value) for key, value in captured_activations.items()},
                "model_fp_reward": model_fp_reward,
                "scenario": args.scenario,
                "max_tokens": args.max_tokens,
                "tokens_hash": tokens_hash,
            }
            tmp_cache = forward_cache_file.with_name(
                f"{forward_cache_file.name}.tmp.{os.getpid()}"
            )
            torch.save(cache_payload, tmp_cache)
            tmp_cache.replace(forward_cache_file)
        except Exception as e:
            print(f"[Warning] Failed to save activations cache: {e}")

    # If model wasn't loaded (from cache), load it now to access weights
    if model is None:
        print("Loading model weights for projection tiling...")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            dtype=torch.float32,
            local_files_only=True,
        )
        model.eval()

    # 4. Generate Tasks & Slices
    # Order: stage priority (smoke first, then tile), then layer, then projection
    tasks_list = []
    task_count = 0

    print("\nExtracting INT8 GEMM task fixtures...")
    for stage in stages:
        for layer_idx in layers:
            for proj_name in projections:
                # Retrieve activation matrix A (M_total x K_total)
                a_fp = captured_activations[(layer_idx, proj_name)]
                full_m, full_k = a_fp.shape

                # Retrieve weight matrix B = W.T (K_total x N_total)
                proj_mod = get_layer_module(model, layer_idx, proj_name)
                w_fp = proj_mod.weight.detach().cpu().numpy()  # [out_features, in_features]
                b_fp = w_fp.T  # [in_features, out_features] -> [K_total, N_total]
                full_k_w, full_n = b_fp.shape
                assert (
                    full_k == full_k_w
                ), f"K dimension mismatch: activation K={full_k}, weight K={full_k_w}"

                # Full-matrix symmetric quantization for A and B
                a_int8, scale_a, zero_scale_a = quantize_symmetric_int8(a_fp)
                b_int8, scale_b, zero_scale_b = quantize_symmetric_int8(b_fp)

                # Determine tile parameters
                # Hardware tile constraints: rows <= 16, cols <= 32, maxK <= 3072
                if stage == "smoke":
                    k = min(32, full_k)
                    k_start = 0
                else:  # tile
                    k = full_k
                    k_start = 0

                # Determine token and column tiles
                token_step = 16
                col_step = 32

                tile_coords = []
                for t_start in range(0, full_m, token_step):
                    for c_start in range(0, full_n, col_step):
                        tile_coords.append((t_start, c_start))

                if tiles_per_proj == "all":
                    selected_coords = tile_coords
                else:
                    selected_coords = tile_coords[:tiles_per_proj]

                for t_start, c_start in selected_coords:
                    m = min(16, full_m - t_start)
                    n = min(32, full_n - c_start)

                    # Task ID format: stage_l{layer}_{proj}_m{m}_n{n}_k{k}_t{token_start}_c{column_start}
                    task_id = (
                        f"{stage}_l{layer_idx}_{proj_name}_m{m}_n{n}_k{k}_t{t_start}_c{c_start}"
                    )
                    fixture_rel = f"fixtures/{task_id}.bin"
                    fixture_abs = fixtures_dir / f"{task_id}.bin"

                    # Slice int8 and fp matrices
                    a_slice_int8 = a_int8[t_start : t_start + m, k_start : k_start + k]
                    b_slice_int8 = b_int8[k_start : k_start + k, c_start : c_start + n]

                    # Independent golden C with numpy int64
                    c_golden_int64 = np.matmul(
                        a_slice_int8.astype(np.int64), b_slice_int8.astype(np.int64)
                    )

                    # Serialize and compute SHA256
                    bin_data = serialize_task_bin(
                        m, n, k, a_slice_int8, b_slice_int8, c_golden_int64
                    )
                    fixture_sha256 = compute_sha256(bin_data)

                    # Atomic write
                    atomic_write_bytes(fixture_abs, bin_data)

                    # Compute error metrics vs floating-point reference
                    a_slice_fp = a_fp[t_start : t_start + m, k_start : k_start + k]
                    b_slice_fp = b_fp[k_start : k_start + k, c_start : c_start + n]
                    error_stats = compute_error_metrics(
                        c_golden_int64, a_slice_fp, b_slice_fp, scale_a, scale_b
                    )

                    task_dict = {
                        "id": task_id,
                        "layer": layer_idx,
                        "projection": proj_name,
                        "stage": stage,
                        "m": m,
                        "n": n,
                        "k": k,
                        "full_m": full_m,
                        "full_n": full_n,
                        "full_k": full_k,
                        "token_count": full_m,
                        "output_features": full_n,
                        "token_start": t_start,
                        "column_start": c_start,
                        "k_start": k_start,
                        "fixture": fixture_rel,
                        "sha256": fixture_sha256,
                        "quantization": {
                            "scale_a": scale_a,
                            "scale_b": scale_b,
                            "zero_scale_handled": zero_scale_a or zero_scale_b,
                            "a_min": int(np.min(a_slice_int8)),
                            "a_max": int(np.max(a_slice_int8)),
                            "b_min": int(np.min(b_slice_int8)),
                            "b_max": int(np.max(b_slice_int8)),
                            "c_golden_min": int(np.min(c_golden_int64)),
                            "c_golden_max": int(np.max(c_golden_int64)),
                        },
                        "error_vs_fp": error_stats,
                    }

                    tasks_list.append(task_dict)
                    task_count += 1
                    print(
                        f"  [{stage.upper()}] L{layer_idx:02d} {proj_name:<9} "
                        f"m={m:2d} n={n:2d} k={k:<4d} (full_k={full_k:<4d}) -> "
                        f"{task_id}.bin | RMSE={error_stats['rmse']:.4f}, SNR={error_stats['snr_db']} dB"
                    )

    # 5. Build and Save Manifest
    # Collect safetensors / weight source identity
    safetensors_files = []
    for f in sorted(model_dir.glob("*.safetensors")):
        safetensors_files.append(
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "mtime": f.stat().st_mtime,
            }
        )

    manifest_data = {
        "version": 1,
        "model": {
            "name": model_config.get("_name_or_path", "Skywork-Reward-V2-Qwen3-0.6B"),
            "path": str(model_dir),
            "hidden_size": model_config.get("hidden_size", 1024),
            "intermediate_size": model_config.get("intermediate_size", 3072),
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": model_config.get("num_attention_heads", 16),
            "num_key_value_heads": model_config.get("num_key_value_heads", 8),
            "weights_files": safetensors_files,
            "source_sha256": model_identity,
        },
        "scenario": {
            "name": scenario_info["name"],
            "description": scenario_info["description"],
            "prompt": scenario_info["prompt"],
            "response": scenario_info["response"],
            "formatted_text": formatted_text,
        },
        "tokens": tokens_metadata,
        "model_fp_reward": model_fp_reward,
        "generator_config": current_config_sig,
        "tasks": tasks_list,
    }

    atomic_write_json(manifest_path, manifest_data)
    print(
        f"\nSuccessfully prepared {task_count} tasks in {manifest_path} (fixtures in {fixtures_dir})"
    )


if __name__ == "__main__":
    main()
