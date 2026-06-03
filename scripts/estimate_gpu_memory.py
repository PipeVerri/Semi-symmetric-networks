#!/usr/bin/env python3
"""Estimate GPU memory for a simple LM/hyena configuration (rough approximation).

Usage:
  python scripts/estimate_gpu_memory.py --d_model 32 --n_layer 2 --d_inner 128 \
    --seq_len 1024 --batch_size 256 --precision 16 --vocab_size 12 --model_type hyena

This is a conservative, heuristic estimator (params + optimizer states + activations).
"""
import argparse
import math
import subprocess


def get_gpu_mem_gb():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", shell=True
        )
        # take first GPU
        v = int(out.decode().strip().splitlines()[0])
        return v / 1024.0
    except Exception:
        return None


def estimate_params(d_model, n_layer, d_inner, vocab_size, model_type="transformer", hyena_filter_order=64):
    # embeddings
    params = vocab_size * d_model
    if model_type == "transformer":
        per_layer = 4 * d_model * d_model + 2 * d_model * d_inner
        params += n_layer * per_layer
        # lm_head
        params += d_model * vocab_size
    else:
        # rough hyena estimate: input/out projections per layer + filter MLP
        per_layer = (d_model * (d_model * (2 + 1)))  # in_proj and out_proj rough
        # filter MLP params (implicit filter): hyena_filter_order * d_model * 3 (very rough)
        filter_params = hyena_filter_order * d_model * 3
        params += n_layer * per_layer + filter_params
        params += d_model * vocab_size
    return int(params)


def human(x):
    if x >= 1<<30:
        return f"{x/(1<<30):.2f} GB"
    if x >= 1<<20:
        return f"{x/(1<<20):.2f} MB"
    return f"{x} B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--n_layer", type=int, required=True)
    p.add_argument("--d_inner", type=int, required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--precision", type=int, choices=[16,32], default=16)
    p.add_argument("--vocab_size", type=int, default=30522)
    p.add_argument("--model_type", choices=["transformer","hyena"], default="transformer")
    p.add_argument("--hyena_filter_order", type=int, default=64)
    args = p.parse_args()

    bytes_per_param = 2 if args.precision == 16 else 4

    param_count = estimate_params(args.d_model, args.n_layer, args.d_inner, args.vocab_size, args.model_type, args.hyena_filter_order)
    param_bytes = param_count * bytes_per_param

    # Optimizer states (conservative): assume Adam-like states stored in fp32 => ~8 bytes per param
    opt_bytes = param_count * 8

    # Activations: store activations for backward ~ batch * seq * d_model * bytes * factor
    act_bytes = args.batch_size * args.seq_len * args.d_model * bytes_per_param * max(1.0, args.n_layer * 0.75)

    # Extra buffers (gradients, temporary) ~ 1x params
    extra = param_bytes

    total_bytes = param_bytes + opt_bytes + act_bytes + extra

    total_gb = total_bytes / (1024 ** 3)

    gpu_mem_gb = get_gpu_mem_gb()

    print("--- GPU memory estimator (HEURISTIC) ---")
    print(f"Model type: {args.model_type}")
    print(f"Param count (approx): {param_count:,}")
    print(f"Params memory: {human(param_bytes)}")
    print(f"Optimizer states (est): {human(opt_bytes)}")
    print(f"Activations (est): {human(act_bytes)}")
    print(f"Other buffers: {human(extra)}")
    print(f"Estimated total (training): {total_gb:.2f} GB")
    if gpu_mem_gb is not None:
        print(f"Detected GPU memory: {gpu_mem_gb:.2f} GB")
        print("Fits?", "YES" if total_gb < gpu_mem_gb * 0.9 else "NO (likely not)")
    else:
        print("GPU memory not detected on this machine.")


if __name__ == "__main__":
    main()
