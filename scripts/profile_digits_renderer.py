"""Profile the Digits benchmark tensor renderer with PyTorch profiler."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from leibniz.benchmark_implementations import load_benchmark
from leibniz.observation_generation import StateSpaceMeasureRequest
from leibniz.tensor_runtime import resolve_tensor_runtime
from leibniz.timing import TimingCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one or more Digits renderer batches.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("src/leibniz/benchmarks/digits"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum", type=float, default=9.321928094887362)
    parser.add_argument("--maximum", type=float, default=10.321928094887362)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument(
        "--memory-summary",
        action="store_true",
        help="Print torch.cuda.memory_summary() after profiling CUDA.",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_root)
    generator = benchmark.generator
    runtime = resolve_tensor_runtime(args.device)
    torch = runtime.torch
    outcome_ids = tuple(
        outcome.id for outcome in benchmark.manifest.resolve_outcome_space().outcomes
    )
    request = StateSpaceMeasureRequest(minimum=args.minimum, maximum=args.maximum)

    for index in range(args.warmup):
        batch = generator(
            shape=args.batch_size,
            seed=args.seed + index,
            include_fields=False,
            runtime=runtime,
            outcome_ids=outcome_ids,
            state_space_request=request,
        )
        batch.require_tensors()
    if runtime.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if runtime.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "cuda_time_total"

    timing = TimingCollector()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profile:
        for index in range(args.repeat):
            batch = generator(
                shape=args.batch_size,
                seed=args.seed + args.warmup + index,
                include_fields=False,
                runtime=runtime,
                outcome_ids=outcome_ids,
                state_space_request=request,
                timing=timing,
                timing_prefix="profile_",
            )
            batch.require_tensors()
    if runtime.device.type == "cuda":
        torch.cuda.synchronize()

    phases = cast(dict[str, dict[str, object]], timing.to_record()["phases"])
    render = phases["profile_batch_tensor_render"]
    print("TimingCollector")
    print(f"  render_seconds: {render['seconds']:.6f}")
    if "profile_batch_tensor_render.mark_chunk" in phases:
        chunk = phases["profile_batch_tensor_render.mark_chunk"]
        print(f"  mark_chunk_seconds: {chunk['seconds']:.6f}")
        print(f"  mark_chunk_calls: {chunk['calls']}")
        print(f"  mark_chunk_samples: {chunk['sample_count']}")
    if runtime.device.type == "cuda":
        print(
            "  peak_allocated_mib: "
            f"{torch.cuda.max_memory_allocated() / (1024 * 1024):.3f}"
        )

    print()
    print(profile.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))

    if args.trace_path is not None:
        profile.export_chrome_trace(str(args.trace_path))
        print(f"wrote chrome trace: {args.trace_path}")
    if args.memory_summary and runtime.device.type == "cuda":
        print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
