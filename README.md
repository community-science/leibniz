# Leibniz

Leibniz is a protocol for scientific work: a way to make identity, frame,
statement, measurement, record, and program artifacts explicit, addressable,
validated, and composable without a central authority.

## Principles

1. **Protocol, not platform.** Leibniz is defined by artifact formats,
   semantics, and executable validation. Reference implementations demonstrate
   compliance; they do not define the protocol by themselves.
2. **Code, not prose, is the specification.** Public claims about the protocol
   should be grounded in working validation, interpreters, or tests.
3. **Typed, versioned, composable artifacts.** Durable records should have
   explicit identities, declared inputs and outputs, stated invariants, and
   composition behavior.
4. **Measurement discipline.** Raw measurements should remain re-projectable
   into future frames. Leaderboards, frontiers, and rankings are derived views,
   not the underlying state.
5. **Field-agnostic core.** No core primitive should privilege one scientific
   domain when a generic typed artifact can express the same idea.

## Contribution Model

All changes enter through pull requests. Non-documentation changes must include
tests that prove the semantic contract being introduced or changed. Pull
requests should be small enough that a reviewer can explain their purpose,
boundary, design choice, and tests before merge.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the working rules.

## Development

Use the repository-local Miniforge environment for local development:

```bash
bash scripts/setup_environment.sh
source scripts/activate_environment.sh
```

`environment.yml` declares CUDA-enabled PyTorch on Linux and CPU/Mac PyTorch on
non-Linux hosts using conda selectors.

Hosted console publishers that install Leibniz with pip should install the
console extra and cache pip downloads in CI:

```bash
python -m pip install -e '.[console]'
```

Run the local checks:

```bash
python -m pytest
python -m ruff check .
python -m pyright
python -m leibniz._repository_policy .
python -m build --no-isolation
```

Use narrower checks while iterating when they prove the contract you changed,
such as a focused `python -m pytest tests/test_example.py`. Before opening or
updating a non-documentation pull request, run the full local check set above
unless the pull request explains why a check was skipped. Heavy benchmark
training, GPU jobs, and network-dependent result repository workflows are
manual validation paths, not routine pull-request checks.

## Repository Layout

- `src/leibniz/`: Python package containing protocol artifact definitions,
  validation, CLI commands, benchmark orchestration, local result workflows,
  and console data generation.
- `src/leibniz/_formats/`: shared structured-format helpers used by protocol
  artifact readers and writers.
- `src/leibniz/benchmarks/`: packaged benchmark manifests and benchmark-local
  protocol documents.
- `src/leibniz/console/`: console data, artifact indexing, code generation,
  and the embedded web console source under `_web_src/`.
- `tests/`: Python contract, workflow, and policy tests. Fixtures live under
  `tests/fixtures/`; console web contract tests live beside the Python tests.
- `scripts/`: repository environment setup and activation helpers.
- `.github/workflows/`: pull-request and main-branch CI checks.
- `results/`, caches, checkpoints, and local result checkouts: local
  runtime state only; do not commit these unless a deterministic producer and
  review path are explicitly documented.

## Result Workflow

Benchmark runs write local state under `results/` by default. That path is
ignored by this source repository so the console can discover result views
without making benchmark state part of the source checkout.

Before starting benchmark runs, prepare the result repository with the same
setup command for either Hugging Face API auth or Git/SSH auth:

```bash
leibniz benchmark init --repo owner/leibniz-results
```

With Hugging Face API auth, install `huggingface_hub` and authenticate with
`hf auth login` or `HF_TOKEN`; `benchmark init` can create the empty dataset
repository as part of setup. With SSH-only Git auth, create the empty Hugging
Face dataset repository first; `benchmark init` then clones it into
`results/` using `git@hf.co:datasets/owner/leibniz-results.git`. If you prefer
to keep the clone outside this source checkout, clone it elsewhere, symlink
`results/` to that clone, and then run the same `benchmark init` command to
validate and scaffold it:

```bash
git clone git@hf.co:datasets/owner/leibniz-results.git ../leibniz-results
ln -s ../leibniz-results results
leibniz benchmark init --repo owner/leibniz-results
```

After benchmark evaluation has written accepted evidence and materialized local
views, push the existing result state to that repository:

```bash
leibniz benchmark publish --repo owner/leibniz-results
```

`benchmark publish` commits and pushes the current result checkout. It uses the
Hugging Face API when API credentials are available for `--repo`; otherwise it
falls back to plain Git push when `results/` is a Git checkout. Without a
Hugging Face account, prepare the same local result directory and use
`--no-push` when publishing local-only result state:

```bash
leibniz benchmark init --local-only
```

Run the canonical reference trainer against an explicit architecture:

```bash
leibniz benchmark train \
  digits \
  --architecture tests/fixtures/architecture/digits_pool.json
```

`benchmark train` is a reference implementation for local benchmark training and
evaluation. It trains supplied architecture manifests and does not propose or
choose architectures. `--architecture` may be repeated and may name either a
manifest file or a directory. When omitted, `benchmark train` discovers
architecture manifests under `results/training/<benchmark>/pending/`, skips
manifests whose deterministic completed training summary already exists for the
active benchmark and training controls, sequentially trains the remaining
manifests, and then moves completed queue entries to the sibling
`completed/` directory. `benchmark train digits` narrows training to the Digits
benchmark, and omitting the benchmark name scans all local benchmarks. The
command accepts repeated `--benchmark-root` arguments and otherwise resolves
packaged benchmark roots by benchmark id. The default local training profile is
an uncapped convergence run:
competence gates are checked every 32 steps, every gate check updates the
running progress record, model checkpoint artifacts are written every gate
check, patience is 6 gate checks, and convergence min delta is `1e-3`.
Override those with `--train-steps`, `--gate-check-interval`,
`--model-checkpoint-gate-interval`, `--convergence-patience`, and
`--convergence-min-delta` when you need a shorter diagnostic run or a cheaper
checkpoint cadence.

The digits benchmark samples rectangular canvases with independently varying
height and width. Observation formation now derives the lower canvas floor from
generic component discriminability analysis rather than a fixed pixel extent per
digit; the benchmark manifest declares the scalar discriminability margin used
by that live analysis. Spatial variation is sampled as affine matrix
coordinates inside the benchmark-owned identity-preserving envelope. Candidate
architectures must therefore accept variable spatial input shapes, for example
by using adaptive pooling before any fixed readout. Fixed `input_shape`-only
architectures are rejected for sampled digits runs because later training
batches may have different canvas dimensions than the validation batch used
during initial inspection.

Digits is a single-label finite-outcome benchmark. The task contract is to
predict which one of the ten digit identities is visible in the observation.
The benchmark does not use an explicit sequence-length or complexity coordinate
for scoring; observation difficulty is derived from the number of possible
distinguishable states under the active scoring contract. In the current
single-digit task this includes the ten digit identities and a finite grid over
the identity-preserving affine nuisance envelope as perceived on the sampled
canvas. Larger canvases can therefore expose more distinguishable affine states;
sampled canvas shape is not counted as extra score-bearing difficulty by itself
unless it changes what nuisance states can be distinguished. This keeps the
model output fixed at a 10-way probability measure while allowing score to grow
as formation rules add real distinguishable observation states.

While a benchmark run is training, gate-check progress may be written under
`results/training/` as the current training-run record. The console may
materialize those records as tentative plot points, using the training run's own
score estimate, but leaderboard rows and frontiers are composed only from
completed accepted benchmark evaluations. Saved model checkpoints, checkpoint
artifact sidecars, and model manifests are written under `results/models/`, and
completed training runs atomically replace their running records with final
training summaries in the same `results/training/` location. Benchmark evidence
is generated separately by handing a checkpoint artifact record to the evaluator
with a fresh unpredictable evaluation seed; those evaluation records are written
under `results/evaluations/` and replace matching tentative points as the
accepted local benchmark records consumed by the console. Each accepted
evaluation is a self-contained benchmark evaluation bundle: it embeds the
benchmark manifest, architecture manifest, model manifest, checkpoint artifact
record, model inspection, measurement dataset, score view, sampled competence
record, evaluation protocol, evaluation curriculum, seed, and throughput.
`benchmark evaluate` accepts an explicit `--checkpoint-artifact`; when omitted,
it discovers selected checkpoint artifact sidecars from local training summaries
under `results/training/` and evaluates completed training runs whose matching
checkpoints do not already have accepted evaluation bundles. Pending queue
manifests and in-progress training records are ignored by evaluation discovery.
`benchmark evaluate digits` narrows evaluation to the Digits benchmark, and
omitting the benchmark name scans all local benchmarks. The command accepts
repeated `--benchmark-root` arguments and otherwise resolves packaged benchmark
roots by benchmark id. Materialized console views are refreshed by evaluation
and written per benchmark under
`results/views/<benchmark>/benchmark_results.json`. The console exposes a
single vertical `Score` axis, accepted directly from the evaluation harness.

Remove generated local benchmark state while keeping the architecture manifest
suite and result checkout scaffolding:

```bash
leibniz benchmark clean
```

Commit and push the existing local result checkout:

```bash
leibniz benchmark publish
```

Commit without pushing when you only want to save local result state:

```bash
leibniz benchmark publish --no-push
```

## License

This repository is dedicated to the public domain under CC0-1.0. See
[LICENSE](LICENSE).
