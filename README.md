# Leibniz

Leibniz is a protocol for scientific work: a way to make identity, frame,
statement, measurement, record, and program artifacts explicit, addressable,
validated, and composable without a central authority.

How the protocol measures scientific work — universes, tasks, correctness, and
scoring — is described in the **Architecture tab of the console**, an interactive
adaptive precision tree and the console's default view. See [Console](#console)
below to run it.

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
6. **Correctness is convergence.** A prediction is correct to the degree it
   converges to a law under refinement. The protocol scores quantities it
   measures or derives — a convergence gap, a certified distance, a validated
   bit count — never a verdict it is told to accept. Analytic or reference
   solutions enter only as submitted programs, never as a privileged oracle
   inside the scoring path.
7. **Models are submitted, not supplied.** A model is a submitted computation
   graph, evaluated against the same contract and cost meter as any other
   submission. The repository owns the instruments that measure submissions —
   contracts, structural composition, refinement ladders, cost metrology — and
   ships no blessed model or network vocabulary of its own.

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

### Console

The web console's default view is the Architecture tab — an interactive adaptive
precision tree describing universes, tasks, correctness, and scoring. With the
environment loaded, start the development server with:

```bash
leibniz console dev
```

Run the full local check set before review:

```bash
python scripts/validate.py
```

The script is the local command registry for routine pull-request gates; CI uses
the same registry for its Python, package, and console check commands. Run
`python scripts/validate.py --help` to list the named checks available for
targeted iteration.

Use narrower checks while iterating when they prove the contract you changed.
Before opening or updating a non-documentation pull request, run
`python scripts/validate.py` unless the pull request explains why a check was
skipped. Heavy benchmark training, GPU jobs, and network-dependent result
repository workflows are manual validation paths, not routine pull-request
checks.

### Tensor Kernel Conventions

Benchmark tensor programs should stay behind the `leibniz.tensor_runtime`
adapter rather than importing backend tensor libraries directly. A
`TensorBatchProgram` kernel may opt into runtime helpers by declaring an `ops`
keyword-only parameter; `ops.broadcast_zeros(axis_coordinates)` returns a
zero-valued tile with the full coordinate broadcast shape without hand-written
zero-sum expressions. For compiled kernels, prefer a small number of packed
parameter tensors over many scalar parameters so backend compilers bind and
cache fewer buffers.

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

Run the canonical reference trainer against an explicit submitted program:

```bash
leibniz benchmark train \
  digits \
  --program tests/fixtures/programs/digits_inverse_conv_encoder.py
```

Run the same reference trainer against the variable-resolution KS prediction
benchmark with a submitted timestepper program:

```bash
leibniz benchmark train \
  ks \
  --program tests/fixtures/programs/ks_variable_conv.py
```

`benchmark train` is a reference implementation for local benchmark training and
evaluation. It trains supplied submitted program graphs and does not propose or
choose models. `--program` may be repeated and may name either a Python source
file exposing `build_program_graph(runtime)` or a directory of such files. When
omitted, `benchmark train` discovers queued program sources under
`results/programs/<benchmark>/pending/`, skips programs whose deterministic
completed training summary already exists for the active benchmark and training
controls, and sequentially trains the remaining programs. `benchmark train
digits` narrows training to the Digits benchmark, and omitting the benchmark
name scans all local benchmarks. The command accepts repeated `--benchmark-root`
arguments and otherwise resolves packaged benchmark roots by benchmark id. The
default local training profile is an uncapped convergence run with the
hyperparameter-free `loss-search` optimizer:
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
coordinates inside the benchmark-owned identity-preserving envelope. Submitted
Digits inverse programs must therefore accept the observation tensor shape and
emit the latent-vector output declared by their program graph. Prediction
programs can declare symbolic support axes, allowing one program family to run
against variable-size field-valued tasks such as KS without a benchmark-specific
model name.

Digits is an oracle-free inverse benchmark. The task contract is to recover a
latent vector from an observed image under the benchmark-owned differentiable
renderer: identity logits plus continuous nuisance coordinates. Training uses
only reconstruction residual against that renderer; labels are not part of the
training or scoring path. Certified score is the ambient epsilon-entropy of the
product latent space, with image residual converted to latent precision by the
static renderer geometry. Identity contributes only where classes remain
distinguishable at the certified precision, and nuisance bits come from the
continuous affine chart. Larger canvases can expose more distinguishable
nuisance states, but sampled canvas shape is not score-bearing by itself unless
it changes what latent states can be certified.

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
benchmark manifest, submitted program graph, model manifest, checkpoint artifact
record, model inspection, measurement dataset, score view, sampled competence
record, evaluation protocol, evaluation curriculum, seed, and throughput.
`benchmark evaluate` accepts an explicit `--checkpoint-artifact`; when omitted,
it discovers selected checkpoint artifact sidecars from local training summaries
under `results/training/` and evaluates completed training runs whose matching
checkpoints do not already have accepted evaluation bundles. Pending program
sources and in-progress training records are ignored by evaluation discovery.
`benchmark evaluate digits` narrows evaluation to the Digits benchmark, and
omitting the benchmark name scans all local benchmarks. The command accepts
repeated `--benchmark-root` arguments and otherwise resolves packaged benchmark
roots by benchmark id. Materialized console views are refreshed by evaluation
and written per benchmark under
`results/views/<benchmark>/benchmark_results.json`. The console exposes a
single vertical `Score` axis, accepted directly from the evaluation harness.

Remove generated local benchmark state while keeping submitted programs and
result checkout scaffolding:

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
