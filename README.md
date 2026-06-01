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
training, GPU jobs, and network-dependent publication or federation workflows
are manual validation paths, not routine pull-request checks.

## Repository Layout

- `src/leibniz/`: Python package containing protocol artifact definitions,
  validation, CLI commands, benchmark orchestration, local publication flows,
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
- `results/`, caches, checkpoints, and local publication checkouts: local
  runtime state only; do not commit these unless a deterministic producer and
  review path are explicitly documented.

## Result Publication Workflow

Benchmark runs write local state under `results/` by default. That path is
ignored by this source repository so the console can discover result views
without making benchmark state part of the source checkout.

Before starting benchmark runs, prepare the result repository with the same
setup command for either Hugging Face API auth or Git/SSH auth:

```bash
leibniz results init-publication --repo owner/leibniz-results
```

With Hugging Face API auth, install `huggingface_hub` and authenticate with
`hf auth login` or `HF_TOKEN`; `init-publication` can create the empty dataset
repository as part of setup. With SSH-only Git auth, create the empty Hugging
Face dataset repository first; `init-publication` then clones it into
`results/` using `git@hf.co:datasets/owner/leibniz-results.git`. If you prefer
to keep the clone outside this source checkout, clone it elsewhere, symlink
`results/` to that clone, and then run the same `init-publication` command to
validate and scaffold it:

```bash
git clone git@hf.co:datasets/owner/leibniz-results.git ../leibniz-results
ln -s ../leibniz-results results
leibniz results init-publication --repo owner/leibniz-results
```

After benchmark runs have written results locally, push them to that repository:

```bash
leibniz results publish --push --repo owner/leibniz-results
```

`results publish --push` uses the Hugging Face API when API credentials are
available for `--repo`; otherwise it falls back to plain Git push when
`results/` is a Git checkout. Without a Hugging Face account, prepare the same
local result directory and skip any push step:

```bash
leibniz results init-publication --local-only
```

Run benchmarks against that checkout:

```bash
leibniz benchmark shakedown \
  --benchmark-root src/leibniz/benchmarks/digits
```

`benchmark shakedown` is the fast smoke-test path. The default `benchmark run`,
`benchmark loop`, and `results propose` local training profile is an uncapped
convergence run: validation every 250 steps, 500 minimum steps before early
stopping, patience 12, and convergence min delta `1e-3`. Override those with
`--train-steps`, `--convergence-min-steps`, `--validation-interval`,
`--convergence-patience`, and `--convergence-min-delta` when you need a shorter
diagnostic run.

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

Digits is a variable-length token-sequence benchmark. The task contract is to
predict the complete digit sequence visible in the observation; exact-sequence
scoring gives probability credit only to that full sequence. Sequence model
interfaces assign probability to finite token sequences with model-determined
length; the benchmark does not require an end-of-sequence token or an enumerated
vector over every possible sequence. The local PyTorch training workflow in this
repository is transitional and remains separate from the benchmark task
definition so richer training recipes can move to a separate repository.

While a benchmark loop is training a reserved candidate, validation checkpoints
are written under `results/training-progress/` and materialized into the local
benchmark result view as running leaderboard entries with accumulated validation
history. Completed runs replace that progress state with final measurement,
model-inspection, and training-summary records.

Publish the local dirty state as a commit:

```bash
leibniz results publish
```

Push only when you want to update the public Hugging Face repository:

```bash
leibniz results publish --push --repo owner/leibniz-results
```

To inspect another publication checkout locally, import its publication bundle
documents into a separate result root and materialize console views:

```bash
leibniz results import --source path/to/checkout --results-root imported-results
leibniz results materialize --results-root imported-results
```

## License

This repository is dedicated to the public domain under CC0-1.0. See
[LICENSE](LICENSE).
