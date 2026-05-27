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

## Result Publication Workflow

Use a Hugging Face dataset repository as the local run-state checkout. The
checkout is ignored by this source repository, but it is its own Git repository:
benchmark runs write dirty state there, and publishing commits that checkout.
Pushing to Hugging Face is always explicit.

Create a public dataset repository and prepare `.runs` as the checkout:

```bash
export HF_TOKEN=...
leibniz results init-publication --repo owner/leibniz-results
```

Without a Hugging Face account, prepare the same local checkout and skip any
Hub API or push step:

```bash
leibniz results init-publication --local-only
```

Run benchmarks against that checkout:

```bash
leibniz benchmark shakedown \
  --benchmark-root src/leibniz/benchmarks/digits
```

Publish the local dirty state as a commit:

```bash
leibniz results publish
```

Push only when you want to update the public Hugging Face repository:

```bash
leibniz results publish --push
```

To inspect another publication checkout locally, import its publication bundle
documents into a separate run root and materialize console views:

```bash
leibniz results import --source path/to/checkout --runs-root .runs-imported
leibniz results materialize --runs-root .runs-imported
```

## License

This repository is dedicated to the public domain under CC0-1.0. See
[LICENSE](LICENSE).
