# Contributing

Leibniz is built through small, reviewable pull requests. The goal is a public
protocol with clear artifacts and rigorous design boundaries.

## Pull Requests

- Every change enters by pull request.
- Do not push directly to `main`.
- Make changes on a topic branch that is used to open the pull request.
- Keep each pull request modular: one coherent component, policy, or artifact
  boundary at a time.
- Documentation-only pull requests do not need tests.
- Every non-documentation pull request must add or update tests for the
  contract it changes.
- Avoid compatibility layers for interfaces that are intentionally being
  removed or redesigned.
- Redesign is allowed and expected when it creates a cleaner protocol boundary.

## Contribution Terms

This repository is dedicated to the public domain under CC0-1.0. By submitting
a pull request, patch, or other contribution to this repository, you agree that,
if accepted, your contribution will be released under the same CC0-1.0 public
domain dedication as the rest of the project. Do not submit contributions that
you do not have the right to dedicate under those terms.

## Development Environment

The package declares support for Python `>=3.11`. The repository-local
Miniforge environment is the expected development environment and currently
uses Python 3.12:

```bash
bash scripts/setup_environment.sh
source scripts/activate_environment.sh
```

Prefer the activated environment over ad hoc `PYTHONPATH` changes. The package
is installed editable by the environment setup. `environment.yml` declares
CUDA-enabled PyTorch on Linux and CPU/Mac PyTorch on non-Linux hosts using
conda selectors. CI uses the same environment specification and
`pyproject.toml` inputs.

## Agent Guidance

Automated coding agents should read `README.md` and `CONTRIBUTING.md` before
editing, inspect nearby code and tests before choosing an implementation, keep
changes scoped to the requested boundary, and preserve unrelated worktree
changes. Agents should run the narrowest meaningful checks while iterating,
report any skipped validation, and avoid committing local runtime state such as
`results/`, caches, checkpoints, registries, or queues.

## Required Pull Request Explanation

Each nontrivial pull request should explain:

- Purpose: what protocol or implementation capability it adds.
- Boundary: what contract the change establishes.
- Public surface: modules, schemas, commands, identifiers, or durable files.
- Dependencies: earlier pull requests or design decisions it relies on.
- Tests: what was tested and what contract those tests prove.
- Rationale: why this should exist in the repository now.
- Design review: which design choices were considered, kept, changed, or
  rejected.

## Public Surface

Treat these as public surface unless the pull request states otherwise:

- CLI commands, options, exit behavior, and generated output formats.
- Importable modules, public classes, functions, and typed constants under
  `src/leibniz/`.
- Protocol artifact schemas, durable document formats, semantic identifiers,
  and versioned artifact names.
- Files intended to be read by other tools, repositories, result flows, or
  benchmark runners.
- Console data contracts and generated data consumed by the embedded web
  console.
- Repository policy checks and the set of files they allow or reject.

When changing public surface, state whether the change is additive, breaking,
or a pre-`1.0.0` redesign, and update tests for the contract that callers or
artifact consumers depend on.

## Review Scope

Pull requests should be small and explicit enough that reviewers can summarize:

- what the pull request does,
- why it exists,
- what boundary it owns,
- what main design choice it makes,
- and what tests prove.

If a pull request cannot be summarized this way, it is too large or too
unclear. Reduce the scope, split the work, or rewrite the explanation before
merging.

## Scientific Rigor

"Works" is not enough. Public code should be mathematically coherent,
logically explicit, reproducible, reviewable, and designed at the standard
expected from strong computational scientists.

Implementation pull requests should make clear:

- the mathematical objects being represented,
- their inputs, outputs, invariants, and composition behavior,
- which records are raw evidence and which are derived interpretations,
- which checks are exact and which are approximations,
- and why generic protocol layers are free of domain-specific shortcuts.

Tests should check semantic laws, artifact contracts, and workflows. They
should not be a parallel implementation of the feature.

## Compatibility Before 1.0.0

Until a release process exists and the first public release is issued, there is
no API or artifact compatibility promise. Breaking changes are expected.

During this period:

- Keep durable semantic identifiers below `1.0.0`.
- State breaking changes clearly in pull request descriptions.
- Bump pre-`1.0.0` identifiers when doing so helps reviewers distinguish
  incompatible public declarations.
- Prefer a cleaner protocol boundary over preserving a temporary interface.

## Tests And CI

The required GitHub Actions gate should be chosen deliberately as public
surfaces are introduced. Expected checks include:

- unit and contract tests,
- package build and import smoke tests,
- static formatting and linting checks once tool choices are fixed,
- type checks for stable typed surfaces,
- repository policy checks for runtime state and artifact boundaries,
- documentation checks once public docs become substantial.

Heavy benchmark training, GPU jobs, and network-dependent result repository tests
should not be required pull-request checks. They belong in scheduled or
manually triggered workflows with explicit resource expectations.

Use validation tiers deliberately:

- While iterating, run targeted tests or checks that exercise the changed
  contract.
- Before review, run the full local check set from `README.md` for
  non-documentation changes unless the pull request explains a skipped check.
- For expensive benchmark, GPU, or result repository workflows, document
  the manual validation path instead of making it a routine pull-request gate.

## Dependency Discipline

Core protocol modules should keep dependencies minimal. Tensor and neural
network concepts should be Leibniz definitions first. Backend frameworks such
as PyTorch may be used later behind thin optional adapter layers, but they
should not define the protocol's tensor, model, parameter, or probability
measure concepts.

## Generated And Local Files

Do not commit local outputs unless they have deterministic producers and a clear
review path. In particular, keep these out of Git:

- `results/`
- measurement records produced by local runs,
- model checkpoints,
- caches,
- local registries,
- work queues.
