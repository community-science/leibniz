# Contributing

Leibniz is built through small, reviewable pull requests. The goal is a public
protocol with clear artifacts and rigorous design boundaries.

## Pull Requests

- Every change enters by pull request.
- Do not push directly to `main`.
- Keep each pull request modular: one coherent component, policy, or artifact
  boundary at a time.
- Documentation-only pull requests do not need tests.
- Every non-documentation pull request must add or update tests for the
  contract it changes.
- Avoid compatibility layers for interfaces that are intentionally being
  removed or redesigned.
- Redesign is allowed and expected when it creates a cleaner protocol boundary.

## Required Pull Request Explanation

Each nontrivial pull request should explain:

- Purpose: what protocol or implementation capability it adds.
- Boundary: what is included and what is explicitly not included.
- Artifact kind: Specification, Implementation, Evidence, or a combination.
- Public surface: modules, schemas, commands, identifiers, or durable files.
- Dependencies: earlier pull requests or design decisions it relies on.
- Tests: what was tested and what contract those tests prove.
- Rationale: why this should exist in the repository now.
- Design review: which design choices were considered, kept, changed, or
  rejected.

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

Heavy benchmark training, GPU jobs, and network-dependent federation tests
should not be required pull-request checks. They belong in scheduled or
manually triggered workflows with explicit resource expectations.

## Dependency Discipline

Core protocol modules should keep dependencies minimal. Tensor and neural
network concepts should be Leibniz definitions first. Backend frameworks such
as PyTorch may be used later behind thin optional adapter layers, but they
should not define the protocol's tensor, model, parameter, or probability
measure concepts.

## Runtime State

Do not commit runtime state. In particular, keep these out of Git:

- `.leibniz/`
- measurement records produced by local runs,
- model checkpoints,
- caches,
- local registries,
- work queues,
- generated artifacts without deterministic producers and verification paths.
