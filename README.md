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

## Artifact Kinds

Every artifact in this repository should be one of four semantic kinds:

- **Specification:** declarative, versioned descriptions of things the system
  knows how to interpret.
- **Implementation:** executable code that validates, interprets, renders,
  trains, evaluates, or materializes specifications.
- **Evidence:** tests, deterministic fixtures, reproducible generated artifacts,
  and inspection aids.
- **State:** mutable operator-owned outputs such as measurements, checkpoints,
  caches, registries, and work queues.

Specification, Implementation, and reproducible Evidence may belong in Git.
State does not. Runtime state should live in ignored `.leibniz/` directories or
external repositories.

## Contribution Model

All changes enter through pull requests. Non-documentation changes must include
tests that prove the semantic contract being introduced or changed. Pull
requests should be small enough that a reviewer can explain their purpose,
boundary, design choice, and tests before merge.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the working rules.

## License

This repository is dedicated to the public domain under CC0-1.0. See
[LICENSE](LICENSE).
