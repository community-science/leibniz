# Architecture: measurement, tasks, and scoring

This document is the durable description of how Leibniz measures scientific
work — how a benchmark declares a *universe*, poses *tasks* in it, defines what
a *correct* answer is, and *scores* a model against the cost of producing it. It
is the conceptual spine that the measured-region sequence (#309, #317, #335),
the cost metrology (#329), the solver/field-operator layer (#323), and the
field-prediction redesign (#338) all serve.

Two things this document is **not**. It is not the specification — *code, not
prose, is the specification* (README principle 2); where this document and the
code disagree, the code wins and this document is wrong. And it is not a
frozen design — the protocol is pre-1.0 and evolving. Each section is marked
with its status:

- **Implemented** — reflected in merged code and tests; the named PRs are its
  provenance.
- **Design direction** — agreed in a planning PR but not yet (fully) in code;
  the named PR carries its open questions.

Live design discussion and unresolved mechanical questions live in the planning
PRs, not here. This document records the *stable shape*; the PR sequence
records the *frontier*.

It is written in two registers throughout, following the repository's module
docstrings: an **informal** account for a reader with working multivariable
calculus, and a **formal** account for precise reference. Read Part I for the
picture and Part II for the definitions.

---

## Part I — Informal overview

### Universes, and the three levels above a task

A benchmark declares a **universe**: all the states some system can be in, on a
domain, together with a **law** relating them. A fluid universe is velocity
fields on a mesh under the Euler equations; a board-game universe is legal
positions under the rules of play; a glyph universe is rendered images.

A law is a *theory* — a human-made model — so there are three levels, not one:
**reality** (the world, reached only through experiment), **the law** (a theory
of it), and **the tasks** we pose inside the universe. Experiment sits between
reality and the law: it *validates the theory*. It does not define the answers
to tasks. Keeping these separate is load-bearing (Part I, "Correctness").

### States, distinguishability, and bits

Every observation a benchmark can show — a glyph image, a board position, a
field on a mesh — is a point in a space of possible observations. A benchmark
generates observations by turning a few **knobs**, which *chart* a structured
island of meaningful states inside that enormous space.

Two knob settings are the *same state* unless the observations they produce are
**distinguishable** at the benchmark's declared resolution. The **volume** of a
region is the number of genuinely distinguishable states in it, reported in
**bits** as `log2(volume)` — the number of yes/no questions needed to single one
out. Volumes multiply across independent knobs, so bits add. Refining the
resolution reveals more distinct states: walking up the bits axis is walking
down the energy cascade of a turbulent flow, or up the move-count of a game.

### Tasks: more kinds than "classify vs. predict"

A single universe supports many **tasks**, and the machine-learning instinct to
sort them into "classification" and "prediction" hides several independent
questions. A task is described by a **signature** with these facets:

- **Access** — what the solver is given and how the universe is sampled.
- **Binding relation** — how the input determines the answer: *evolution*
  (given a state and a time, step forward), *equilibrium* (given a
  specification, find the state it settles into — a folded protein structure is
  the minimum of an energy landscape, not a time-step), or *inverse* (given
  observations, infer the cause — the ill-posed cousin of running time
  backward).
- **Target shape** — a **state** of the universe, or a **readout** of it (a
  label, a scalar diagnostic, a decision).
- **Grounding** — what makes an answer *right*: the universe's own **law**
  (intrinsic), or **law-less convention** (extrinsic — a digit label has no
  governing equation; it is a pre-scientific perception task).

These are orthogonal. A glyph label is a readout, extrinsic. A game's best move
is state-valued and intrinsic, with an exact verifier. Field evolution is
state-valued and intrinsic, with *no* verifier. Structure prediction is
state-valued, equilibrium, and intrinsic — its law is the Schrödinger equation;
we are computationally defeated, not theory-ignorant, so it is *solve the law*,
not *fit the data*.

### Correctness: convergence, not an oracle

For an intrinsic task with no exact verifier, you can never write down "the true
state" to compare against — any simulator you would use to define it suffers the
same compounding discretization error your candidate does. Chaos is the extreme
of this universal fact, not a separate disease.

The way out is what careful computational science already does when it has no
exact answer: **refine and watch.** Ask for the answer at a coarse resolution,
then finer, then finer, and compare on the features they share. If it stops
moving — the sequence is **Cauchy** — the question had a determined answer (the
**Richardson limit**), with the gap between refinements as its error bar. If it
keeps moving, the answer was never determined by the information given, and *no*
model can be scored right there. From this single test fall, with nothing
declared by hand:

- The **resolution** is both the answer's detail (its bits) and the ruler that
  measures closeness — one knob, no separate tolerance.
- The **horizon** is just where refinement stops settling.
- **Directionality** falls out the same way: dissipative universes erase the
  detail that distinguishes their pasts, so the backward sequence never settles
  and the score correctly reads "unknowable"; reversible universes settle both
  ways. We never declare a universe chaotic, or one-way — the refinement
  behavior says so, per question.

An exact verifier (a game's rules) is the degenerate case where the gap is
zero: a discrete law is already at full resolution, the ladder is constant, and
the law hands the answer over directly.

The same "refine and watch" reaches past resolution. The *territory* you push
into can be finer resolution, finer **scale** (molecules → cells), or new
**interventions** (perturb the system in ways it was not shown). A theory is
trustworthy in a regime exactly where new experiments stop changing its
predictions there — Popper as a convergence property. So the instrument that
scores a fluid also scores a discovered theory of a cell; only the source of
the ladder's rungs changes (computation, or experiment).

### Experiment validates the theory; it does not define the answer

This is where models trained on experimental data go wrong. Fitting measured
structures builds a predictor with *no theory inside it*. When the law is
already known, the right object solves the law (intrinsic, by convergence) and
uses experiment to *validate the law*. The genuinely data-grounded frontier is
**theory discovery** — universes whose law we do not yet know, where the task is
to build an understandable, predictive theory from novel-intervention
experiment. A virtual model of a cell is the target: a tower of **scales**,
known law at the bottom, unknown effective theory at the top, experiment
validating throughout, and cross-scale consistency binding the levels.

### Scoring is a hierarchy, not a number

The universe of queries is vast and heterogeneous — laminar and turbulent,
near-attractor and transient. You can neither sample it uniformly nor honestly
compress it to a scalar. So a *second* "refine and watch" runs on the **partition
of the problem**: carve the query space coarsely, estimate competence per cell,
recurse where it warrants, and stop when the score estimate is stable — Cauchy
again, now in sampling. The result is a **tree** of competence over the problem
space; the scalar score is its integral, a lossy summary. Three forces make the
tree the honest object: feasibility, *where*-competent legibility, and
understandability (the minimal tree is the compressed description of the
competence field). The stopping rule must be robust to *adversarial*
subdivision — refine where it would most overturn the verdict — so a model
cannot hide failure in an unrefined cell.

### Cost: one axis, and why it is the point

Science is not just prediction; it is prediction in a form a human can
*understand* — a **theory**, a compressed account that predicts much from few
principles. A working black box is not yet science. We measure this on every
task, from the start, with a single principled cost: **algorithmic (Levin)
complexity** — the operator's **description length** plus the **log of its
operation count**. This is Occam's razor made computable: the shortest theory is
the ideal, the operation-count term is the price of being runnable. Dropping the
description-length term leaves plain compute — which is exactly why compute alone
is an imperfect proxy: it cannot tell a compact law-solver from a
billion-parameter interpolator that runs as fast.

This folds into the currency already in use — a score is already in **bits**, and
`log2` of a state count is a description length — so theory-length and compute
join the same ledger: **science is compression**, and the best operator
minimizes the total description length of validated reality. Two constraints
keep it real: the cost must be computable from the submitted program and
independent of the machine it ran on. So the operation-count term is the
protocol's machine-independent per-op cost model, and the description-length
term is a property of the program (its parameters and architecture), never of
wall-clock time.

### The frontier, and two ratchets

A model earns trust by predicting where prior models failed while agreeing where
they held, so the **trusted ground** ratchets outward, as physical measurement
standards do. The research prize is **validated bits of prediction per unit of
algorithmic cost** — depth of understanding per unit of theory-and-compute. Two
ratchets run, the same shape: trusted ground tightening toward truth, and the
cost proxy and its description-language tightening toward the single algorithmic
ideal.

---

## Part II — The formalism

### Levels and universe *(design direction: #338; ambient/region machinery implemented: #309, #317, #335)*

A universe `U = (Ω, X, L)`: a geometric domain `Ω`, an ambient state space `X`
of fields on `Ω`, and a declared law `L`. For physical universes a validation
relation binds `U` to reality `R` (experiment) on the `R↔U` edge; for formal
universes `L` is self-contained. Task scoring is internal to `U` (grounded in
`L`); experiment never acts on the `task↔U` edge.

### States, distinguishability, regions, measure *(implemented: #309, #317, #335)*

A benchmark declares `(Ω, X, d, ε, Θ, g)`: an ambient field space `X` over `Ω`,
a distinguishability `(d, ε)` (a metric and resolution, or exact
distinguishability), a chart space `Θ` of measured latent axes, and a generator
`g : Θ → X`. A **realized region** is `R = g(Θ_R)` for a charted `Θ_R ⊆ Θ`. Its
**volume** `μ(R)` is the `ε`-covering count of `R` under `d` — defined in
ambient field space, computed exactly through the chart, hence invariant to
reparameterization of `g`. Regions form the semiring of finite disjoint unions
of products of per-axis regions; bits are `log2 μ`, additive across independent
axes. Qualitative labels are **strata** — typed annotations on union
components, never axes (no monotone variation measure runs along them). Volume
may be **exact** or an interval **estimate** (`MeasureEstimate`), with the
estimated case bracketing `log2 μ`. Implementing records: `StateSpaceAmbient`,
`StateSpaceAxis`, `StateSpaceRegion`/`ProductRegion`, `Distinguishability`,
`MeasureEstimate`, `RegionFiltration`, `AccessibleSubspace`, `SamplingProtocol`.

### Task signature `(A, β, τ, γ)` *(design direction: #338)*

- `A` — **access**: how `U` is sampled and what the solver is given.
- `β` — **binding relation**: `evolution` (a semiflow `Φ : X × ℝ → X`),
  `equilibrium` (an `argmin`/stationarity map `spec → X`), or `inverse`
  (`observation → X`, generally ill-posed).
- `τ` — **target shape**: `state-valued` (answer in `X`) or `readout-valued` (a
  measurable functional `X → Y`).
- `γ` — **grounding**: `intrinsic` (the law `L` decides correctness) or
  `extrinsic` (law-less convention — pre-scientific).

The canonical `state-valued`/`evolution`/`intrinsic` task asks for the
universe's evolution semiflow `Φ`, queried across all of `X` and all `T`; a
submission provides an approximant `Φ̂` at a declared resolution. No
reversibility or horizon is declared; both are derived.

### Correctness by refinement *(design direction: #338)*

For a query at refinement level `ℓ` (resolution, scale, or
intervention-regime), build a ladder `r_k = R_ℓ Φ̂(refine_k(input))` projected
onto shared coarse content, with gaps `g_k = ‖r_{k+1} − r_k‖`. The query is
**Cauchy-resolved** when `g_k → 0` within tolerance; the **answer** is the limit
`r_∞`, the **ruler** is `ε = g_∞`. Where `g_k` does not contract the answer is
undefined and no operator scores. Verifier-grounding is the degenerate `ε = 0`
(the law supplies `r_∞` directly). The Cauchy-resolved set is the validated
region; its boundary is the horizon. Directionality and irreversibility are
outputs of the test, never declared.

### Scoring *(competence density implemented: #335; convergence and hierarchy: design direction #338)*

Competence is the predictive mass `Φ̂` places within `ε` of `r_∞`; its density
`dν/dμ` (the Radon–Nikodym derivative of the demonstrated-competence measure
`ν(A)=∫_A c dμ`) is estimated by sampling the access measure. The **value** of a
query is its validated bits, `log2` of the distinguishable states pinned within
`ε`. The score is value per unit cost, integrated along the `log2`-bit
filtration and swept over the refinement territory.

The integral is not flat over a uniform measure on `X` but over a recursive
partition `𝒯` of the query space — a tree whose nodes are regions with their
covering measure, competence density, ladder status, and cost. `𝒯` is refined by
the same Cauchy criterion as resolution, on a second axis: subdivide a node
until its competence estimate is stable under further, *adversarial*,
subdivision (refine wherever subdivision would most change the verdict — the
problem-space analogue of agreement-on-overlap). The scalar score is the
measure-weighted contraction of `𝒯`; `𝒯` itself is the reported object. A
benchmark is in general a product of such hierarchies (resolution, partition,
horizon `T`, physical scale, family); a submission's claim is a subtree or
frontier of `𝒯`.

### Cost *(per-op count implemented: #329; algorithmic reframe: design direction #338)*

```text
  cost(Φ̂) = description_length(Φ̂) + log2( operation_count(Φ̂) )
```

`operation_count` is the protocol's machine-independent per-op cost model;
`description_length` is a codelength of the program (parameters + architecture)
against a declared description language. Both terms are computable from the
submitted program and invariant to the executing machine. Equivalently, the
single quantity to minimize is the total codelength of validated reality:
`description_length(Φ̂) + log2(ops) + (unpredicted residual)`. Value, parsimony,
and effort are three terms of one ledger.

### Bootstrap *(design direction: #338)*

Ladder rungs are operators — solvers, prior submissions, and (for
discovery/extrinsic universes) experiments. Trusted ground `G` is the union of
Cauchy-resolved regions of the best operators. A submission extends `G` by being
Cauchy outside `G` while agreeing within `ε` on the overlap (the anchor against
self-consistent nonsense); for unknown-law universes the overlap test also
includes cross-scale consistency with converged sub-laws. Two ratchets, the same
shape: trusted ground tightening toward truth, and the cost proxy /
description-language tightening toward the single algorithmic ideal.

---

## Part III — Provenance and status

| Concept | Status | PRs |
| --- | --- | --- |
| Measured state-space regions; volume as distinguishable-state count; bits axis; competence-density integral; strata | Implemented | #309 (#310–#315) |
| Region filtration; increment sampling; domain-growth charts | Implemented | #317 |
| Claimed regions; sampled covering measures; certification | Implemented | #335 |
| Machine-independent per-op cost metrology | Implemented | #329 |
| Solver programs; dimension-general field operators | Implemented | #323 |
| Toy field benchmark (K-S) — stand-in scoring, to be re-expressed | Implemented | #324 |
| Three levels; task signature; binding/target/grounding axes | Design direction | #338 |
| Oracle-free convergence correctness; ruler as convergence gap; horizon/directionality as outputs | Design direction | #338 |
| Algorithmic (Levin) cost; science-as-compression ledger | Design direction | #338 |
| Hierarchical query-space scoring; adversarial refinement | Design direction | #338 |
| Bootstrap / trusted-ground ratchet; theory-discovery and multi-scale universes | Design direction | #338 |

The design-direction rows carry open mechanical questions — what "Cauchy"
means operationally, how `description_length` is read from a program, the
adversarial stopping rule, the partition's origin, trusted-ground persistence —
which are tracked in the planning PRs, not resolved here. This document is
updated when a design direction lands as implemented code, or when the stable
shape itself changes.
