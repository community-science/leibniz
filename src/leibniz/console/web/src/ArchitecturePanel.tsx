import { renderToString } from 'katex';
import 'katex/dist/katex.min.css';
import { useEffect, useRef, useState } from 'react';

type RungKind = 'add' | 'code' | 'horizon';

type Rung = {
  level: number;
  kind: RungKind;
  tag: string;
  text: string;
};

type NodeStatus = 'implemented' | 'direction' | 'mixed';

type Verdict = {
  tone: 'ok' | 'partial' | 'open';
  text: string;
};

type ConceptNode = {
  id: string;
  gist: string;
  meta: string;
  status: NodeStatus;
  anchor: string;
  verify?: { module: string; symbols: string[] };
  step?: string;
  rungs: Rung[];
  verdict: Verdict;
};

type RootId = 'U' | 'Q' | 'R';

type RootTree = {
  id: RootId;
  name: string;
  ask: string;
  nodes: ConceptNode[];
};

type CouplingEdge = {
  id: string;
  roots: RootId[];
  title: string;
  warn?: boolean;
  body: string;
};

type RoadmapStep = {
  id: string;
  title: string;
  outcome: string;
};

const PRECISION_LEVELS: { value: number; label: string }[] = [
  { value: 0, label: 'gist' },
  { value: 1, label: '+1' },
  { value: 2, label: '+2' },
  { value: 3, label: '+3' },
  { value: 4, label: 'all → code' },
];

const ROOTS: RootTree[] = [
  {
    id: 'U',
    name: 'Universe',
    ask: 'what exists',
    nodes: [
      {
        id: 'U-states',
        gist: 'A universe is a space of states on a domain, governed by a law.',
        meta: 'An evolving field on a mesh, a board game’s positions, a labeled glyph.',
        status: 'implemented',
        anchor: 'state_space.py: StateSpaceAmbient, StateSpaceAxis',
        verify: { module: 'leibniz.state_space', symbols: ['StateSpaceAmbient', 'StateSpaceAxis'] },
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'ambient vs chart',
            text: 'The states live in an ambient field space. A benchmark reaches into it by turning a handful of measured knobs, its chart axes — a generator $g: \\Theta \\to X$.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'charts are not the semantics',
            text: 'The chart is only a parameterization. The meaning lives in the ambient space, so whatever you count there does not depend on how the generator happens to be written.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py: StateSpaceAmbient, StateSpaceAxis, generator surface',
          },
        ],
        verdict: { tone: 'ok', text: 'Specified in code.' },
      },
      {
        id: 'U-law',
        gist: 'A universe either has a law, or only a convention.',
        meta: 'A field equation or a game’s rules are laws; a glyph’s label is just a convention.',
        status: 'direction',
        anchor: 'target contract: law status on the universe',
        step: 'signature',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'intrinsic vs extrinsic',
            text: 'With a law, correctness is intrinsic. With only a convention — an agreed-on label, no governing law — it is extrinsic. Predicting that label from data is honest work scored on the data alone (the $h$ part); what is pre-scientific, and forks off, is treating a match against a stored label as if it were the law\'s residual — a convention masquerading as a law.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one half of grounding',
            text: 'People usually bundle this with a second question under one word, “grounding.” It is only the first half: is there a law at all? The other half, verifier or convergence, belongs to Refinement (see the grounding edge).',
          },
        ],
        verdict: { tone: 'partial', text: 'Grounding splits between Universe and Refinement; see the edge.' },
      },
      {
        id: 'U-levels',
        gist: 'Three levels: reality, law, tasks.',
        meta: 'Experiment tests the law; it never defines a task’s answer.',
        status: 'direction',
        anchor: 'multi-scale universe records',
        step: 'discovery',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'where experiment acts',
            text: 'Experiment sits between reality and the law and asks whether the theory matches the world. Scoring sits between a task and the law.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'known vs unknown law',
            text: 'When the law is known, you put it to work. When it is not, you have to discover it from experiments that intervene in new ways. A virtual cell is the hard case: a whole tower of scales.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'fitting is not theory',
            text: 'Fit enough data and you get a predictor with no theory inside it. In a tower of scales, the effective theory at each level still has to agree with the converged law underneath.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'known part plus unknown part',
            text: 'Most real problems are $f = g + h$: a part $g$ whose law we know, scored by the law\'s residual, and a part $h$ we have only data for, scored by how completely a model accounts for that data. Discovering the law of an $h$ converts it into a $g$ — its score moving from data-only to law-grounded. That migration is the measurable trace of a field becoming a science.',
          },
        ],
        verdict: { tone: 'open', text: 'The $g + h$ split orients discovery, and a discovered law moves a part from data-only to law-grounded; the single quantity that puts the law-residual and the data-only score on one ledger, once it is no longer a bit count, is unresolved, and multi-scale universes still need to be specified.' },
      },
    ],
  },
  {
    id: 'Q',
    name: 'Query',
    ask: 'what you ask',
    nodes: [
      {
        id: 'Q-task',
        gist: 'A task is a question you put to a universe.',
        meta: 'One universe holds many tasks; they do not split neatly into classification versus prediction.',
        status: 'direction',
        anchor: 'target contract: task signature',
        step: 'signature',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the form',
            text: 'Three parts: how you sample the universe (access), how the input fixes the answer (binding), and what shape the answer takes (target).',
          },
          {
            level: 1,
            kind: 'add',
            tag: 'benchmarks',
            text: 'A finite readout task can let the input vary while keeping the answer fixed by the benchmark label. A field prediction task asks for $f(t)$ itself, so the input field and predicted future field both scale with spatial resolution.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'Is a query really its own thing, or just a slice of the universe you have pointed at? If it is the latter, there are two roots here, not three.',
          },
        ],
        verdict: { tone: 'open', text: 'Whether a query stands on its own is unsettled.' },
      },
      {
        id: 'Q-binding',
        gist: 'Binding: evolution, equilibrium, or inverse.',
        meta: 'Step forward in time · settle into a state · work back to a cause.',
        status: 'implemented',
        anchor: 'target contract: binding relation',
        verify: { module: 'leibniz.target_contracts', symbols: ['TargetContract'] },
        step: 'signature',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'examples',
            text: 'Evolution moves a field forward in time. Equilibrium is the state a system settles into; a folded protein, for instance, is an energy minimum, not a point on a trajectory. Inverse works backward from observations to a cause, which is usually ill-posed.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'relation to the law',
            text: 'The law decides which bindings are even possible. The field-evolution benchmark exercises evolution; the inverse-renderer benchmark exercises inverse binding by asking a submission to infer a latent cause, including pose and graded deformation, from an observed renderer output.',
          },
        ],
        verdict: { tone: 'partial', text: 'Evolution and inverse are live benchmark bindings; equilibrium remains a design direction.' },
      },
      {
        id: 'Q-target',
        gist: 'Target: a state, or a readout.',
        meta: 'A whole state (a field, a position), or a readout off it (a label, a number, a decision).',
        status: 'direction',
        anchor: 'target contract: target shape',
        step: 'signature',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'independent of grounding',
            text: 'What you ask for and how you check it are separate choices. A state-valued task or a readout task can each be checked by a verifier or by convergence. Folding the two together into “classification versus prediction” hides that.',
          },
        ],
        verdict: { tone: 'ok', text: 'An independent axis.' },
      },
    ],
  },
  {
    id: 'R',
    name: 'Refinement',
    ask: 'how you score',
    nodes: [
      {
        id: 'R-correctness',
        gist: 'Correctness is how completely a candidate satisfies the law.',
        meta: 'The score is the law\'s residual, integrated over initial conditions, space, and forward time, for a law whose solution is unique.',
        status: 'direction',
        anchor: 'score: the law residual over the sampled domain',
        step: 'ladder',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'residual',
            text: 'The score is the law\'s residual: put the candidate trajectory into the equation and measure how far it is from satisfying it, integrated over the initial conditions, over a region of space, and forward in time, anchored to each initial condition. It is the measured residual itself, earned by satisfying the equation, and continuous — the score scales with how nearly the candidate satisfies the law.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'uniqueness',
            text: 'The residual is the score for a law whose solution is unique; then a small initial-condition-anchored residual is the correct prediction. A complete law includes whatever makes its solution unique: for shock-forming conservation laws like Euler or shallow water that is the entropy/admissibility condition — the arrow of time — which selects the single physical solution and is part of stating the law, equivalently its viscous parent. A self-regularized field law is already unique on its own.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'law-induced metric',
            text: 'The residual and the distance to a solution are measured in the metric the law itself induces — its entropy, its energy or Lyapunov functional, a renderer\'s Jacobian — derived from the equation rather than declared. Forward in time the residual accumulates as the law amplifies it about the candidate\'s own trajectory: $L^2$ for a dissipative law, the entropy metric for a conservation law, and for a static inverse map the renderer-Jacobian conditioning at the submitted latent. The derived functional is itself checked, not trusted.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'admissible tasks',
            text: 'A task earns its place when the forward law hands you the answer as a byproduct — sample a cause, run the law, and you hold the answer — so minting an instance is cheap and the difficulty lives in the inverse. Where the answer is instead an adversarial game tree, minting an instance costs as much as solving it and there is no cheap ground, so such games sit on the cost axis as the search for a cheaper algorithm, not here.',
          },
        ],
        verdict: { tone: 'partial', text: 'This is the score we believe in; the code still computes it as certified bits behind a refusal gate, and the near-term work lets it read as the residual directly. A generic learned model does not cheaply satisfy a known law to a meaningful residual — that is expected, and it is why the data-only side exists.' },
      },
      {
        id: 'R-territory',
        gist: 'One operator, many territories.',
        meta: 'Correctness, the score hierarchy, and the bootstrap are one refinement pointed at different territories.',
        status: 'implemented',
        anchor: 'partition_score.py: adversarial partition competence integral',
        verify: { module: 'leibniz.partition_score', symbols: ['PartitionScore', 'adversarial_partition_competence_integral'] },
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'territories',
            text: 'Refinement always runs over some territory. Refine resolution and you get correctness. Refine the problem partition and you get the score hierarchy. Refine over operators across time and you get the bootstrap. Refine over scale or intervention and you get multi-scale work and discovery.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one refinement',
            text: 'So those three are one refinement pointed at different territories, not three separate ideas.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'partition tree',
            text: 'On the problem partition, the score is a tree of regions. Each node is a state-space region, each leaf carries the competence measured from samples that landed there, and the reported Score is the measure-weighted competence integral $\\sum_r \\mu(r)\\,c(r)$ over the leaves. Here $\\mu(r)$ is the leaf\'s share of the territory measure, so independent territories add; the normalized mean competence is retained as a capability view.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'adversarial refinement',
            text: 'The partition is refined by the region grammar itself: split along an available axis or stratum, choose the candidate with the largest between-child competence disparity, and keep it only when that disparity exceeds the measured sampling noise. There is no declared floor; an unmeasurable split has unbounded noise, and more samples shrink the noise that determines the finest resolvable scale.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'convergent value',
            text: 'The record carries the refinement ladder: each depth contracts the current leaves back to the same extensive competence integral and records the movement from the previous rung. The reported value is the converged leaf contraction with its propagated sampling uncertainty, not a score stopped by a hand-tuned minimum region size.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'capability map',
            text: 'The console renders the same tree as a capability map, coloring leaves by competence and showing the refinement ladder next to the single score. Failure pockets therefore remain visible even when their measure-weighted contribution is small.',
          },
          {
            level: 3,
            kind: 'code',
            tag: 'tree',
            text: 'partition_score.py: fixed_partition_competence_integral, adversarial_partition_competence_integral, PartitionScore; local_results.py and the console result-view transport expose the capability map.',
          },
          {
            level: 3,
            kind: 'horizon',
            tag: 'deferred',
            text: 'Sampling is still dense and post-hoc. Adaptive resampling toward suspicious splits is a follow-up diagnostic path, not part of the scorer in code today.',
          },
        ],
        verdict: { tone: 'partial', text: 'The query-space partition tree — measure-weighted integral, convergence ladder, capability map — is in code, realizing the score-hierarchy territory; the bootstrap and multi-scale territories (and adaptive sampling) are still ahead, so the one-mechanism claim is shown on one territory, not all.' },
      },
      {
        id: 'R-ratchet',
        gist: 'Trusted ground is a ratchet.',
        meta: 'Truth is a limit nobody reaches; the field just tightens toward it.',
        status: 'direction',
        anchor: 'trusted-ground persistence',
        step: 'bootstrap',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'how it grows',
            text: 'A model earns new ground by settling where earlier ones came apart, while still agreeing with them where they overlap. That overlap check is what stops a model from confidently agreeing with itself all the way to a wrong answer.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'rungs are operators',
            text: 'The rungs are themselves operators: solvers, earlier submissions, and in discovery settings, experiments. When the law is unknown, agreeing on the overlap also means staying consistent with the converged sub-laws at smaller scales.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'who promotes a law',
            text: 'A proposed law joins the trusted ground — an $h$ becoming a $g$ — by meeting the same criteria: it converges under refinement, agrees with established ground on the overlap, and its derived functional self-verifies. Anyone can submit one, and the criteria promote it, not the maintainer. The field moves the boundary, not whoever keeps the repository.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'two ratchets',
            text: 'Two ratchets turn at once: trusted ground creeps toward truth, and the cost measure, with its description language, creeps toward the ideal algorithmic one.',
          },
        ],
        verdict: { tone: 'partial', text: 'Carrying trusted ground across submissions is not built yet.' },
      },
      {
        id: 'R-resolution',
        gist: 'Two states are distinct only if you can tell them apart at the declared resolution.',
        meta: 'The resolution and region grammar the metric and the partition are built on.',
        status: 'implemented',
        anchor: 'state_space.py: Distinguishability, RegionFiltration',
        verify: { module: 'leibniz.state_space', symbols: ['Distinguishability', 'RegionFiltration'] },
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'resolution decides distinctness',
            text: 'Two states count as different only when you can tell them apart at the declared resolution, measured in the metric the law induces. Resolution decides what is distinguishable, not the chart you happened to write.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'ambient, not chart',
            text: 'Distinguishability is taken in the ambient space under the law-induced metric, not the chart, so it does not change if you reparameterize. In that metric a smooth field and a shock are both finite — a Fourier chart only made the shock look complex — and an inverse renderer separates identity from continuous nuisance.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'region grammar',
            text: 'Regions are finite disjoint unions of products of per-axis regions; qualitative labels are strata — typed annotations, never axes. This is the grammar the partition refines over.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py: Distinguishability, RegionFiltration',
          },
        ],
        verdict: { tone: 'ok', text: 'Specified in code.' },
      },
      {
        id: 'R-ledger',
        gist: 'One ledger: what a candidate accounts for, set against its cost.',
        meta: 'The currency is the residual; cost is the second axis; a common unit across laws is open.',
        status: 'direction',
        anchor: 'score: law-satisfaction against cost',
        step: 'cost',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'two axes',
            text: 'The score has two axes: how completely a candidate satisfies the law — its residual, where there is a law ($g$) — or how completely it accounts for the data where there is not ($h$), set against the energy cost of computing it. Value earned, cost paid.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the currency is the residual',
            text: 'The currency is the residual itself, earned by satisfying the equation, not a count of bits laid over it. Distance and residual are read in the law-induced metric, and the cost is energy priced from machine-independent counts under a declared profile.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'g and h on one ledger',
            text: 'Where there is a law the score is the law-residual ($g$); where there is not it is how completely a model accounts for the data ($h$) — the same kind of quantity, law-satisfaction versus data-fit, with $h$ a lower bound that tightens toward $g$ as the law is discovered. Moving the $g$/$h$ boundary moves a part from data-only to law-grounded for the same content; it manufactures nothing, so there is nothing to gain by elaborating a known law to swallow data it does not generalize.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'predict a distribution',
            text: 'A submission earns its score by predicting a distribution over the observable and being scored on what it resolves, not by emitting a point estimate or a hard label. So the line that matters is probabilistic prediction versus point output, not classification versus reconstruction. Only a stored answer-key forks off.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'Two things are open once the unit is no longer a bit count: whether a single quantity can put the law-residual and the data-only score on one ledger across different laws, and whether value and cost collapse into one number or stay two. Each score still lives in its own law\'s terms.',
          },
        ],
        verdict: { tone: 'open', text: 'The currency is the residual; the single cross-law unit, and whether value and cost are one number, are unresolved.' },
      },
      {
        id: 'R-cost',
        gist: 'Cost is declared roofline energy from invariant program counts.',
        meta: 'Measured, not a description-length penalty. Parsimony is pressured by low cost, not minimized directly.',
        status: 'implemented',
        anchor: 'cost metrology: DeviceCostProfile + bytes resident + energy breakdown',
        verify: { module: 'leibniz.cost_metrology', symbols: ['DeviceCostProfile', 'EnergyCostBreakdown', 'price_cost_measurement_energy'] },
        step: 'cost',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'measured, not penalized',
            text: 'Cost is a single priced quantity from the submitted program, not the machine it happened to run on: invariant counts for op class, dtype, bytes moved, and bytes resident are priced by a declared reference device profile. Parsimony is emergent, not minimized — a table-heavy memorizer pays when held tables enter the operation footprint, and a simple but expensive brute force is still excluded by cost; what survives — a model that generalizes at low energy — is the parsimonious explanation.',
          },
          {
            level: 2,
            kind: 'code',
            tag: 'roofline profile',
            text: 'cost_metrology.py declares DeviceCostProfile records for compute energy by (op class, dtype), bytes moved, and bytes resident. The bundled registry supplies a default reference profile plus alternate device profiles; changing profiles re-prices the same counts without changing the counts themselves.',
          },
          {
            level: 3,
            kind: 'code',
            tag: 'residency',
            text: 'bytes_resident is a peak per-operation input-plus-output tensor footprint from the operation stream. It charges large held operands and working-set spikes by $\\gamma$, but it is not a true held-buffer set: multi-buffer models and sharded table access can be undercounted. The time-integrated resident × duration model and trace identity needed for exact held-set residency remain flagged future refinements, not faked in v1.',
          },
        ],
        verdict: { tone: 'ok', text: 'Energy under a declared roofline profile is now the cost denominator; operation count remains as a diagnostic. Description length is rejected as an objective — parsimony is emergent.' },
      },
    ],
  },
];

const EDGES: CouplingEdge[] = [
  {
    id: 'edge-distinguishability',
    roots: ['U', 'R'],
    title: 'Distinguishability',
    body: 'States come from the universe; what makes “how far from the law” well-defined is the metric the residual is measured in. The metric $(d, \\varepsilon)$ is the handoff — the geometry Refinement reads the residual against. There is no distance, and no score, without it.',
  },
  {
    id: 'edge-grounding',
    roots: ['U', 'R'],
    title: 'Grounding',
    warn: true,
    body: 'Grounding is really two questions wearing one name: does the universe have a law (Universe), and is an answer checked by an exact verifier or by convergence (Refinement)? So a contract can just record whether there is a law and let the verifier-or-convergence part fall out of refinement, instead of declaring “grounding” outright. And where there is no law yet, correctness does not vanish — it becomes a measure of how completely a model accounts for the real data, just data-only: a lower bound nobody can yet prove optimal, which tightens as the law is discovered. Known-law and no-law are the two ends of one dial — law-grounded versus data-only — so the same ledger measures both, and how useful a known law is shows up as the share of the total it can ground, the law’s g-coverage.',
  },
  {
    id: 'edge-score',
    roots: ['U', 'R'],
    title: 'The score',
    body: 'Competence is how completely a candidate satisfies the law: the residual in the law-induced metric, to the resolution Refinement reaches, integrated over the sampled state-space territory from the Universe. The partition tree makes that handoff explicit: leaves are regions, color is competence, and the headline number is their measure-weighted competence contraction.',
  },
];

const STEPS: RoadmapStep[] = [
  {
    id: 'signature',
    title: 'Task signature in the contract grammar',
    outcome:
      'Make (access, binding, target) and a law-status real fields in the contract, then rebuild the existing benchmarks on top of them. Verifier scoring falls out as the zero-gap case of convergence.',
  },
  {
    id: 'ladder',
    title: 'Refinement-ladder records and convergence-grounded scoring',
    outcome:
      'Emit convergence diagnostics for the ladder and score field prediction as the residual integrated over the domain in the law-induced metric, with the predictability boundary emerging where the law stops determining the future. The boundary-climbing test shows the score rising with submitted capability and strong submissions reaching a deep boundary. The binary gate and its parameters are retired.',
  },
  {
    id: 'cost',
    title: 'Algorithmic cost',
    outcome:
      'Price invariant program counts as declared roofline energy: op class × dtype compute, bytes moved, and bytes resident under the default reference profile. Keep abstract operation count as a diagnostic, not the denominator. There is no description-length term — parsimony stays emergent. The test: two models with identical predictions but different resident footprints score differently.',
  },
  {
    id: 'hierarchy',
    title: 'Hierarchical query-space scoring (implemented core)',
    outcome:
      'The recursive partition scorer, disparity-above-noise refinement, convergence ladder, and console capability map are now in code. Adaptive sampling toward suspicious splits remains a follow-up.',
  },
  {
    id: 'bootstrap',
    title: 'Trusted-ground persistence',
    outcome: 'Carry convergent regions forward and establish newly reached ground by the agreement-on-overlap check.',
  },
  {
    id: 'lawmetric',
    title: 'Law-induced metric from typed equations',
    outcome:
      'Recognize the structural type of a benchmark equation from its typed symbols, and derive the stability functional it implies — hence the one metric the residual is measured in — instead of declaring it. How the residual accumulates forward (verifier, amplification, or functional dissipation) falls out of the type, and the derived functional is itself checked.',
  },
  {
    id: 'reexpress',
    title: 'Re-express the toy field benchmark',
    outcome:
      'Move the existing field benchmark onto the new instrument, and write its tests against the instrument itself, not a stand-in metric.',
  },
  {
    id: 'discovery',
    title: 'Readout and unknown-law universes (reserved)',
    outcome:
      'A readout-valued, convergence-grounded task, plus the $g + h$ ledger: a probabilistic submission scored on how completely it accounts for the data-only part $h$ (no point-estimate or reference fallback), aggregated with the law-residual $g$ through the partition tree, with a law promoted from $h$ into $g$ by the trusted-ground criteria rather than by the maintainer, and the scaffolding for unknown-law and multi-scale universes. Parked until the consolidated residual spine is solid.',
  },
];

const STATUS_LABEL: Record<NodeStatus, string> = {
  implemented: 'implemented',
  direction: 'design direction',
  mixed: 'mixed',
};

// Render a string with inline LaTeX spans delimited by `$...$`; prose passes
// through untouched. Bad expressions render as KaTeX error markup rather than
// throwing, so a typo can never crash the panel.
function MathText({ children }: { children: string }) {
  const segments = children.split(/(\$[^$]+\$)/g);
  return (
    <>
      {segments.map((segment, index) => {
        if (segment.length >= 2 && segment.startsWith('$') && segment.endsWith('$')) {
          return (
            <span
              className="architecture-math"
              key={index}
              dangerouslySetInnerHTML={{
                __html: renderToString(segment.slice(1, -1), { throwOnError: false }),
              }}
            />
          );
        }
        return <span key={index}>{segment}</span>;
      })}
    </>
  );
}

export function ArchitecturePanel() {
  const [maxLevel, setMaxLevel] = useState(4);
  const [collapsedNodes, setCollapsedNodes] = useState<Set<string>>(() => new Set());
  const [highlightRoot, setHighlightRoot] = useState<RootId | null>(null);
  const rootRefs = useRef<Partial<Record<RootId, HTMLElement | null>>>({});

  const annotatedNodes = ROOTS.flatMap((root) => root.nodes.map((node) => ({ ...node, rootId: root.id })));
  const totalCount = annotatedNodes.length;
  const implementedCount = annotatedNodes.filter((node) => node.step === undefined).length;

  useEffect(() => {
    if (highlightRoot === null) {
      return;
    }
    const timer = window.setTimeout(() => setHighlightRoot(null), 1500);
    return () => window.clearTimeout(timer);
  }, [highlightRoot]);

  const selectLevel = (value: number) => {
    setMaxLevel(value);
    setCollapsedNodes(new Set());
  };

  const toggleNode = (id: string) => {
    setCollapsedNodes((previous) => {
      const next = new Set(previous);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const jumpToRoot = (id: RootId) => {
    const element = rootRefs.current[id];
    if (element) {
      element.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
    setHighlightRoot(null);
    window.requestAnimationFrame(() => setHighlightRoot(id));
  };

  return (
    <div className="architecture-panel">
      <header className="architecture-intro">
        <h2>Architecture &mdash; adaptive precision tree</h2>
        <p>
          Every concept starts as a one-line gist; open it to go deeper a refinement at a time, until you reach the
          code that backs it (<span className="architecture-tag code">&#x27c2;</span>) or a question that is still open
          (<span className="architecture-tag horizon">open</span>). There are three roots, and they do not overlap: the
          universe, what you ask of it, and how you score an answer. Everything else is built
          from those, and where two roots lean on each other it shows up below as an edge.
        </p>
        <p className="architecture-intro-provenance">
          A concept is tagged <span className="architecture-status implemented">implemented</span> when working code
          backs it, or <span className="architecture-status direction">design direction</span> when it is planned but
          not built, with the open questions noted inline. The roadmap at the bottom is the order in which the planned
          ones turn into code.
        </p>
        <div className="architecture-controls" role="group" aria-label="Precision level">
          <span className="architecture-controls-label">Show precision up to</span>
          {PRECISION_LEVELS.map((level) => (
            <button
              className={`architecture-level ${maxLevel === level.value ? 'active' : ''}`}
              key={level.value}
              onClick={() => selectLevel(level.value)}
              type="button"
              aria-pressed={maxLevel === level.value}
            >
              {level.label}
            </button>
          ))}
        </div>
      </header>

      <div className="architecture-forest">
        {ROOTS.map((root) => (
          <section
            className={`architecture-root ${highlightRoot === root.id ? 'highlight' : ''}`}
            data-root={root.id}
            key={root.id}
            ref={(element) => {
              rootRefs.current[root.id] = element;
            }}
          >
            <div className="architecture-root-cap">
              <span className="architecture-glyph">{root.id}</span>
              <span className="architecture-root-name">{root.name}</span>
              <span className="architecture-root-ask">{root.ask}</span>
            </div>
            {root.nodes.map((node) => {
              const isCollapsed = collapsedNodes.has(node.id);
              return (
                <div className={`architecture-node ${isCollapsed ? 'collapsed' : 'open'}`} key={node.id}>
                  <button
                    className="architecture-node-head"
                    onClick={() => toggleNode(node.id)}
                    type="button"
                    aria-expanded={!isCollapsed}
                  >
                    <span className="architecture-twist" aria-hidden="true">
                      &#x25b6;
                    </span>
                    <span className="architecture-gist">
                      <MathText>{node.gist}</MathText>
                      <span className="architecture-meta">
                        <MathText>{node.meta}</MathText>
                        <span className={`architecture-status ${node.status}`}>{STATUS_LABEL[node.status]}</span>
                      </span>
                    </span>
                  </button>
                  {isCollapsed ? null : (
                    <div className="architecture-rungs">
                      {node.rungs.map((rung, index) => {
                        if (rung.level > maxLevel) {
                          return null;
                        }
                        return (
                          <div className={`architecture-rung ${rung.kind}`} key={index}>
                            <span className={`architecture-tag ${rung.kind}`}>{rung.tag}</span>
                            <span className="architecture-rung-text">
                              <MathText>{rung.text}</MathText>
                            </span>
                          </div>
                        );
                      })}
                      <div className={`architecture-verdict ${node.verdict.tone}`}>
                        <MathText>{node.verdict.text}</MathText>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </section>
        ))}
      </div>

      <section className="architecture-edges" aria-label="Edges where the roots couple">
        <h3>Edges &mdash; where the roots couple</h3>
        <p className="architecture-edges-note">
          A few concepts do not live in any single root; they tie two or three together. Those are the edges.
        </p>
        {EDGES.map((edge) => (
          <div className={`architecture-edge ${edge.warn ? 'warn' : ''}`} key={edge.id}>
            <div className="architecture-edge-title">
              {edge.roots.map((rootId, index) => (
                <span key={rootId}>
                  {index > 0 ? <span className="architecture-edge-times"> &times; </span> : null}
                  <button
                    className="architecture-pill"
                    data-root={rootId}
                    onClick={() => jumpToRoot(rootId)}
                    type="button"
                  >
                    {rootId}
                  </button>
                </span>
              ))}
              <span className="architecture-edge-name">{edge.title}</span>
            </div>
            <p className="architecture-edge-body">
              <MathText>{edge.body}</MathText>
            </p>
          </div>
        ))}
      </section>

      <section className="architecture-roadmap" aria-label="Closing the delta">
        <h3>Closing the delta</h3>
        <p className="architecture-roadmap-note">
          {implementedCount} of {totalCount} concepts are in code today. The rest get there through the sequence below,
          one pull request per step: write the code it names, and flip its concepts to implemented right here.
          Contributing means editing this panel and the code behind it.
        </p>
        <ol className="architecture-steps">
          {STEPS.map((step, index) => {
            const closes = annotatedNodes.filter((node) => node.step === step.id);
            return (
              <li className="architecture-step" key={step.id}>
                <div className="architecture-step-head">
                  <span className="architecture-step-index">{index + 1}</span>
                  <span className="architecture-step-title">{step.title}</span>
                </div>
                <p className="architecture-step-outcome">
                  <MathText>{step.outcome}</MathText>
                </p>
                {closes.length > 0 ? (
                  <ul className="architecture-step-closes">
                    {closes.map((node) => (
                      <li className="architecture-closes-item" key={node.id}>
                        <button
                          className="architecture-closes-chip"
                          data-root={node.rootId}
                          onClick={() => jumpToRoot(node.rootId)}
                          type="button"
                        >
                          <span className="architecture-closes-glyph">{node.rootId}</span>
                          <MathText>{node.gist}</MathText>
                        </button>
                        <span className="architecture-closes-anchor">{node.anchor}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="architecture-step-infra">uses the instrument; does not close a new concept</div>
                )}
              </li>
            );
          })}
        </ol>
      </section>
    </div>
  );
}
