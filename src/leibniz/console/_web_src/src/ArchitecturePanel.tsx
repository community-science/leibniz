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

type RootId = 'U' | 'Q' | 'R' | 'D';

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
            text: 'With a law, correctness is intrinsic. With only a convention it is extrinsic: there is no governing law, just an agreed-on label. That is pre-scientific.',
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
        ],
        verdict: { tone: 'open', text: 'Discovery and multi-scale universes still need to be specified.' },
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
            text: 'Is a query really its own thing, or just a slice of the universe you have pointed at? If it is the latter, there are three roots here, not four.',
          },
        ],
        verdict: { tone: 'open', text: 'Whether a query stands on its own is unsettled.' },
      },
      {
        id: 'Q-binding',
        gist: 'Binding: evolution, equilibrium, or inverse.',
        meta: 'Step forward in time · settle into a state · work back to a cause.',
        status: 'direction',
        anchor: 'target contract: binding relation',
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
            text: 'The law decides which bindings are even possible. Picking one is up to you.',
          },
        ],
        verdict: { tone: 'partial', text: 'We have not pinned down the full list of bindings yet.' },
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
    ask: 'how you know',
    nodes: [
      {
        id: 'R-correctness',
        gist: 'Correctness is convergence to a law, not an oracle.',
        meta: 'Refine space and time, test the residual, and score the resolved bits.',
        status: 'implemented',
        anchor: 'benchmark_runner.py: operator-aware field competence and raw bit scoring',
        verify: { module: 'leibniz.benchmark_runner', symbols: ['_FieldTrainingCompetenceRequest', '_training_score_integral'] },
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the move',
            text: 'You cannot look up the exact answer, so here is the move: a field answer is correct where refinement makes it converge to a solution of the law. A hand-selected scoring parameter is a bug; the scored quantities must be measured from the ladder or derived from the law.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'space-time ladder',
            text: 'Compute the operator on nested space-time grids and compare on the common grid. For a field law, the residual is evaluated consistently in space and time. If its Richardson extrapolated limit is indistinguishable from zero within its own extrapolation uncertainty, the law gate holds.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'measured ruler',
            text: '$\\varepsilon$ is not declared and there is no gate. It is the residual-certified distance to a solution of the law: $\\varepsilon(t) = \\sum_{s<t} G(s\\to t)\\,\\lVert r(s)\\rVert\\,dt$, where $r$ is the law residual and $G$ is the leading operator-norm amplification of the law linearized about the submitted trajectory itself (oracle-free). Validated bits are continuous in $\\varepsilon$, and the predictability boundary emerges where the certified bits reach zero.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'the metric is the law geometry',
            text: 'The residual, the certified $\\varepsilon$, and the bit count all live in one metric: the geometry the law itself induces through its entropy or energy/Lyapunov functional, derived from the equation rather than declared. It is $L^2$ for a dissipative law like KS and the entropy metric for a conservation law; certification is amplification when no monotone functional exists and functional dissipation when one does. The derived functional is itself verified, not trusted.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'formal objects',
            text: 'Formally: query $\\hat\\Phi(x_h, t)$ across refined grids, restrict each rung to the common grid, estimate observed order and limit by Richardson, and credit only the time-prefix that stays inside the residual plus field-error gate.',
          },
          {
            level: 4,
            kind: 'horizon',
            tag: 'open',
            text: 'Open rungs remain: the IC distribution is provisional; the certified $\\varepsilon$ uses the leading-amplification estimator, the smooth and chaotic-dynamics instance of a general law-induced one (entropy stability for shock-forming conservation laws is the harder case); and a submitted program that climbs a deep boundary is still wanted. The retired-gate observed-order tolerance and rung-count parameters are gone.',
          },
        ],
        verdict: { tone: 'partial', text: 'Being rebuilt on continuous certified bits (#344): the binary gate is retired for a residual-certified distance in the law-induced metric, validated across five experiments; the general entropy-stability certification and a boundary-climbing program remain open.' },
      },
      {
        id: 'R-territory',
        gist: 'One operator, many territories.',
        meta: 'Correctness, the score hierarchy, and the bootstrap are all the same move.',
        status: 'direction',
        anchor: 'partition-tree scoring',
        step: 'hierarchy',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the parameter',
            text: 'Refinement always runs over some territory. Refine resolution and you get correctness. Refine the problem partition and you get the score hierarchy. Refine over operators across time and you get the bootstrap. Refine over scale or intervention and you get multi-scale work and discovery.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one mechanism',
            text: 'So those three are one mechanism pointed at different territories, not three separate ideas.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the partition tree',
            text: 'On the problem partition, the score is a tree of regions. You keep subdividing a region until its competence estimate holds up even against an adversary choosing where to split, so a model cannot tuck its failures into a cell nobody looked at. A claim is a subtree, and the single number is that tree weighted by measure.',
          },
          {
            level: 3,
            kind: 'horizon',
            tag: 'open',
            text: 'Whether all these territories are really one kind of axis, or several, is open.',
          },
        ],
        verdict: { tone: 'open', text: 'The shared mechanism is clear; whether its territories are uniform is not.' },
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
            level: 3,
            kind: 'add',
            tag: 'two ratchets',
            text: 'Two ratchets turn at once: trusted ground creeps toward truth, and the cost measure, with its description language, creeps toward the ideal algorithmic one.',
          },
        ],
        verdict: { tone: 'partial', text: 'Carrying trusted ground across submissions is not built yet.' },
      },
    ],
  },
  {
    id: 'D',
    name: 'Description length',
    ask: 'what you count',
    nodes: [
      {
        id: 'D-volume',
        gist: 'Volume counts distinguishable states; bits add up.',
        meta: 'The unit everything else is measured in.',
        status: 'implemented',
        anchor: 'state_space.py: Distinguishability, RegionFiltration',
        verify: { module: 'leibniz.state_space', symbols: ['Distinguishability', 'RegionFiltration'] },
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'defines “different”',
            text: 'Two states only count as different if you can tell them apart at the declared resolution. Bits $= \\log_2(\\text{count})$, so each doubling of variety is one more bit.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'locates it, justifies invariance',
            text: 'It is an $\\varepsilon$-covering count under the metric the law itself induces — the geometry of its entropy or energy/Lyapunov functional, derived from the equation rather than declared — taken in the ambient space rather than the chart, so it does not change if you reparameterize. In that metric a smooth field and a shock are both finite information (a Fourier chart only made the shock look complex), and the count is of the field evolution above persistence. Independent axes multiply, which is why their bits add.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the region grammar',
            text: 'Regions are finite disjoint unions of products of per-axis regions. Qualitative labels are strata: typed annotations, never axes. A volume is either exact or a bracketed estimate.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py: Distinguishability, log2 μ, RegionFiltration',
          },
        ],
        verdict: { tone: 'ok', text: 'Specified in code.' },
      },
      {
        id: 'D-ledger',
        gist: 'One ledger: value as credit, cost as debit.',
        meta: 'Same unit, opposite signs.',
        status: 'direction',
        anchor: 'score: validated bits per unit cost',
        step: 'cost',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the two entries',
            text: 'Credit is validated bits: the ambient $\\varepsilon$-entropy of the field evolution above persistence, resolved to the residual-certified $\\varepsilon$ (continuous, no gate), replacing the earlier grid-degree-of-freedom count $N_{dof}\\log_2(\\sigma/\\varepsilon_{field})$. Debit is the bits it took to describe and run the operator.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one quantity to minimize',
            text: 'The value side now has a concrete validated-bit quantity for field prediction. The cost side still needs the description-length term; today the runner can attach operation cost and integrate validated bits over the measured frontier.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'Whether value and cost really collapse into one number, or have to stay two, is open.',
          },
        ],
        verdict: { tone: 'open', text: 'Whether it is truly one number is not shown yet.' },
      },
      {
        id: 'D-cost',
        gist: 'Cost $= \\text{description length} + \\log_2(\\text{operations})$.',
        meta: 'Levin complexity: Occam’s razor you can actually compute. Drop the description-length term and you are left with plain compute.',
        status: 'mixed',
        anchor: 'cost metrology: operation count (in code) + description length',
        step: 'cost',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the two constraints',
            text: 'Both terms come from the submitted program, not the machine it ran on: log-operations from the per-op cost model, description length from the program’s own size.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'How you read description length off a program (compressed weights plus source, or a two-part code), and which reference language you measure it against, still needs defining.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'cost metrology: operation count is in code; the algorithmic reframe is a direction',
          },
        ],
        verdict: { tone: 'partial', text: 'Operation count is in code; description length still needs defining.' },
      },
    ],
  },
];

const EDGES: CouplingEdge[] = [
  {
    id: 'edge-distinguishability',
    roots: ['U', 'D'],
    title: 'Distinguishability',
    body: 'States come from the universe; counting them in bits is the currency’s job. The metric $(d, \\varepsilon)$ is the handoff, the thing that gives the currency something to count. You cannot count without a metric to count against.',
  },
  {
    id: 'edge-grounding',
    roots: ['U', 'R'],
    title: 'Grounding',
    warn: true,
    body: 'Grounding is really two questions wearing one name: does the universe have a law (Universe), and is an answer checked by an exact verifier or by convergence (Refinement)? So a contract can just record whether there is a law and let the verifier-or-convergence part fall out of refinement, instead of declaring “grounding” outright.',
  },
  {
    id: 'edge-score',
    roots: ['U', 'R', 'D'],
    title: 'The score (triple point)',
    body: 'Competence is the validated information a model resolves where the refinement gate holds: measured convergence error from Refinement, bits from Description length, integrated over the query space from Universe. That sum is one of the same adaptive trees. So the score is not a fourth root; it is where the other three meet, and it needs nothing else.',
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
      'Emit convergence diagnostics for the ladder and score field prediction as continuous validated evolution bits resolved to the residual-certified precision, with the predictability boundary emerging where certified bits reach zero. The binary gate and its parameters are retired.',
  },
  {
    id: 'cost',
    title: 'Algorithmic cost',
    outcome:
      'Read description length off the submitted program, add it to the machine-independent operation count, and turn the score into validated bits per unit of cost. The test: two models with identical predictions but different description length score differently.',
  },
  {
    id: 'hierarchy',
    title: 'Hierarchical query-space scoring',
    outcome:
      'Build the recursive partition, the problem-space refinement, and the adversarial stopping rule, so the score becomes a tree that contracts to the single number.',
  },
  {
    id: 'bootstrap',
    title: 'Trusted-ground persistence',
    outcome: 'Carry convergent regions forward and certify newly reached ground by the agreement-on-overlap check.',
  },
  {
    id: 'lawmetric',
    title: 'Law-induced metric from typed equations',
    outcome:
      'Recognize the structural type of a benchmark equation from its typed symbols, and derive the stability functional it implies — hence the one metric used for both the ambient entropy and the certified error — instead of declaring it. The verifier-or-amplification form of certification falls out of the type, and the derived functional is itself verified.',
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
      'A readout-valued, convergence-grounded task, plus the scaffolding for unknown-law and multi-scale universes. Parked until the basic resolution case is solid.',
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

export function ProgramPanel() {
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
        <h2>Program &mdash; adaptive precision tree</h2>
        <p>
          Every concept starts as a one-line gist; open it to go deeper a refinement at a time, until you reach the
          code that backs it (<span className="architecture-tag code">&#x27c2;</span>) or a question that is still open
          (<span className="architecture-tag horizon">open</span>). There are four roots, and they do not overlap: the
          universe, what you ask of it, how you decide an answer is right, and what you count. Everything else is built
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
