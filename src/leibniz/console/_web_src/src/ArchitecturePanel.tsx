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
  defaultOpen?: boolean;
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
        gist: 'A universe is states on a domain, under a law.',
        meta: 'An evolving field on a mesh, a board game’s positions, a labeled glyph.',
        status: 'implemented',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'ambient vs chart',
            text: 'The states form an ambient field space; a benchmark reaches them by turning a few measured chart axes (a generator g: Θ → X).',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'charts are not the semantics',
            text: 'The chart only parameterizes; meaning lives in the ambient space, so everything counted on it is invariant to how the generator is written.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py — StateSpaceAmbient, StateSpaceAxis, generator surface',
          },
        ],
        verdict: { tone: 'ok', text: 'Specified in code.' },
      },
      {
        id: 'U-law',
        gist: 'A universe carries a law, or only a convention.',
        meta: 'A field equation or game rules carry a law; a glyph’s label rests on convention.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'intrinsic vs extrinsic',
            text: 'A law makes correctness intrinsic; a convention makes it extrinsic — a pre-scientific labeling rather than a question with a governing law.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one half of grounding',
            text: 'Whether a universe carries a law is one of the two questions usually bundled as “grounding.” The other — exact verifier versus convergence — belongs to Refinement (see the grounding edge).',
          },
        ],
        verdict: { tone: 'partial', text: 'Grounding resolves across Universe and Refinement; see the grounding edge.' },
      },
      {
        id: 'U-levels',
        gist: 'Three levels: reality, law, tasks.',
        meta: 'Experiment validates the law; it never defines a task’s answer.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'where experiment acts',
            text: 'Experiment connects reality and the law — does the theory model the world? Task scoring connects a task to the law.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'known vs unknown law',
            text: 'A known law is operationalized; an unknown law is discovered from novel-intervention experiment — for example a virtual cell, a tower of scales.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'fitting is not theory',
            text: 'Fitting measured data yields a predictor with no theory inside it. In a multi-scale universe an effective theory must stay consistent with the converged law beneath it.',
          },
        ],
        verdict: { tone: 'open', text: 'Discovery and multi-scale universes remain to be specified.' },
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
        gist: 'A task is a question put to a universe.',
        meta: 'One universe supports many tasks — more than a classification-vs-prediction split.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the form',
            text: 'A signature (access, binding, target): how the universe is sampled, how the input determines the answer, and what shape the answer takes.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'Whether a query is an independent facet or simply a region of the universe singled out — and so whether there are four roots or three — remains open.',
          },
        ],
        verdict: { tone: 'open', text: 'A query’s independence from the universe remains to be settled.' },
      },
      {
        id: 'Q-binding',
        gist: 'Binding: evolution, equilibrium, or inverse.',
        meta: 'Step forward in time · settle to an extremal state · infer a cause.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'examples',
            text: 'Evolution: an advancing field. Equilibrium: a molecule’s folded structure — a free-energy minimum, not a time-step. Inverse: a cause read from observations, the ill-posed counterpart of running time backward.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'relation to the law',
            text: 'The law fixes the menu of bindings; choosing one is the free part of asking a question.',
          },
        ],
        verdict: { tone: 'partial', text: 'The full menu of bindings remains to be enumerated.' },
      },
      {
        id: 'Q-target',
        gist: 'Target: a state, or a readout.',
        meta: 'A field or position, vs a label, scalar, or decision.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'independent of grounding',
            text: 'Target shape is independent of how truth is established: a state-valued or a readout task can each be verifier- or convergence-grounded. That independence is what a classification-vs-prediction split obscures.',
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
        gist: 'Correctness is convergence, not an oracle.',
        meta: 'When no exact answer exists, refinement decides correctness.',
        status: 'direction',
        defaultOpen: true,
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the move',
            text: 'When the exact answer is unknowable, an answer counts as correct when it stops changing as it is computed more carefully — and it is trusted only as far as it stays stable.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'ladder and ruler',
            text: 'Compute at coarser, then finer resolution and compare on shared features. If the sequence settles (Cauchy), the settled value is the answer and the leftover gap is the ruler ε. If it never settles, the answer is undefined and no model scores there.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'what falls out',
            text: 'The horizon, the direction of time, and chaotic-versus-laminar character are outputs of whether the sequence settles — not declarations. A verifier is the limiting case where the gap is zero: a discrete law is already fully resolved.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'formal objects',
            text: 'r_k = R_ℓ Φ̂(refine_k(in)), gaps g_k = ‖r_{k+1} − r_k‖, Cauchy when g_k → 0 within tolerance, answer = r_∞, ruler ε = g_∞.',
          },
          {
            level: 4,
            kind: 'horizon',
            tag: 'open',
            text: 'The operational meaning of “within tolerance” — which norm, which tolerance, how many rungs — and the precise stability criterion remain to be specified.',
          },
        ],
        verdict: { tone: 'open', text: 'Defined down to its formal objects; the operational details remain open.' },
      },
      {
        id: 'R-territory',
        gist: 'One operator, many territories.',
        meta: 'Correctness, the score hierarchy, and the bootstrap share one mechanism.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the parameter',
            text: 'Refinement applies to a territory: resolution gives correctness; the problem partition gives the score hierarchy; operators over time give the bootstrap; scale and intervention give multi-scale and discovery.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one mechanism',
            text: 'Correctness, the score hierarchy, and the bootstrap are the same refinement over different territories — one mechanism, not three.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the partition tree',
            text: 'Over the problem partition the score is a tree of regions, refined until each region’s competence estimate is stable under adversarial subdivision — so a model cannot hide failure in an unrefined cell. A submission’s claim is a subtree; the scalar score is its measure-weighted contraction.',
          },
          {
            level: 3,
            kind: 'horizon',
            tag: 'open',
            text: 'Whether these territories form one uniform axis or several distinct kinds remains open.',
          },
        ],
        verdict: { tone: 'open', text: 'The shared mechanism is clear; the uniformity of its territories is open.' },
      },
      {
        id: 'R-ratchet',
        gist: 'Trusted ground is a ratchet.',
        meta: 'Truth is an unreached limit the community tightens toward.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'how it grows',
            text: 'A model extends trusted ground by settling where earlier ones diverged while agreeing with them on the overlap — the safeguard against confidently converging to nonsense.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'rungs are operators',
            text: 'The rungs are operators — solvers, earlier submissions, and, in discovery universes, experiments. In an unknown-law universe the overlap test also requires consistency with converged sub-laws across scales.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'two ratchets',
            text: 'Two ratchets tighten together: trusted ground toward truth, and the cost measure and its description language toward the single algorithmic ideal.',
          },
        ],
        verdict: { tone: 'partial', text: 'Persisting trusted ground across submissions remains to be built.' },
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
        gist: 'Volume is a distinguishable-state count; bits add.',
        meta: 'The unit the whole score is built on.',
        status: 'implemented',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'defines “different”',
            text: 'Two states differ only when distinguishable at the declared resolution; bits = log2(count), so doubling the variety adds one bit.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'locates it, justifies invariance',
            text: 'An ε-covering count under the declared metric, taken in the ambient space rather than the chart — invariant to parameterization; bits add because independent axes form a product measure.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the region grammar',
            text: 'Regions are finite disjoint unions of products of per-axis regions; qualitative labels are strata (typed annotations, never axes); volume is exact or a bracketed estimate.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py — Distinguishability, log2 μ, RegionFiltration',
          },
        ],
        verdict: { tone: 'ok', text: 'Specified in code.' },
      },
      {
        id: 'D-ledger',
        gist: 'One ledger: value (credit) minus cost (debit).',
        meta: 'Value and cost share one unit, opposite signs.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the two entries',
            text: 'Credit is bits of validated prediction; debit is bits to describe and run the operator. The net score is compression: world explained minus operator spent.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one quantity to minimize',
            text: 'The single quantity is the total codelength of validated reality — description length + log2(operations) + unpredicted residual. The frontier is validated bits of prediction per unit of algorithmic cost.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'Whether value and cost reduce to one number, or require separate axes, remains open.',
          },
        ],
        verdict: { tone: 'open', text: 'The single-number reduction remains to be demonstrated.' },
      },
      {
        id: 'D-cost',
        gist: 'Cost = description length + log2(operations).',
        meta: 'Levin complexity: Occam’s razor made computable; plain compute drops the description-length term.',
        status: 'mixed',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the two constraints',
            text: 'Computable from the submitted program and independent of the machine: log-operations from the per-op cost model, description length from the program itself.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: 'open',
            text: 'How description length is read from a program — compressed parameters and architecture, or a two-part code — and against which reference language, remains to be defined.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'cost metrology — operation count is in code; the algorithmic reframe is a direction',
          },
        ],
        verdict: { tone: 'partial', text: 'Operation count is in code; description length remains to be defined.' },
      },
    ],
  },
];

const EDGES: CouplingEdge[] = [
  {
    id: 'edge-distinguishability',
    roots: ['U', 'D'],
    title: 'Distinguishability',
    body: 'States belong to the universe; counting them in bits belongs to the currency. The metric (d, ε) is where the universe gives the currency something to count — a clean coupling, and a real one: counting needs a metric to count against.',
  },
  {
    id: 'edge-grounding',
    roots: ['U', 'R'],
    title: 'Grounding',
    warn: true,
    body: 'Grounding bundles two independent questions: whether the universe carries a law (Universe) and whether an answer is checked by an exact verifier or by convergence (Refinement). A contract can carry a law-status on the universe and let the verifier-versus-convergence distinction follow from refinement, rather than declaring grounding directly.',
  },
  {
    id: 'edge-score',
    roots: ['U', 'R', 'D'],
    title: 'The score (triple point)',
    body: 'Competence is the predictive mass within ε (from Refinement) of the converged answer, measured in bits (Description length), integrated over the query space (Universe) — and the integral is itself a refined adaptive tree. The score is not a root but the point where all three meet, and it needs nothing more.',
  },
];

const STATUS_LABEL: Record<NodeStatus, string> = {
  implemented: 'implemented',
  direction: 'design direction',
  mixed: 'mixed',
};

export function ArchitecturePanel() {
  const [maxLevel, setMaxLevel] = useState(4);
  const [openNodes, setOpenNodes] = useState<Set<string>>(
    () => new Set(ROOTS.flatMap((root) => root.nodes).filter((node) => node.defaultOpen).map((node) => node.id)),
  );
  const [highlightRoot, setHighlightRoot] = useState<RootId | null>(null);
  const rootRefs = useRef<Partial<Record<RootId, HTMLElement | null>>>({});

  useEffect(() => {
    if (highlightRoot === null) {
      return;
    }
    const timer = window.setTimeout(() => setHighlightRoot(null), 1500);
    return () => window.clearTimeout(timer);
  }, [highlightRoot]);

  const selectLevel = (value: number) => {
    setMaxLevel(value);
    setOpenNodes(new Set());
  };

  const toggleNode = (id: string) => {
    setOpenNodes((previous) => {
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
          The architecture as a forest of adaptive precision ladders. Each concept opens with a one-line gist and
          descends into refinements that add precision, ending at a code reference{' '}
          <span className="architecture-tag code">&#x27c2;</span> or an open design question{' '}
          <span className="architecture-tag horizon">open</span>. Four orthogonal roots organize it &mdash; what
          exists, what you ask, how you know, what you count &mdash; and every other concept is a composition of them,
          shown as the edges where the roots couple.
        </p>
        <p className="architecture-intro-provenance">
          Each node is marked <span className="architecture-status implemented">implemented</span> &mdash; backed by
          working code &mdash; or <span className="architecture-status direction">design direction</span> &mdash;
          planned, with open questions noted inline.
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
              const isOpen = openNodes.has(node.id);
              return (
                <div className={`architecture-node ${isOpen ? 'open' : ''}`} key={node.id}>
                  <button className="architecture-node-head" onClick={() => toggleNode(node.id)} type="button" aria-expanded={isOpen}>
                    <span className="architecture-twist" aria-hidden="true">
                      &#x25b6;
                    </span>
                    <span className="architecture-gist">
                      {node.gist}
                      <span className="architecture-meta">
                        {node.meta}
                        <span className={`architecture-status ${node.status}`}>{STATUS_LABEL[node.status]}</span>
                      </span>
                    </span>
                  </button>
                  <div className="architecture-rungs">
                    {node.rungs.map((rung, index) => {
                      const visible = isOpen || rung.level <= maxLevel;
                      if (!visible) {
                        return null;
                      }
                      return (
                        <div className={`architecture-rung ${rung.kind}`} key={index}>
                          <span className={`architecture-tag ${rung.kind}`}>{rung.tag}</span>
                          <span className="architecture-rung-text">{rung.text}</span>
                        </div>
                      );
                    })}
                    <div className={`architecture-verdict ${node.verdict.tone}`}>{node.verdict.text}</div>
                  </div>
                </div>
              );
            })}
          </section>
        ))}
      </div>

      <section className="architecture-edges" aria-label="Edges where the roots couple">
        <h3>Edges &mdash; where the roots couple</h3>
        <p className="architecture-edges-note">
          Some concepts are not contained in a single root; they couple two or more. These edges record where the
          roots meet.
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
            <p className="architecture-edge-body">{edge.body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
