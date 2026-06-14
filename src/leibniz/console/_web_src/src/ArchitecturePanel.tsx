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
            tag: 'why charts aren’t the semantics',
            text: 'The chart only parameterizes; meaning lives in the ambient space, so everything counted on it is invariant to how g is written.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py — StateSpaceAmbient, StateSpaceAxis, generator surface',
          },
        ],
        verdict: { tone: 'ok', text: 'converges? yes — to code in two rungs.' },
      },
      {
        id: 'U-law',
        gist: 'A universe either has a law, or only convention.',
        meta: 'A field equation or game rules carry a law; a glyph’s label has none.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'names the axis',
            text: 'Call this intrinsic (a law decides) vs extrinsic (law-less, pre-scientific).',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: '⚠',
            text: 'This is half of what we used to call “grounding.” The other half (verifier vs convergence) belongs to R — see the grounding edge below.',
          },
        ],
        verdict: {
          tone: 'partial',
          text: 'converges? partly — clean only once grounding is split across U and R.',
        },
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
            text: 'Experiment lives on the reality↔law edge (does the theory model the world?). Task scoring lives on the task↔law edge.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'known vs unknown law',
            text: 'Known law → operationalize it. Unknown law → discover the theory from novel-intervention experiment (a virtual cell), a tower of scales.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'why fitting isn’t theory',
            text: 'Fitting measured data builds a predictor with no theory inside it. The multi-scale tower is bound by cross-scale consistency: an effective theory must not contradict the converged law beneath it.',
          },
        ],
        verdict: { tone: 'open', text: 'converges? no — the discovery / multi-scale rung is open.' },
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
        meta: 'One universe supports many; “classification vs prediction” is the wrong cut.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the form',
            text: 'A signature (access, binding, target): how it’s sampled, how the input determines the answer, and what shape the answer is.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: '⚠',
            text: 'Is Q its own root, or just “a region of U you point at”? If the latter, the basis is three roots, not four. (Thread #2.)',
          },
        ],
        verdict: { tone: 'open', text: 'converges? open at the root — Q’s independence from U is the shakiest claim.' },
      },
      {
        id: 'Q-binding',
        gist: 'Binding: evolution / equilibrium / inverse.',
        meta: 'Step forward in time · settle to an extremal state · infer a cause.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'examples',
            text: 'Evolution: an advancing field. Equilibrium: a molecule’s folded structure (a free-energy minimum, not a time-step). Inverse: read a cause from observations — the ill-posed cousin of running time backward.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'relation to the law',
            text: 'The menu of bindings is set by the law (U); which one you ask is the free choice that makes Q a root.',
          },
        ],
        verdict: { tone: 'partial', text: 'converges? mostly — completeness of the menu is open.' },
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
            tag: 'the orthogonality prize',
            text: 'Target shape is independent of how truth is established (R). “Target ⊥ grounding,” fought for over the structure-prediction turns, is just Q ⊥ R.',
          },
        ],
        verdict: { tone: 'ok', text: 'converges? yes — and confirms the basis.' },
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
        meta: 'The subtle one — opens deep.',
        status: 'direction',
        defaultOpen: true,
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the move',
            text: 'When you can’t know the exact answer, you call it correct by checking it stops changing as you compute it more carefully — and trust it as far as it stays stable.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'ladder & ruler',
            text: 'Ask at coarser, then finer resolution; compare on shared features. If it settles (Cauchy), the settled value is the answer and the leftover gap is the ruler ε. If it never settles, the answer is undefined and no model scores there.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'what falls out',
            text: 'Horizon, directionality, chaotic / laminar character are all outputs of whether it settles — never declared. A verifier is the degenerate ε = 0 corner (a discrete law is already fully resolved).',
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
            tag: '⚠ horizon',
            text: 'Diverges exactly where it should reach code: “within tolerance” (which tolerance? which norm? how many rungs?) and the adversarial robustness of “stable.” These are the named open questions.',
          },
        ],
        verdict: {
          tone: 'open',
          text: 'converges? no — and that is the signal. Cauchy down to the formal rung, then diverges at the code rung, matching the open-questions list.',
        },
      },
      {
        id: 'R-territory',
        gist: 'One operator, many territories.',
        meta: 'Correctness, the score-hierarchy, and the bootstrap are the same move.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the parameter',
            text: 'R refines a territory: resolution → correctness; problem-partition → the adaptive score-tree; operators over time → the bootstrap; and scale / intervention for multi-scale and discovery.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the collapse it buys',
            text: 'Three former “roots” (correctness, hierarchy, bootstrap) are one R with a territory parameter — the largest compression in the document.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the partition tree',
            text: 'On the problem-partition territory the score is a tree 𝒯 of regions, refined until the competence estimate is stable under adversarial subdivision — so a model cannot hide failure in an unrefined cell. A submission’s claim is a subtree of 𝒯; the scalar score is its measure-weighted contraction.',
          },
          {
            level: 3,
            kind: 'horizon',
            tag: '⚠',
            text: 'Is {resolution, partition, scale, intervention, operators-time} one clean axis, or does it hide structure? (Thread #3.)',
          },
        ],
        verdict: { tone: 'open', text: 'converges? open — the territory list may not be homogeneous.' },
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
            text: 'A model extends trusted ground by being Cauchy where prior rungs diverged while agreeing on the overlap (the anchor against self-consistent nonsense).',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'rungs are operators',
            text: 'The ladder’s rungs are operators — solvers, prior submissions, and (for discovery universes) experiments. For unknown-law universes the overlap test also demands cross-scale consistency with converged sub-laws.',
          },
          {
            level: 3,
            kind: 'add',
            tag: 'two ratchets',
            text: 'Two ratchets tighten over time, the same shape: trusted ground toward truth, and the cost proxy and its description-language toward the single algorithmic ideal.',
          },
        ],
        verdict: { tone: 'partial', text: 'converges? partly — persistence of trusted ground is unbuilt.' },
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
        meta: 'The settled one — bottoms out fast.',
        status: 'implemented',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'defines “different”',
            text: 'Two states differ only if distinguishable at the declared resolution; bits = log2(count), so doubling the variety adds one bit.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'locates it, justifies invariance',
            text: 'An ε-covering count under the declared metric, taken in the ambient space (not the chart) — invariant to parameterization; bits add because independent axes form a product measure.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'the region grammar',
            text: 'Regions are finite disjoint unions of products of per-axis regions; qualitative labels are strata (typed annotations, never axes); volume may be exact or a bracketed estimate.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'state_space.py — Distinguishability, log2 μ, RegionFiltration',
          },
        ],
        verdict: { tone: 'ok', text: 'converges? yes — two rungs to code; a short ladder, honestly short.' },
      },
      {
        id: 'D-ledger',
        gist: 'One ledger: value (credit) − cost (debit).',
        meta: 'Value and cost are the same unit, opposite signs.',
        status: 'direction',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the two entries',
            text: 'Credit = bits of validated prediction. Debit = bits to describe and run the operator. Net score = compression = world explained − operator spent.',
          },
          {
            level: 2,
            kind: 'add',
            tag: 'one quantity to minimize',
            text: 'Science is compression: the single quantity is the total codelength of validated reality — description-length + log2(ops) + unpredicted residual. The research frontier is validated bits of prediction per unit algorithmic cost.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: '⚠',
            text: 'Does the credit − debit collapse really hold as one number, or do value and cost resist unification (forcing a second axis)? (Thread #4.)',
          },
        ],
        verdict: { tone: 'open', text: 'converges? open — “one number” is asserted, not yet shown.' },
      },
      {
        id: 'D-cost',
        gist: 'Cost = description length + log2(operations).',
        meta: 'Levin complexity: Occam made computable. Compute alone is the term-dropping case.',
        status: 'mixed',
        rungs: [
          {
            level: 1,
            kind: 'add',
            tag: 'the two constraints',
            text: 'Computable from the submitted program; independent of the machine — so log-ops is the per-op cost model, description-length is a property of the program.',
          },
          {
            level: 2,
            kind: 'horizon',
            tag: '⚠',
            text: 'How is description-length read from a program (compressed params + architecture? two-part MDL?), and against which reference language? Open.',
          },
          {
            level: 3,
            kind: 'code',
            tag: '⟂',
            text: 'cost metrology — operation count is implemented; the algorithmic reframe is direction',
          },
        ],
        verdict: { tone: 'partial', text: 'converges? half — log-ops reaches code; description-length is open.' },
      },
    ],
  },
];

const EDGES: CouplingEdge[] = [
  {
    id: 'edge-distinguishability',
    roots: ['U', 'D'],
    title: 'Distinguishability',
    body: '“States” belong to the universe; “counted in bits” belongs to the currency. The metric (d, ε) is where U hands D something to count. A clean, low-tension edge — but real: D cannot run without U supplying a metric.',
  },
  {
    id: 'edge-grounding',
    roots: ['U', 'R'],
    title: 'Grounding (the disentanglement)',
    warn: true,
    body: 'The sharpest finding. “Grounding” was never primitive: intrinsic / extrinsic is a U question (has a law?), verifier / convergence is an R question (is ε = 0?). It felt primitive because it sits on this edge. Thread #1: if so, the contract should carry a law-status on U and let the ε = 0 / ε > 0 split emerge from R — no grounding field at all.',
  },
  {
    id: 'edge-score',
    roots: ['U', 'R', 'D'],
    title: 'The score (triple point)',
    body: 'Competence = predictive mass within ε (from R) of the converged answer, in bits (D), integrated over the query space (U) — and the integral is itself an R-refined adaptive tree. The score is not a root; it is the point where all three meet, and it needs nothing else. That is the basis passing its own test.',
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
          The architecture as a forest of adaptive precision ladders. Every concept shows a one-line gist; open it to
          descend its refinements, each tagged with the precision it <em>adds</em> (never retracts), bottoming out at
          code <span className="architecture-tag code">&#x27c2;</span> or at a{' '}
          <span className="architecture-tag horizon">&#x26a0; horizon</span> where understanding stops converging. The
          claim under test: the four roots are an orthogonal basis &mdash; what exists, what you ask, how you know, what
          you count &mdash; and every other concept is a composition of them. The places they couple are the edges.
        </p>
        <p className="architecture-intro-provenance">
          Each node is tagged by status: <span className="architecture-status implemented">implemented</span> parts are
          backed by merged work; <span className="architecture-status direction">design direction</span> parts are
          agreed but unbuilt, with their open questions tracked in the planning sequence. This view is the canonical home
          for the architecture; it supersedes the earlier prose proposal.
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
          A perfectly orthogonal basis would have no edges. There are three, and they are the interesting strain: each
          is a concept that lives <em>between</em> roots rather than inside one.
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
