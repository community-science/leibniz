import { useState } from 'react';

import type { ArtifactReferenceRecord } from './artifactIndex.ts';
import type { ModelInspectionRecord } from './modelInspections.ts';

export function ModelInspectionPanel({
  inspections,
}: {
  inspections: ModelInspectionRecord[];
}) {
  const [selectedId, setSelectedId] = useState(inspections[0]?.id ?? '');
  const selected =
    inspections.find((inspection) => inspection.id === selectedId) ?? inspections[0];

  if (selected === undefined) {
    return (
      <section className="model-layout">
        <p className="artifact-detail-note">No model inspections are available.</p>
      </section>
    );
  }

  return (
    <section className="model-layout" aria-label="Model inspections">
      <aside className="model-list" aria-label="Model inspection records">
        {inspections.map((inspection) => (
          <button
            className={`model-list-item ${inspection.id === selected.id ? 'active' : ''}`}
            key={inspection.id}
            onClick={() => setSelectedId(inspection.id)}
            type="button"
          >
            <span>{inspectionLabel(inspection)}</span>
            <small>{inspection.source_path}</small>
          </button>
        ))}
      </aside>

      <article className="model-detail">
        <header className="model-header">
          <div>
            <h2>{selected.id}</h2>
            <p>{selected.source_path}</p>
          </div>
          <ModelCostSummary inspection={selected} />
        </header>

        <section className="model-section">
          <h3>Architecture</h3>
          <dl className="model-shape-grid">
            <div>
              <dt>Input</dt>
              <dd>{shapeLabel(selected.input_shape)}</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>{shapeLabel(selected.output_shape)}</dd>
            </div>
            <div>
              <dt>Architecture</dt>
              <dd>{referenceLabel(selected.architecture)}</dd>
            </div>
          </dl>
        </section>

        <section className="model-section">
          <h3>Layers</h3>
          <div className="model-layer-list">
            {selected.layers.map((layer) => (
              <article className="model-layer" key={layer.index}>
                <div className="model-layer-heading">
                  <span>{layer.index}</span>
                  <strong>{layer.kind}</strong>
                </div>
                <dl>
                  <dt>Input</dt>
                  <dd>{optionalShapeLabel(layer.input_shape)}</dd>
                  <dt>Output</dt>
                  <dd>{optionalShapeLabel(layer.output_shape)}</dd>
                  <dt>Parameters</dt>
                  <dd>{optionalNumberLabel(layer.parameter_count)}</dd>
                  <dt>Config</dt>
                  <dd>{parameterLabel(layer.parameters)}</dd>
                </dl>
              </article>
            ))}
          </div>
        </section>

        <ModelSources inspection={selected} />
      </article>
    </section>
  );
}

function ModelCostSummary({ inspection }: { inspection: ModelInspectionRecord }) {
  return (
    <dl className="model-metrics">
      <div>
        <dt>Layers</dt>
        <dd>{inspection.cost_summary.layer_count}</dd>
      </div>
      <div>
        <dt>Parameters</dt>
        <dd>{optionalNumberLabel(inspection.cost_summary.parameter_count)}</dd>
      </div>
      <div>
        <dt>Unknown</dt>
        <dd>{inspection.cost_summary.unknown_parameter_layers.join(', ') || 'none'}</dd>
      </div>
    </dl>
  );
}

function ModelSources({ inspection }: { inspection: ModelInspectionRecord }) {
  const sourceReferences = [
    referenceEntry('Model manifest', inspection.model_manifest),
    referenceEntry('Submission package', inspection.submission_package),
    referenceEntry('Benchmark', inspection.benchmark_manifest),
    referenceEntry('Measurements', inspection.measurement_dataset),
    ...inspection.model_artifacts.map((reference, index) => ({
      label: `Model artifact ${index + 1}`,
      reference,
    })),
    ...inspection.training_provenance.map((reference, index) => ({
      label: `Training provenance ${index + 1}`,
      reference,
    })),
  ].filter(
    (entry): entry is { label: string; reference: ArtifactReferenceRecord } => entry !== null,
  );

  if (sourceReferences.length === 0) {
    return null;
  }

  return (
    <section className="model-section">
      <h3>Sources</h3>
      <div className="model-source-list">
        {sourceReferences.map((entry) => (
          <dl className="model-source" key={`${entry.label}:${referenceLabel(entry.reference)}`}>
            <dt>{entry.label}</dt>
            <dd>{referenceLabel(entry.reference)}</dd>
          </dl>
        ))}
      </div>
    </section>
  );
}

function referenceEntry(label: string, reference: ArtifactReferenceRecord | undefined) {
  if (reference === undefined) {
    return null;
  }
  return { label, reference };
}

function inspectionLabel(inspection: ModelInspectionRecord): string {
  return `${shapeLabel(inspection.input_shape)} -> ${shapeLabel(inspection.output_shape)}`;
}

function shapeLabel(shape: number[]): string {
  return shape.join(' x ');
}

function optionalShapeLabel(shape: number[] | undefined): string {
  if (shape === undefined) {
    return 'unknown';
  }
  return shapeLabel(shape);
}

function optionalNumberLabel(value: number | undefined): string {
  if (value === undefined) {
    return 'unknown';
  }
  return value.toLocaleString();
}

function parameterLabel(parameters: Record<string, unknown>): string {
  const entries = Object.entries(parameters);
  if (entries.length === 0) {
    return 'none';
  }
  return entries.map(([key, value]) => `${key}: ${parameterValueLabel(value)}`).join(', ');
}

function parameterValueLabel(value: unknown): string {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return JSON.stringify(value);
}

function referenceLabel(reference: ArtifactReferenceRecord): string {
  return (
    reference.protocol_id ??
    reference.record_digest ??
    reference.content_digest ??
    reference.external_uri ??
    reference.kind
  );
}
