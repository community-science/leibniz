import { useEffect, useRef, useState } from 'react';

import {
  decodeFieldPreview,
  type ObservationInspectionRecord,
} from './observationInspections.ts';

export function ObservationInspectionPanel({
  inspections,
}: {
  inspections: ObservationInspectionRecord[];
}) {
  const [selectedId, setSelectedId] = useState(inspections[0]?.id ?? '');
  const selected = inspections.find((inspection) => inspection.id === selectedId) ?? inspections[0];

  if (selected === undefined) {
    return (
      <section className="inspection-layout">
        <p className="artifact-detail-note">No observation inspections are available.</p>
      </section>
    );
  }

  return (
    <section className="inspection-layout" aria-label="Observation inspections">
      <aside className="inspection-list" aria-label="Inspection samples">
        {inspections.map((inspection) => (
          <button
            className={`inspection-list-item ${inspection.id === selected.id ? 'active' : ''}`}
            key={inspection.id}
            onClick={() => setSelectedId(inspection.id)}
            type="button"
          >
            <span>{inspection.label}</span>
            <small>{axisAssignmentLabel(inspection.complexity_assignment)}</small>
          </button>
        ))}
      </aside>
      <article className="inspection-detail">
        <div className="inspection-preview-shell">
          <FieldPreviewCanvas inspection={selected} />
        </div>
        <div className="inspection-metadata">
          <h2>{selected.label}</h2>
          <dl>
            <dt>Benchmark</dt>
            <dd>{selected.benchmark_id}</dd>
            <dt>Outcome</dt>
            <dd>{selected.outcome_id ?? 'unlabeled'}</dd>
            <dt>Scale</dt>
            <dd>{axisAssignmentLabel(selected.scale_assignment)}</dd>
            <dt>Complexity</dt>
            <dd>{axisAssignmentLabel(selected.complexity_assignment)}</dd>
            <dt>Resolution</dt>
            <dd>{axisAssignmentLabel(selected.resolution_assignment)}</dd>
            <dt>Components</dt>
            <dd>{selected.component_sequence.join(', ')}</dd>
            <dt>Field</dt>
            <dd>{shapeLabel(selected.field_shape)}</dd>
            <dt>Digest</dt>
            <dd>{selected.field_digest}</dd>
            <dt>Showcase</dt>
            <dd>{selected.showcase.source_path}</dd>
          </dl>
        </div>
      </article>
    </section>
  );
}

function FieldPreviewCanvas({ inspection }: { inspection: ObservationInspectionRecord }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const [_channels, height, width] = inspection.field_preview.shape;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (context === null) {
      return;
    }
    const values = decodeFieldPreview(inspection.field_preview);
    const image = context.createImageData(width, height);
    for (let index = 0; index < width * height; index += 1) {
      const value = values[index] ?? 0;
      const offset = index * 4;
      image.data[offset] = 255;
      image.data[offset + 1] = 255;
      image.data[offset + 2] = 255;
      image.data[offset + 3] = value;
    }
    context.clearRect(0, 0, width, height);
    context.putImageData(image, 0, 0);
  }, [inspection]);

  return <canvas aria-label={inspection.label} className="inspection-preview" ref={canvasRef} />;
}

function axisAssignmentLabel(assignment: { values: { axis: string; value: number }[] }): string {
  return assignment.values.map((item) => `${item.axis}: ${item.value}`).join(', ');
}

function shapeLabel(shape: number[]): string {
  return shape.join(' x ');
}
