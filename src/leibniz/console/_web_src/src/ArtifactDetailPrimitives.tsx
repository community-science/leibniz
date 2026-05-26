import type { ReactNode } from 'react';

export function DetailSection({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="artifact-detail-section">
      <h4>{title}</h4>
      {children}
    </section>
  );
}

export function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <dl className="artifact-detail-item">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </dl>
  );
}
