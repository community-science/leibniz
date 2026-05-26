import { Database, FileJson, Network } from 'lucide-react';

type ConsoleSection = {
  id: string;
  label: string;
  description: string;
};

const sections: ConsoleSection[] = [
  {
    id: 'artifacts',
    label: 'Artifacts',
    description: 'Protocol documents, digests, validation status, and references.',
  },
  {
    id: 'dependencies',
    label: 'Dependencies',
    description: 'Explicit dependency edges between public artifact references.',
  },
  {
    id: 'documents',
    label: 'Documents',
    description: 'Typed views over already-public Leibniz document families.',
  },
];

export function ConsoleShell() {
  return (
    <main className="console-shell">
      <header className="console-header">
        <div>
          <p className="eyebrow">Leibniz</p>
          <h1>Console</h1>
        </div>
      </header>

      <section className="console-grid" aria-label="Console sections">
        {sections.map((section) => (
          <article className="console-section" key={section.id}>
            <div className="section-icon" aria-hidden="true">
              {section.id === 'artifacts' ? (
                <Database size={20} />
              ) : section.id === 'dependencies' ? (
                <Network size={20} />
              ) : (
                <FileJson size={20} />
              )}
            </div>
            <div>
              <h2>{section.label}</h2>
              <p>{section.description}</p>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
