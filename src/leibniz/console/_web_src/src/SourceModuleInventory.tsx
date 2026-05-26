import { Boxes, CheckCircle2, TerminalSquare } from 'lucide-react';
import type { ReactNode } from 'react';

import type { SourceModuleRecord } from './sourceModules.ts';

type SourceModuleInventoryProps = {
  modules: SourceModuleRecord[];
};

export function SourceModuleInventory({ modules }: SourceModuleInventoryProps) {
  const exportCount = modules.reduce(
    (count, sourceModule) => count + sourceModule.public_exports.length,
    0,
  );

  return (
    <section className="source-inventory" aria-label="Source module inventory">
      <header className="artifact-browser-header">
        <div>
          <p className="section-label">Source Inventory</p>
          <h2>Public Modules</h2>
        </div>
        <dl className="artifact-metrics" aria-label="Source inventory summary">
          <div>
            <dt>Modules</dt>
            <dd>{modules.length}</dd>
          </div>
          <div>
            <dt>Exports</dt>
            <dd>{exportCount}</dd>
          </div>
          <div>
            <dt>Mode</dt>
            <dd>Read-only</dd>
          </div>
        </dl>
      </header>

      <div className="source-module-list" aria-label="Public package modules">
        {modules.map((sourceModule) => (
          <article className="source-module-row" key={sourceModule.module_name}>
            <div className="artifact-row-icon" aria-hidden="true">
              <Boxes size={18} />
            </div>
            <div className="source-module-main">
              <div className="artifact-row-title">
                <h3>{sourceModule.module_name}</h3>
                <span className="artifact-kind">{sourceModule.public_exports.length} exports</span>
              </div>
              <p className="artifact-path">{sourceModule.source_path}</p>

              <div className="source-module-sections">
                <SourceModuleList
                  icon={<CheckCircle2 size={14} />}
                  items={sourceModule.public_exports}
                  label="Public exports"
                  placeholder="No public exports declared."
                />
                <SourceModuleList
                  icon={<TerminalSquare size={14} />}
                  items={sourceModule.validation_commands}
                  label="Validation"
                  placeholder="No validation command declared."
                />
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function SourceModuleList({
  icon,
  items,
  label,
  placeholder,
}: {
  icon: ReactNode;
  items: string[];
  label: string;
  placeholder: string;
}) {
  return (
    <section className="source-module-section">
      <h4>{label}</h4>
      {items.length > 0 ? (
        <ul>
          {items.map((item) => (
            <li key={item}>
              {icon}
              <span>{item}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>{placeholder}</p>
      )}
    </section>
  );
}
