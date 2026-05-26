import { Database, FileJson, FolderTree, Network, ScrollText } from 'lucide-react';
import { useState } from 'react';

import { ArtifactBrowser } from './ArtifactBrowser';
import consoleData from 'virtual:leibniz-console-data';

type ConsoleSection = {
  id: string;
  label: string;
  description: string;
};

type TabId = 'home' | 'artifacts' | 'source';

type ConsoleTab = {
  id: TabId;
  label: string;
};

const tabs: ConsoleTab[] = [
  { id: 'home', label: 'Home' },
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'source', label: 'Source' },
];

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
  const [currentTab, setCurrentTab] = useState<TabId>('home');

  return (
    <main className="mission-control">
      <header className="mission-header">
        <div className="logo">LEIBNIZ</div>
      </header>

      <nav className="tab-navigation" aria-label="Console views">
        {tabs.map((tab) => (
          <button
            className={`tab-button ${currentTab === tab.id ? 'active' : ''}`}
            key={tab.id}
            onClick={() => setCurrentTab(tab.id)}
            type="button"
          >
            {tab.id === 'home' ? (
              <ScrollText size={16} />
            ) : tab.id === 'artifacts' ? (
              <Database size={16} />
            ) : (
              <FolderTree size={16} />
            )}
            {tab.label}
          </button>
        ))}
      </nav>

      <section className="tab-content">
        <div className="console-overview" hidden={currentTab !== 'home'}>
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
        </div>

        <div className="console-overview" hidden={currentTab !== 'artifacts'}>
          <ArtifactBrowser
            details={consoleData.artifact_details}
            index={consoleData.artifact_index}
          />
        </div>

        <div className="console-overview" hidden={currentTab !== 'source'}>
          <section className="console-grid" aria-label="Source views">
            <article className="console-section">
              <div className="section-icon" aria-hidden="true">
                <FolderTree size={20} />
              </div>
              <div>
                <h2>Source</h2>
                <p>Public package modules, exports, tests, and validation commands.</p>
              </div>
            </article>
          </section>
        </div>
      </section>
    </main>
  );
}
