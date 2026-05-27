import {
  BookOpenCheck,
  Database,
} from 'lucide-react';
import { useState } from 'react';

import { ArtifactBrowser } from './ArtifactBrowser';
import { BenchmarksPanel } from './BenchmarksPanel';
import consoleData from 'virtual:leibniz-console-data';

type TabId = 'benchmarks' | 'artifacts';

type ConsoleTab = {
  id: TabId;
  label: string;
};

const tabs: ConsoleTab[] = [
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'artifacts', label: 'Artifacts' },
];

export function ConsoleShell() {
  const [currentTab, setCurrentTab] = useState<TabId>('benchmarks');

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
            {tab.id === 'artifacts' ? (
              <Database size={16} />
            ) : (
              <BookOpenCheck size={16} />
            )}
            {tab.label}
          </button>
        ))}
      </nav>

      <section className="tab-content">
        <div className="console-overview" hidden={currentTab !== 'benchmarks'}>
          <BenchmarksPanel
            modelInspections={consoleData.model_inspections}
            resultViews={consoleData.result_views}
            tasks={consoleData.benchmark_tasks}
          />
        </div>

        <div className="console-overview" hidden={currentTab !== 'artifacts'}>
          <ArtifactBrowser
            details={consoleData.artifact_details}
            index={consoleData.artifact_index}
          />
        </div>
      </section>
    </main>
  );
}
