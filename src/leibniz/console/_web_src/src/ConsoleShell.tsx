import {
  BarChart3,
  BookOpenCheck,
  Cpu,
  Database,
  FileJson,
  FolderTree,
  Network,
  ScrollText,
} from 'lucide-react';
import { useState } from 'react';

import { ArtifactBrowser } from './ArtifactBrowser';
import { BenchmarksPanel } from './BenchmarksPanel';
import { ModelInspectionPanel } from './ModelInspectionPanel';
import { PerformanceViewPanel } from './PerformanceViewPanel';
import { SourceModuleInventory } from './SourceModuleInventory';
import consoleData from 'virtual:leibniz-console-data';

type ConsoleSection = {
  id: string;
  label: string;
  description: string;
};

type TabId = 'home' | 'artifacts' | 'benchmarks' | 'performance' | 'models' | 'source';

type ConsoleTab = {
  id: TabId;
  label: string;
};

const tabs: ConsoleTab[] = [
  { id: 'home', label: 'Home' },
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'performance', label: 'Performance' },
  { id: 'models', label: 'Models' },
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
  {
    id: 'benchmarks',
    label: 'Benchmarks',
    description: 'Benchmark task panes, generated samples, and benchmark-owned data views.',
  },
  {
    id: 'performance',
    label: 'Performance',
    description: 'Derived competence-integral views over benchmark-owned measurement cases.',
  },
  {
    id: 'models',
    label: 'Models',
    description: 'Read-only architecture, dimensionality, cost, and model source inspection.',
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
            ) : tab.id === 'benchmarks' ? (
              <BookOpenCheck size={16} />
            ) : tab.id === 'performance' ? (
              <BarChart3 size={16} />
            ) : tab.id === 'models' ? (
              <Cpu size={16} />
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
                  ) : section.id === 'benchmarks' ? (
                    <BookOpenCheck size={20} />
                  ) : section.id === 'performance' ? (
                    <BarChart3 size={20} />
                  ) : section.id === 'models' ? (
                    <Cpu size={20} />
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

        <div className="console-overview" hidden={currentTab !== 'benchmarks'}>
          <BenchmarksPanel tasks={consoleData.benchmark_tasks} />
        </div>

        <div className="console-overview" hidden={currentTab !== 'performance'}>
          <PerformanceViewPanel
            resultViews={consoleData.result_views}
            views={consoleData.performance_views}
          />
        </div>

        <div className="console-overview" hidden={currentTab !== 'models'}>
          <ModelInspectionPanel inspections={consoleData.model_inspections} />
        </div>

        <div className="console-overview" hidden={currentTab !== 'source'}>
          <SourceModuleInventory modules={consoleData.source_modules} />
        </div>
      </section>
    </main>
  );
}
