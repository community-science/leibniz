import { BookOpenCheck, Network } from 'lucide-react';
import type { ComponentType } from 'react';
import { useEffect, useState } from 'react';

import { ArchitecturePanel } from './ArchitecturePanel';
import { BenchmarksPanel } from './BenchmarksPanel';
import { usePersistentState } from './persistentState';
import initialConsoleData, { subscribeConsoleData } from 'virtual:leibniz-console-data';

type TabId = 'architecture' | 'benchmarks';

type ConsoleTab = {
  id: TabId;
  label: string;
  icon: ComponentType<{ size?: number }>;
};

const tabs: ConsoleTab[] = [
  { id: 'architecture', label: 'Architecture', icon: Network },
  { id: 'benchmarks', label: 'Benchmarks', icon: BookOpenCheck },
];

export function ConsoleShell() {
  const [currentTab, setCurrentTab] = usePersistentState<TabId>(
    'leibniz.console.currentTab',
    'architecture',
  );
  const [consoleData, setConsoleData] = useState(initialConsoleData);
  const activeTab = tabs.some((tab) => tab.id === currentTab) ? currentTab : 'benchmarks';

  useEffect(() => subscribeConsoleData(setConsoleData), []);

  return (
    <main className="mission-control">
      <header className="mission-header">
        <div className="logo">LEIBNIZ</div>
      </header>

      <nav className="tab-navigation" aria-label="Console views">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              key={tab.id}
              onClick={() => setCurrentTab(tab.id)}
              type="button"
              aria-current={activeTab === tab.id ? 'page' : undefined}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      <section className="tab-content">
        <div className="console-architecture" hidden={activeTab !== 'architecture'}>
          <ArchitecturePanel />
        </div>
        <div className="console-overview" hidden={activeTab !== 'benchmarks'}>
          <BenchmarksPanel
            modelInspections={consoleData.model_inspections}
            operatorVocabulary={consoleData.operator_vocabulary}
            resultViews={consoleData.result_views}
            tasks={consoleData.benchmark_tasks}
          />
        </div>
      </section>
    </main>
  );
}
