import { BookOpenCheck } from 'lucide-react';
import { useEffect, useState } from 'react';

import { BenchmarksPanel } from './BenchmarksPanel';
import initialConsoleData, { subscribeConsoleData } from 'virtual:leibniz-console-data';

type TabId = 'benchmarks';

type ConsoleTab = {
  id: TabId;
  label: string;
};

const tabs: ConsoleTab[] = [
  { id: 'benchmarks', label: 'Benchmarks' },
];

export function ConsoleShell() {
  const [currentTab, setCurrentTab] = useState<TabId>('benchmarks');
  const [consoleData, setConsoleData] = useState(initialConsoleData);

  useEffect(() => subscribeConsoleData(setConsoleData), []);

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
            aria-current={currentTab === tab.id ? 'page' : undefined}
          >
            <BookOpenCheck size={16} />
            {tab.label}
          </button>
        ))}
      </nav>

      <section className="tab-content">
        <div className="console-overview" hidden={currentTab !== 'benchmarks'}>
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
