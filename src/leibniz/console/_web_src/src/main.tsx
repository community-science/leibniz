import { createRoot } from 'react-dom/client';

import { ConsoleShell } from './ConsoleShell';
import './styles.css';

const root = document.getElementById('root');

if (root === null) {
  throw new Error('console root element is missing');
}

createRoot(root).render(<ConsoleShell />);
