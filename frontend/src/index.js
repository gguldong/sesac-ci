import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

const container = document.getElementById('root');
const root = createRoot(container); // createRoot를 사용하여 렌더링합니다.

root.render(
  <App />
);
