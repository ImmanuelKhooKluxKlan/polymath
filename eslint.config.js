import js from '@eslint/js';
import react from 'eslint-plugin-react';

const browserGlobals = Object.fromEntries([
  'window', 'document', 'navigator', 'console', 'alert', 'fetch', 'URL', 'URLSearchParams',
  'Blob', 'FileReader', 'Headers', 'TextEncoder', 'TextDecoder', 'performance',
  'requestAnimationFrame', 'cancelAnimationFrame', 'setTimeout', 'clearTimeout',
  'setInterval', 'clearInterval', 'AudioContext', 'btoa', 'atob', 'FormData', 'DOMParser',
].map((name) => [name, 'readonly']));

export default [
  {
    ignores: ['dist/**', 'node_modules/**', 'server/**', '**/*.backup'],
  },
  js.configs.recommended,
  {
    files: ['src/**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: browserGlobals,
    },
    plugins: { react },
    rules: {
      'react/jsx-uses-vars': 'error',
      'react/jsx-uses-react': 'off',
      'no-empty': ['error', { allowEmptyCatch: true }],
      'no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
];
