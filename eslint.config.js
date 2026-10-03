import js from "@eslint/js";

const runtimeGlobals = {
  AbortController: "readonly",
  AbortSignal: "readonly",
  Blob: "readonly",
  Buffer: "readonly",
  clearImmediate: "readonly",
  clearInterval: "readonly",
  clearTimeout: "readonly",
  console: "readonly",
  fetch: "readonly",
  FormData: "readonly",
  Headers: "readonly",
  performance: "readonly",
  process: "readonly",
  queueMicrotask: "readonly",
  Request: "readonly",
  Response: "readonly",
  setImmediate: "readonly",
  setInterval: "readonly",
  setTimeout: "readonly",
  structuredClone: "readonly",
  TextDecoder: "readonly",
  TextEncoder: "readonly",
  URL: "readonly",
  URLSearchParams: "readonly",
};

export default [
  {
    ignores: ["node_modules/**"],
  },
  js.configs.recommended,
  {
    files: ["**/*.{js,mjs,cjs}"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: runtimeGlobals,
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "no-console": "off",
      eqeqeq: "error",
      "no-var": "error",
      "prefer-const": "error",
    },
  },
];
