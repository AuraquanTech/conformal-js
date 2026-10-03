/**
 * Verify that the shipped declaration file stays semantically compatible with
 * declarations generated from the JavaScript implementation's JSDoc.
 *
 * The comparison is bidirectional for the value API. This catches declaration
 * drift without requiring the committed declaration file to use the compiler's
 * exact formatting or inline-vs-alias choices. Temporary files are written only
 * under the OS temp directory and are removed before exit.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, normalize, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(import.meta.url);
const canonical = (file) => normalize(resolve(file));
const sourceFile = resolve(root, 'src/conformal.js');
const publishedFile = resolve(root, 'src/conformal.d.ts');
let ts;
try {
  ts = require(resolve(root, 'node_modules/typescript'));
} catch {
  console.error('Declaration verification requires local TypeScript. Run npm ci --ignore-scripts.');
  process.exit(1);
}

const formatDiagnostics = (diagnostics) => ts.formatDiagnosticsWithColorAndContext(diagnostics, {
  getCanonicalFileName: canonical,
  getCurrentDirectory: () => root,
  getNewLine: () => '\n',
});

function sourceConfig() {
  const configFile = resolve(root, 'tsconfig.source.json');
  const loaded = ts.readConfigFile(configFile, ts.sys.readFile);
  if (loaded.error) throw new Error(formatDiagnostics([loaded.error]));
  const parsed = ts.parseJsonConfigFileContent(loaded.config, ts.sys, root, undefined, configFile);
  if (parsed.errors.length) throw new Error(formatDiagnostics(parsed.errors));
  assert.ok(parsed.fileNames.some((file) => canonical(file) === canonical(sourceFile)),
    'Source program must explicitly include src/conformal.js');
  return parsed;
}

function generateImplementationDeclaration() {
  const parsed = sourceConfig();
  const options = {
    ...parsed.options,
    noEmit: false,
    declaration: true,
    emitDeclarationOnly: true,
    declarationMap: false,
    sourceMap: false,
  };
  const program = ts.createProgram(parsed.fileNames, options);
  const errors = ts.getPreEmitDiagnostics(program)
    .filter((diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error);
  assert.equal(errors.length, 0, formatDiagnostics(errors));

  const outputs = [];
  const emitted = program.emit(undefined, (name, text) => {
    if (name.endsWith('.d.ts')) outputs.push({ name, text });
  }, undefined, true);
  const emitErrors = emitted.diagnostics
    .filter((diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error);
  assert.equal(emitErrors.length, 0, formatDiagnostics(emitErrors));
  assert.equal(outputs.length, 1, `Expected one generated declaration, got ${outputs.length}`);
  return outputs[0].text;
}

function exportedTypeNames(text, fileName) {
  const source = ts.createSourceFile(fileName, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const names = new Set();
  for (const statement of source.statements) {
    if (!ts.isTypeAliasDeclaration(statement) && !ts.isInterfaceDeclaration(statement)) continue;
    const exported = statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword);
    if (exported) names.add(statement.name.text);
  }
  return names;
}

function exportedObjectReadonlyShapes(text, fileName) {
  const source = ts.createSourceFile(fileName, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const shapes = new Map();
  for (const statement of source.statements) {
    const exported = statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword);
    if (!exported) continue;
    let members;
    if (ts.isInterfaceDeclaration(statement)) members = statement.members;
    else if (ts.isTypeAliasDeclaration(statement) && ts.isTypeLiteralNode(statement.type)) members = statement.type.members;
    else continue;
    const properties = new Map();
    for (const member of members) {
      if (!ts.isPropertySignature(member)) continue;
      const name = member.name.getText(source);
      const readonly = member.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ReadonlyKeyword) ?? false;
      properties.set(name, readonly);
    }
    if (properties.size) shapes.set(statement.name.text, properties);
  }
  return shapes;
}

function readonlySurfaceDrift(actualText, publishedText) {
  const actual = exportedObjectReadonlyShapes(actualText, 'generated.d.ts');
  const published = exportedObjectReadonlyShapes(publishedText, 'published.d.ts');
  const drift = [];
  for (const [typeName, actualProperties] of actual) {
    const publishedProperties = published.get(typeName);
    if (!publishedProperties) continue;
    const propertyNames = new Set([...actualProperties.keys(), ...publishedProperties.keys()]);
    for (const propertyName of propertyNames) {
      if (!actualProperties.has(propertyName)) {
        drift.push(`${typeName}.${propertyName}: published-only property`);
        continue;
      }
      if (!publishedProperties.has(propertyName)) {
        drift.push(`${typeName}.${propertyName}: missing published property`);
        continue;
      }
      if (actualProperties.get(propertyName) !== publishedProperties.get(propertyName)) {
        drift.push(`${typeName}.${propertyName}: readonly modifier differs`);
      }
    }
  }
  return drift;
}

function compatibilityDiagnostics(tempRoot, actualText, publishedText) {
  const actualFile = join(tempRoot, 'actual', 'conformal.d.ts');
  const declaredFile = join(tempRoot, 'published', 'conformal.d.ts');
  const contractFile = join(tempRoot, 'contract.mts');
  mkdirSync(dirname(actualFile), { recursive: true });
  mkdirSync(dirname(declaredFile), { recursive: true });
  writeFileSync(actualFile, actualText, 'utf8');
  writeFileSync(declaredFile, publishedText, 'utf8');
  writeFileSync(contractFile, `
import * as Actual from './actual/conformal.js';
import * as Published from './published/conformal.js';

declare const actual: typeof Actual;
declare const published: typeof Published;
const publishedFromActual: typeof Published = actual;
const actualFromPublished: typeof Actual = published;
void publishedFromActual;
void actualFromPublished;
`, 'utf8');

  const options = {
    strict: true,
    noEmit: true,
    target: ts.ScriptTarget.ES2020,
    module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    skipLibCheck: false,
    types: [],
  };
  const program = ts.createProgram([contractFile], options);
  return ts.getPreEmitDiagnostics(program)
    .filter((diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error);
}

function mutateOnce(text, from, to, id) {
  assert.ok(text.includes(from), `${id}: mutation target no longer exists; update the control deliberately`);
  const changed = text.replace(from, to);
  assert.notEqual(changed, text, `${id}: mutation did not change the declaration`);
  return changed;
}

const tempRoot = mkdtempSync(join(tmpdir(), 'conformal-js-declarations-'));
try {
  const generated = generateImplementationDeclaration();
  const published = readFileSync(publishedFile, 'utf8');

  const generatedTypes = exportedTypeNames(generated, 'generated.d.ts');
  const publishedTypes = exportedTypeNames(published, 'published.d.ts');
  for (const name of generatedTypes) {
    assert.ok(publishedTypes.has(name), `Published declarations dropped implementation type export ${name}`);
  }

  const baseline = compatibilityDiagnostics(tempRoot, generated, published);
  assert.equal(baseline.length, 0,
    `Published declarations drift from implementation:\n${formatDiagnostics(baseline)}`);

  const readonlyBaseline = readonlySurfaceDrift(generated, published);
  assert.deepEqual(readonlyBaseline, [],
    `Published readonly/property surface drifts from implementation: ${readonlyBaseline.join('; ')}`);

  const returnMutation = mutateOnce(
    published,
    'export function brierScore(pHat: number, y: 0 | 1): number;',
    'export function brierScore(pHat: number, y: 0 | 1): string;',
    'return-type-drift-control',
  );
  const returnErrors = compatibilityDiagnostics(tempRoot, generated, returnMutation);
  assert.ok(returnErrors.some((diagnostic) => diagnostic.code === 2322),
    `Return-type drift control escaped detection:\n${formatDiagnostics(returnErrors)}`);

  const parameterMutation = mutateOnce(
    published,
    'export function calibrate(calibScores: NumericArray, alpha: number): number;',
    'export function calibrate(calibScores: NumericArray, alpha: string): number;',
    'parameter-drift-control',
  );
  const parameterErrors = compatibilityDiagnostics(tempRoot, generated, parameterMutation);
  assert.ok(parameterErrors.some((diagnostic) => diagnostic.code === 2322),
    `Parameter drift control escaped detection:\n${formatDiagnostics(parameterErrors)}`);

  const readonlyMutation = mutateOnce(
    published,
    'export interface CalibrationSummary {\n  sampleSize: number;',
    'export interface CalibrationSummary {\n  readonly sampleSize: number;',
    'readonly-drift-control',
  );
  const readonlyErrors = readonlySurfaceDrift(generated, readonlyMutation);
  assert.ok(readonlyErrors.some((message) => message.includes('CalibrationSummary.sampleSize')),
    `Readonly drift control escaped detection: ${readonlyErrors.join('; ')}`);

  console.log(JSON.stringify({
    status: 'PASS',
    compilerVersion: ts.version,
    generatedDeclarationBytes: Buffer.byteLength(generated),
    generatedTypeExportsRequired: [...generatedTypes].sort(),
    compatibility: 'bidirectional',
    injectedDeclarationDriftCaught: 3,
    checks: [
      'implementation-generated declarations compatible with shipped declarations',
      'implementation-generated type exports preserved',
      'brierScore return-type drift rejected',
      'calibrate parameter-type drift rejected',
      'CalibrationSummary readonly drift rejected',
    ],
  }, null, 2));
} catch (error) {
  console.error(`Declaration verification failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
} finally {
  rmSync(tempRoot, { recursive: true, force: true });
}
