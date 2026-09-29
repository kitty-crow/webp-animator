import { describe, expect, test } from 'bun:test';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import ts from 'typescript';

const root = join(import.meta.dir, '..');
const sourceRoot = join(root, 'src');

const requiredStrictFlags = [
  'strict',
  'noImplicitAny',
  'strictNullChecks',
  'strictFunctionTypes',
  'strictBindCallApply',
  'strictPropertyInitialization',
  'useUnknownInCatchVariables',
  'noUncheckedIndexedAccess',
  'exactOptionalPropertyTypes',
  'noImplicitOverride',
  'noImplicitReturns',
  'noFallthroughCasesInSwitch',
  'noPropertyAccessFromIndexSignature',
] as const;

function walk(node: ts.Node, fileName: string, failures: string[]): void {
  if (node.kind === ts.SyntaxKind.AnyKeyword) {
    const location = node.getSourceFile().getLineAndCharacterOfPosition(node.getStart());
    failures.push(`${fileName}:${location.line + 1}:${location.character + 1} explicit any type`);
  }
  ts.forEachChild(node, (child: ts.Node) => walk(child, fileName, failures));
}

describe('strict browser typing regression gate', () => {
  test('tsconfig cannot loosen strict compiler checks', async () => {
    const raw = JSON.parse(await readFile(join(root, 'tsconfig.json'), 'utf8')) as unknown;
    if (typeof raw !== 'object' || raw === null || !('compilerOptions' in raw)) throw new Error('tsconfig compilerOptions missing.');
    const compilerOptions = (raw as { readonly compilerOptions: unknown }).compilerOptions;
    if (typeof compilerOptions !== 'object' || compilerOptions === null) throw new Error('tsconfig compilerOptions invalid.');
    const record = compilerOptions as Record<string, unknown>;
    for (const flag of requiredStrictFlags) expect(record[flag], `${flag} must stay enabled`).toBe(true);
    expect(record.skipLibCheck, 'library checks must not be silently skipped').toBe(false);
  });

  test('browser sources contain no explicit any or type-check suppression', async () => {
    const glob = new Bun.Glob('**/*.ts');
    const failures: string[] = [];
    for await (const relativePath of glob.scan({ cwd: sourceRoot, onlyFiles: true })) {
      const fullPath = join(sourceRoot, relativePath);
      const source = await readFile(fullPath, 'utf8');
      if (source.includes('@ts-ignore')) failures.push(`${relativePath}: @ts-ignore is forbidden`);
      if (source.includes('@ts-nocheck')) failures.push(`${relativePath}: @ts-nocheck is forbidden`);
      const tree = ts.createSourceFile(relativePath, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
      walk(tree, relativePath, failures);
    }
    expect(failures).toEqual([]);
  });
});
