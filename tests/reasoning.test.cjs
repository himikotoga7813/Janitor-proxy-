const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

const sourcePath = path.resolve(__dirname, '../src/utils/reasoning.ts');
const source = fs.readFileSync(sourcePath, 'utf8');
const compiled = ts.transpile(source, {
  module: ts.ModuleKind.CommonJS,
  target: ts.ScriptTarget.ES2022,
  esModuleInterop: true,
});

const loadedModule = new Module(sourcePath, module);
loadedModule.filename = sourcePath;
loadedModule.paths = Module._nodeModulePaths(path.dirname(sourcePath));
loadedModule._compile(compiled, sourcePath);

const {
  extractReasoningText,
  ReasoningStreamFormatter,
  normalizeAssistantMessage,
  normalizeOpenAIResponse,
} = loadedModule.exports;

test('stream formatter opens one think block across multiple reasoning chunks', () => {
  const formatter = new ReasoningStreamFormatter();

  assert.equal(formatter.pushReasoning('analysis A'), '<think>\nanalysis A');
  assert.equal(formatter.pushReasoning(' + analysis B'), ' + analysis B');
  assert.equal(formatter.pushContent('Final answer'), '\n</think>\nFinal answer');
  assert.equal(formatter.finish(), '');
});

test('stream formatter does not invent a think block when reasoning is absent', () => {
  const formatter = new ReasoningStreamFormatter();

  assert.equal(formatter.pushContent('Plain answer'), 'Plain answer');
  assert.equal(formatter.finish(), '');
});

test('stream formatter closes a reasoning-only stream on finish', () => {
  const formatter = new ReasoningStreamFormatter();

  assert.equal(formatter.pushReasoning('analysis'), '<think>\nanalysis');
  assert.equal(formatter.finish(), '\n</think>');
  assert.equal(formatter.finish(), '');
});

test('reasoning extraction reads nested text summaries but ignores opaque metadata', () => {
  const value = extractReasoningText(
    { summary: [{ type: 'summary_text', text: 'Part one' }] },
    { content: [{ text: ' and part two' }] },
    { signature: 'opaque-signature' }
  );

  assert.equal(value, 'Part one and part two');
});

test('non-stream normalization preserves image content parts', () => {
  const imagePart = {
    type: 'image_url',
    image_url: { url: 'data:image/png;base64,AAAA' },
  };
  const normalized = normalizeAssistantMessage({
    role: 'assistant',
    reasoning_content: 'brief reasoning',
    content: [imagePart],
  });

  assert.deepEqual(normalized.content, [
    { type: 'text', text: '<think>\nbrief reasoning\n</think>\n' },
    imagePart,
  ]);
  assert.equal('reasoning_content' in normalized, false);
});

test('completion normalization wraps reasoning in all assistant choices', () => {
  const normalized = normalizeOpenAIResponse({
    choices: [
      {
        index: 0,
        message: { role: 'assistant', reasoning: 'A', content: 'B' },
        finish_reason: 'stop',
      },
      {
        index: 1,
        message: { role: 'assistant', thinking: 'C', content: 'D' },
        finish_reason: 'stop',
      },
    ],
  });

  assert.equal(normalized.choices[0].message.content, '<think>\nA\n</think>\nB');
  assert.equal(normalized.choices[1].message.content, '<think>\nC\n</think>\nD');
});
