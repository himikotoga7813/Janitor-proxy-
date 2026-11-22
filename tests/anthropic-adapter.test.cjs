const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

require.extensions['.ts'] = (module, filename) => {
  const source = fs.readFileSync(filename, 'utf8');
  const compiled = ts.transpile(source, {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
    esModuleInterop: true,
  });
  module._compile(compiled, filename);
};

const sourcePath = path.resolve(__dirname, '../src/utils/anthropic-adapter.ts');
const { AnthropicStreamAdapter, convertAnthropicRequestToOpenAI } = require(sourcePath);

function parseSSE(text) {
  return text
    .split(/\r?\n\r?\n/)
    .filter(Boolean)
    .map((frame) => {
      const event = frame.match(/^event: (.+)$/m)?.[1];
      const data = frame.match(/^data: (.+)$/m)?.[1];
      return data ? { event, data: JSON.parse(data) } : null;
    })
    .filter(Boolean);
}

async function transformOpenAIChunks(chunks) {
  const adapter = new AnthropicStreamAdapter('claude-test', 'msg_test');
  const input = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });

  const output = input.pipeThrough(adapter.createTransformStream());
  const reader = output.getReader();
  const decoder = new TextDecoder();
  let result = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    result += decoder.decode(value, { stream: true });
  }
  result += decoder.decode();
  return parseSSE(result);
}

test('Anthropic request conversion preserves system blocks, thinking, images and tool history', () => {
  const converted = convertAnthropicRequestToOpenAI({
    model: 'alias-model',
    system: [{ type: 'text', text: 'System instruction' }],
    messages: [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'Reasoning summary', signature: 'opaque' },
          {
            type: 'tool_use',
            id: 'call_weather',
            name: 'get_weather',
            input: { city: 'Tangier' },
          },
        ],
      },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_weather',
            content: 'Sunny',
          },
          {
            type: 'image',
            source: {
              type: 'url',
              url: 'https://example.invalid/image.png',
            },
          },
        ],
      },
    ],
  });

  assert.deepEqual(converted.messages[0], {
    role: 'system',
    content: 'System instruction',
  });
  assert.equal(converted.messages[1].reasoning_content, 'Reasoning summary');
  assert.equal(converted.messages[1].tool_calls[0].function.name, 'get_weather');
  assert.equal(converted.messages[2].role, 'tool');
  assert.equal(converted.messages[2].tool_call_id, 'call_weather');
  assert.equal(converted.messages[3].role, 'user');
  assert.deepEqual(converted.messages[3].content[0].image_url, {
    url: 'https://example.invalid/image.png',
  });
});

test('Anthropic stream adapter keeps interleaved parallel tool arguments separate', async () => {
  const chunks = [
    {
      id: 'chatcmpl-test',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'test-model',
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: 'call-a',
                type: 'function',
                function: { name: 'tool_a', arguments: '{"a":' },
              },
              {
                index: 1,
                id: 'call-b',
                type: 'function',
                function: { name: 'tool_b', arguments: '{"b":' },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    },
    {
      id: 'chatcmpl-test',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'test-model',
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              { index: 0, function: { arguments: '1}' } },
              { index: 1, function: { arguments: '2}' } },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
    },
  ];

  const events = await transformOpenAIChunks(chunks);
  const starts = events
    .filter((item) => item.event === 'content_block_start')
    .map((item) => item.data.content_block)
    .filter((block) => block.type === 'tool_use');
  const deltas = events
    .filter((item) => item.event === 'content_block_delta')
    .map((item) => item.data.delta)
    .filter((delta) => delta.type === 'input_json_delta');

  assert.deepEqual(
    starts.map((block) => [block.id, block.name]),
    [
      ['call-a', 'tool_a'],
      ['call-b', 'tool_b'],
    ]
  );
  assert.deepEqual(
    deltas.map((delta) => delta.partial_json),
    ['{"a":1}', '{"b":2}']
  );
  assert.equal(
    events.find((item) => item.event === 'message_delta').data.delta.stop_reason,
    'tool_use'
  );
});
