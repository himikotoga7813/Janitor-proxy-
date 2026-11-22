/**
 * Example JavaScript/Node.js client using AI Worker Proxy.
 * Install the optional SDK example dependency with: npm install openai
 */

const BASE_URL = 'https://your-worker.workers.dev/v1';
const CHAT_URL = `${BASE_URL}/chat/completions`;
const API_KEY = 'your-secret-proxy-token-here';

async function assertOk(response) {
  if (response.ok) return;
  const details = await response.text();
  throw new Error(`Proxy request failed (${response.status}): ${details}`);
}

/** Read SSE frames without assuming that network chunks align with event boundaries. */
async function streamChat(response) {
  await assertOk(response);
  if (!response.body) throw new Error('The response does not contain a readable stream');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary;
    while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
      const frame = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);

      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n');

      if (!data) continue;
      if (data === '[DONE]') return;

      const payload = JSON.parse(data);
      if (payload.error) {
        throw new Error(payload.error.message || 'Streaming provider error');
      }

      const text = payload.choices?.[0]?.delta?.content;
      if (typeof text === 'string' && text.length > 0) process.stdout.write(text);
    }

    if (done) break;
  }

  // Process a final frame even if the upstream omitted the trailing blank line.
  if (buffer.trim()) {
    const data = buffer
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n');
    if (data && data !== '[DONE]') {
      const payload = JSON.parse(data);
      if (payload.error) throw new Error(payload.error.message || 'Streaming provider error');
      const text = payload.choices?.[0]?.delta?.content;
      if (typeof text === 'string' && text.length > 0) process.stdout.write(text);
    }
  }
}

async function streamingExample() {
  console.log('Example 1: Streaming response');

  const response = await fetch(CHAT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: 'deep-think',
      messages: [{ role: 'user', content: 'Tell me a short story about a robot.' }],
      stream: true,
    }),
  });

  await streamChat(response);
  console.log('\n');
}

async function nonStreamingExample() {
  console.log('Example 2: Non-streaming response');

  const response = await fetch(CHAT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: 'fast',
      messages: [{ role: 'user', content: 'What is 2+2?' }],
      stream: false,
    }),
  });

  await assertOk(response);
  const data = await response.json();
  console.log(data.choices?.[0]?.message?.content ?? data);
  console.log();
}

async function openaiSdkExample() {
  const OpenAI = require('openai');
  const client = new OpenAI({
    // The SDK appends /chat/completions; use the API base URL, not the endpoint URL.
    baseURL: BASE_URL,
    apiKey: API_KEY,
  });

  console.log('Example 3: Using OpenAI SDK');
  const stream = await client.chat.completions.create({
    model: 'deep-think',
    messages: [{ role: 'user', content: 'Count from 1 to 5.' }],
    stream: true,
  });

  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content;
    if (typeof text === 'string' && text.length > 0) process.stdout.write(text);
  }
  console.log('\n');
}

async function functionCallingExample() {
  console.log('Example 4: Function calling');

  const response = await fetch(CHAT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: 'deep-think',
      messages: [{ role: 'user', content: "What's the weather in London?" }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get current weather',
            parameters: {
              type: 'object',
              properties: { location: { type: 'string' } },
              required: ['location'],
            },
          },
        },
      ],
    }),
  });

  await assertOk(response);
  const data = await response.json();
  const toolCall = data.choices?.[0]?.message?.tool_calls?.[0];
  if (toolCall) {
    console.log('Function called:', toolCall.function.name);
    console.log('Arguments:', toolCall.function.arguments);
  } else {
    console.log(data.choices?.[0]?.message?.content ?? data);
  }
  console.log();
}

async function main() {
  await streamingExample();
  await nonStreamingExample();
  await functionCallingExample();
  // Uncomment to test the optional OpenAI SDK:
  // await openaiSdkExample();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
