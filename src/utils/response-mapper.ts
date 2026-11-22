import { OpenAIChatResponse, OpenAIMessage, StreamToolCallDelta, ToolCall } from '../types';

export function generateId(length = 29): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const bytes = new Uint8Array(length);

  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes);
  } else {
    for (let index = 0; index < length; index++) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }

  let result = '';
  for (const byte of bytes) result += chars[byte % chars.length];
  return result;
}

export function createOpenAIResponse(
  content: string,
  model: string,
  finishReason: 'stop' | 'length' | 'tool_calls' = 'stop',
  toolCalls?: ToolCall[]
): OpenAIChatResponse {
  const message: OpenAIMessage = {
    role: 'assistant',
    content: toolCalls && toolCalls.length > 0 && !content ? null : content,
  };

  if (toolCalls && toolCalls.length > 0) message.tool_calls = toolCalls;

  return {
    id: `chatcmpl-${generateId()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
  };
}

/** Delta for a single stream chunk */
export interface StreamDelta {
  role?: 'assistant';
  content?: string | null;
  tool_calls?: StreamToolCallDelta[];
}

/** Manages one OpenAI-compatible stream with a stable completion ID. */
export class StreamSession {
  readonly id: string;
  readonly created: number;
  private encoder = new TextEncoder();

  constructor(private model: string) {
    this.id = `chatcmpl-${generateId()}`;
    this.created = Math.floor(Date.now() / 1000);
  }

  chunk(
    delta: StreamDelta,
    finishReason: 'stop' | 'length' | 'tool_calls' | null = null,
    choiceIndex = 0
  ): Uint8Array {
    const payload = {
      id: this.id,
      object: 'chat.completion.chunk',
      created: this.created,
      model: this.model,
      choices: [{ index: choiceIndex, delta, finish_reason: finishReason }],
    };
    return this.encoder.encode(`data: ${JSON.stringify(payload)}\n\n`);
  }

  roleChunk(): Uint8Array {
    return this.chunk({ role: 'assistant', content: '' });
  }

  textChunk(text: string, choiceIndex = 0): Uint8Array {
    return this.chunk({ content: text }, null, choiceIndex);
  }

  toolCallStartChunk(
    index: number,
    callId: string,
    functionName: string,
    choiceIndex = 0
  ): Uint8Array {
    return this.chunk(
      {
        tool_calls: [
          {
            index,
            id: callId,
            type: 'function',
            function: { name: functionName, arguments: '' },
          },
        ],
      },
      null,
      choiceIndex
    );
  }

  toolCallArgsChunk(index: number, args: string, choiceIndex = 0): Uint8Array {
    return this.chunk(
      { tool_calls: [{ index, function: { arguments: args } }] },
      null,
      choiceIndex
    );
  }

  finishChunk(reason: 'stop' | 'tool_calls' | 'length' = 'stop', choiceIndex = 0): Uint8Array {
    return this.chunk({}, reason, choiceIndex);
  }

  done(): Uint8Array {
    return this.encoder.encode('data: [DONE]\n\n');
  }

  errorChunk(message: string): Uint8Array {
    return this.encoder.encode(
      `data: ${JSON.stringify({ error: { message, type: 'provider_stream_error' } })}\n\n`
    );
  }
}
