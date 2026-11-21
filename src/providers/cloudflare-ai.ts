import { BaseProvider } from './base';
import { OpenAIChatRequest, ProviderResponse, OpenAIMessage, Tool, ToolCall } from '../types';
import { createOpenAIResponse, StreamSession, generateId } from '../utils/response-mapper';

export class CloudflareAIProvider extends BaseProvider {
  constructor(
    model: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private aiBinding: any
  ) {
    super(model);
  }

  async chat(request: OpenAIChatRequest, _apiKey: string): Promise<ProviderResponse> {
    try {
      if (!this.aiBinding) {
        throw new Error('Cloudflare AI binding not available');
      }

      const messages = this.convertMessages(request.messages);

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const params: any = {
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: request.stream || false,
      };

      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }

      if (request.stream) {
        return this.handleStream(params);
      }
      return this.handleNonStream(params);
    } catch (error) {
      return this.handleError(error, 'CloudflareAIProvider');
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNonStream(params: any): Promise<ProviderResponse> {
    const response = await this.aiBinding.run(this.model, params);

    let content = '';
    if (typeof response === 'string') {
      content = response;
    } else if (typeof response?.response === 'string') {
      content = response.response;
    } else if (typeof response?.choices?.[0]?.message?.content === 'string') {
      // OpenAI-compatible responses: deliberately ignore reasoning_content.
      content = response.choices[0].message.content;
    }

    let toolCalls: ToolCall[] | undefined;
    let finishReason: 'stop' | 'tool_calls' = 'stop';
    if (response?.tool_calls && response.tool_calls.length > 0) {
      toolCalls = this.convertToolCalls(response.tool_calls);
      finishReason = 'tool_calls';
    } else if (response?.choices?.[0]?.message?.tool_calls?.length) {
      toolCalls = response.choices[0].message.tool_calls as ToolCall[];
      finishReason = 'tool_calls';
    }

    const openAIResponse = createOpenAIResponse(content, this.model, finishReason, toolCalls);
    return { success: true, response: openAIResponse };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleStream(params: any): Promise<ProviderResponse> {
    const cfStream = await this.aiBinding.run(this.model, params);
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const session = new StreamSession(this.model);

    (async () => {
      let hasToolCalls = false;
      const decoder = new TextDecoder();
      let pending = '';
      let sawSse = false;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const emitPayload = async (payload: any): Promise<void> => {
        if (!payload || typeof payload !== 'object') return;

        // Native Cloudflare Workers AI stream shape.
        if (typeof payload.response === 'string') {
          await writer.write(session.textChunk(payload.response));
        }
        if (Array.isArray(payload.tool_calls) && payload.tool_calls.length > 0) {
          hasToolCalls = true;
          for (let i = 0; i < payload.tool_calls.length; i++) {
            const tc = payload.tool_calls[i];
            const callId = `call_${generateId(24)}`;
            await writer.write(session.toolCallStartChunk(i, callId, tc.name));
            await writer.write(session.toolCallArgsChunk(i, JSON.stringify(tc.arguments ?? {})));
          }
        }

        // OpenAI-compatible SSE shape. Only delta.content is exposed;
        // delta.reasoning_content and all other metadata are intentionally ignored.
        if (Array.isArray(payload.choices)) {
          for (const choice of payload.choices) {
            const delta = choice?.delta;
            if (typeof delta?.content === 'string' && delta.content.length > 0) {
              await writer.write(session.textChunk(delta.content));
            }
            const calls = delta?.tool_calls;
            if (Array.isArray(calls) && calls.length > 0) {
              hasToolCalls = true;
              for (const call of calls) {
                const index = Number.isInteger(call.index) ? call.index : 0;
                const fn = call.function ?? {};
                if (call.id || fn.name) {
                  const callId = call.id || `call_${generateId(24)}`;
                  await writer.write(session.toolCallStartChunk(index, callId, fn.name || ''));
                }
                if (typeof fn.arguments === 'string' && fn.arguments.length > 0) {
                  await writer.write(session.toolCallArgsChunk(index, fn.arguments));
                }
              }
            }
          }
        }
      };

      const processEvent = async (event: string): Promise<void> => {
        const data = event
          .split(/\\r?\\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\\n');
        if (!data || data.trim() === '[DONE]') return;

        try {
          await emitPayload(JSON.parse(data));
        } catch (error) {
          // Malformed SSE must never be forwarded as assistant content.
          console.warn('[CloudflareAIProvider] Ignoring malformed SSE event:', error);
        }
      };

      const drainSse = async (flush = false): Promise<void> => {
        let boundary: RegExpExecArray | null;
        const separator = /\\r?\\n\\r?\\n/;
        while ((boundary = separator.exec(pending)) !== null) {
          const event = pending.slice(0, boundary.index);
          pending = pending.slice(boundary.index + boundary[0].length);
          sawSse = true;
          await processEvent(event);
        }
        if (flush && pending.trim()) {
          if (pending.includes('data:')) {
            sawSse = true;
            await processEvent(pending);
          } else if (!sawSse) {
            // Some bindings return plain text rather than SSE.
            await writer.write(session.textChunk(pending));
          }
          pending = '';
        }
      };

      try {
        await writer.write(session.roleChunk());
        const reader = cfStream.getReader();

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;

          if (typeof value === 'string') {
            if (value.startsWith('data:') || value.includes('\\n\\ndata:')) {
              pending += value;
              await drainSse();
            } else if (value && !sawSse) {
              await writer.write(session.textChunk(value));
            }
            continue;
          }

          if (value && typeof value === 'object' && !(value instanceof Uint8Array)) {
            await emitPayload(value);
            continue;
          }

          if (value instanceof Uint8Array) {
            pending += decoder.decode(value, { stream: true });
            if (pending.includes('data:')) {
              await drainSse();
            } else if (!sawSse && pending.length > 0 && !pending.includes('{')) {
              // Buffer ambiguous byte streams until the stream ends so split JSON/SSE
              // frames are never accidentally exposed as plain assistant text.
            }
          }
        }

        pending += decoder.decode();
        await drainSse(true);
        await writer.write(session.finishChunk(hasToolCalls ? 'tool_calls' : 'stop'));
        await writer.write(session.done());
      } catch (error) {
        console.error('[CloudflareAIProvider] Stream error:', error);
        try {
          await writer.write(session.finishChunk('stop'));
          await writer.write(session.done());
        } catch {
          // Writer may already be closed.
        }
      } finally {
        try {
          await writer.close();
        } catch {
          // Already closed.
        }
      }
    })();

    return { success: true, stream: readable };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private convertMessages(messages: OpenAIMessage[]): any[] {
    return messages.map((msg) => {
      if (msg.role === 'tool') {
        const toolContent =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
              : '';
        return { role: 'user', content: toolContent };
      }
      if (msg.role === 'assistant' && msg.tool_calls) {
        const assistantContent =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
              : '';
        return { role: 'assistant', content: assistantContent };
      }
      const textContent =
        typeof msg.content === 'string'
          ? msg.content || ''
          : msg.content
            ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
            : '';
      return {
        role: msg.role === 'system' ? 'user' : msg.role,
        content: textContent,
      };
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private convertTools(tools: Tool[]): any[] {
    return tools.map((tool) => ({
      type: 'function',
      function: {
        name: tool.function.name,
        description: tool.function.description || '',
        parameters: tool.function.parameters || { type: 'object', properties: {} },
      },
    }));
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private convertToolCalls(cfToolCalls: any[]): ToolCall[] {
    return cfToolCalls.map((call) => ({
      id: `call_${generateId(24)}`,
      type: 'function' as const,
      function: {
        name: call.name,
        arguments: JSON.stringify(call.arguments),
      },
    }));
  }
}
