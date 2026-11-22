import { BaseProvider } from './base';
import { ProxyError } from '../utils/error-handler';
import { OpenAIChatRequest, ProviderResponse, OpenAIMessage, Tool, ToolCall } from '../types';
import { createOpenAIResponse, StreamSession, generateId } from '../utils/response-mapper';
import {
  extractReasoningText,
  normalizeOpenAIResponse,
  ReasoningStreamFormatter,
} from '../utils/reasoning';

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

      const unsupportedOptions = [
        'response_format',
        'seed',
        'logprobs',
        'top_logprobs',
        'user',
        'stream_options',
        'reasoning_effort',
        'verbosity',
      ] as const;
      const unsupportedOption = unsupportedOptions.find((key) => request[key] !== undefined);
      if (unsupportedOption) {
        throw new ProxyError(
          `Cloudflare AI adapter does not support the "${unsupportedOption}" chat option.`,
          501,
          'unsupported_request_option'
        );
      }

      const messages = this.convertMessages(request.messages);

      // Forward common OpenAI-compatible generation and tool options.
      const params: Record<string, unknown> = {
        messages,
        stream: request.stream ?? false,
      };
      const supportedOptions = [
        'max_tokens',
        'temperature',
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'stop',
        'tools',
        'tool_choice',
        'n',
        'parallel_tool_calls',
        'top_k',
      ] as const;

      for (const key of supportedOptions) {
        if (request[key] !== undefined) {
          params[key] =
            key === 'tools' && request.tools ? this.convertTools(request.tools) : request[key];
        }
      }

      if (params.max_tokens === undefined && request.max_completion_tokens !== undefined) {
        params.max_tokens = request.max_completion_tokens;
      }

      if (request.stream) {
        return await this.handleStream(params);
      }
      return await this.handleNonStream(params);
    } catch (error) {
      return this.handleError(error, 'CloudflareAIProvider');
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNonStream(params: any): Promise<ProviderResponse> {
    const response = await this.aiBinding.run(this.model, params);

    if (response?.choices?.length) {
      const normalized = normalizeOpenAIResponse(response);
      return {
        success: true,
        response: normalized as ProviderResponse['response'],
      };
    }

    let content = '';
    if (typeof response === 'string') {
      content = response;
    } else if (typeof response?.response === 'string') {
      const reasoning = extractReasoningText(
        response.reasoning_content,
        response.reasoning,
        response.thinking,
        response.reasoning_details
      );
      const formatter = new ReasoningStreamFormatter();
      content =
        (reasoning ? formatter.pushReasoning(reasoning) : '') +
        formatter.pushContent(response.response) +
        formatter.finish();
    }

    let toolCalls: ToolCall[] | undefined;
    let finishReason: 'stop' | 'tool_calls' = 'stop';
    if (Array.isArray(response?.tool_calls) && response.tool_calls.length > 0) {
      toolCalls = this.convertToolCalls(response.tool_calls);
      finishReason = 'tool_calls';
    }

    return {
      success: true,
      response: createOpenAIResponse(content, this.model, finishReason, toolCalls),
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleStream(params: any): Promise<ProviderResponse> {
    const cfStream = await this.aiBinding.run(this.model, params);
    if (!cfStream || typeof cfStream.getReader !== 'function') {
      throw new Error('Cloudflare AI did not return a readable stream');
    }

    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const session = new StreamSession(this.model);
    const formatters = new Map<number, ReasoningStreamFormatter>();
    const startedToolCalls = new Set<string>();
    const toolCallIds = new Map<string, string>();
    const startedChoices = new Set<number>([0]);
    const finishReasons = new Map<number, 'stop' | 'tool_calls' | 'length'>();
    const toolCallChoices = new Set<number>();
    let nativeToolIndex = 0;

    const formatterFor = (index: number): ReasoningStreamFormatter => {
      let formatter = formatters.get(index);
      if (!formatter) {
        formatter = new ReasoningStreamFormatter();
        formatters.set(index, formatter);
      }
      return formatter;
    };

    (async () => {
      const decoder = new TextDecoder();

      const ensureChoiceStarted = async (choiceIndex: number): Promise<void> => {
        if (startedChoices.has(choiceIndex)) return;
        startedChoices.add(choiceIndex);
        await writer.write(session.chunk({ role: 'assistant', content: '' }, null, choiceIndex));
      };
      let pending = '';
      let sawSse = false;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const emitPayload = async (payload: any): Promise<void> => {
        if (!payload || typeof payload !== 'object') return;
        if (payload.error) {
          const providerMessage =
            typeof payload.error === 'string'
              ? payload.error
              : typeof payload.error.message === 'string'
                ? payload.error.message
                : 'Cloudflare AI returned a provider error';
          throw new Error(providerMessage);
        }

        const rootReasoning = extractReasoningText(
          payload.reasoning_content,
          payload.reasoning,
          payload.thinking,
          payload.reasoning_details
        );
        const recognized =
          typeof payload.response === 'string' ||
          (Array.isArray(payload.choices) && payload.choices.length > 0) ||
          (Array.isArray(payload.tool_calls) && payload.tool_calls.length > 0) ||
          Boolean(rootReasoning);
        if (!recognized) throw new Error('Cloudflare AI returned an unsupported stream payload');

        const rootFormatter = formatterFor(0);
        if (rootReasoning)
          await writer.write(session.textChunk(rootFormatter.pushReasoning(rootReasoning)));

        // Native Cloudflare Workers AI response shape.
        if (typeof payload.response === 'string') {
          await writer.write(session.textChunk(rootFormatter.pushContent(payload.response)));
        }

        if (Array.isArray(payload.tool_calls) && payload.tool_calls.length > 0) {
          toolCallChoices.add(0);
          for (const tc of payload.tool_calls) {
            const index = nativeToolIndex++;
            const callId = `call_${generateId(24)}`;
            await writer.write(session.toolCallStartChunk(index, callId, tc.name || ''));
            await writer.write(
              session.toolCallArgsChunk(index, JSON.stringify(tc.arguments ?? {}))
            );
          }
        }

        // OpenAI-compatible response/SSE shape.
        if (Array.isArray(payload.choices)) {
          for (const choice of payload.choices) {
            const index = Number.isInteger(choice?.index) ? choice.index : 0;
            const delta = choice?.delta ?? {};
            const formatter = formatterFor(index);
            const reasoning = extractReasoningText(
              delta.reasoning_content,
              delta.reasoning,
              delta.thinking,
              delta.reasoning_details
            );
            if (reasoning) {
              await ensureChoiceStarted(index);
              await writer.write(session.textChunk(formatter.pushReasoning(reasoning), index));
            }
            if (typeof delta.content === 'string' && delta.content.length > 0) {
              await ensureChoiceStarted(index);
              await writer.write(session.textChunk(formatter.pushContent(delta.content), index));
            }

            if (choice?.finish_reason) {
              const finishReason =
                choice.finish_reason === 'tool_calls'
                  ? 'tool_calls'
                  : choice.finish_reason === 'length'
                    ? 'length'
                    : 'stop';
              finishReasons.set(index, finishReason);
              const closing = formatter.finish();
              if (closing) {
                await ensureChoiceStarted(index);
                await writer.write(session.textChunk(closing, index));
              }
            }

            const calls = delta.tool_calls;
            if (Array.isArray(calls) && calls.length > 0) {
              toolCallChoices.add(index);
              for (const call of calls) {
                const callIndex = Number.isInteger(call.index) ? call.index : 0;
                const fn = call.function ?? {};
                const toolKey = `${index}:${callIndex}`;
                const existingCallId = toolCallIds.get(toolKey);
                const generatedCallId =
                  typeof call.id === 'string' && call.id.length > 0
                    ? call.id
                    : `call_${generateId(24)}`;
                const callId: string = existingCallId || generatedCallId;
                toolCallIds.set(toolKey, callId);

                const functionName: string = typeof fn.name === 'string' ? fn.name : '';
                if (!startedToolCalls.has(toolKey) && (call.id || functionName)) {
                  startedToolCalls.add(toolKey);
                  await ensureChoiceStarted(index);
                  await writer.write(
                    session.toolCallStartChunk(callIndex, callId, functionName, index)
                  );
                }

                const argumentsDelta: string = typeof fn.arguments === 'string' ? fn.arguments : '';
                if (argumentsDelta.length > 0) {
                  await ensureChoiceStarted(index);
                  await writer.write(session.toolCallArgsChunk(callIndex, argumentsDelta, index));
                }
              }
            }
          }
        }
      };

      const processEvent = async (event: string): Promise<void> => {
        const data = event
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (!data || data.trim() === '[DONE]') return;

        try {
          await emitPayload(JSON.parse(data));
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Invalid Cloudflare AI stream event';
          throw new Error(message);
        }
      };

      const drainSse = async (flush = false): Promise<void> => {
        const separator = /\r?\n\r?\n/;
        let boundary: RegExpExecArray | null;
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
            const formatter = formatterFor(0);
            await writer.write(session.textChunk(formatter.pushContent(pending)));
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
            if (value.startsWith('data:') || value.includes('\n\ndata:')) {
              pending += value;
              await drainSse();
            } else if (value && !sawSse) {
              await writer.write(session.textChunk(formatterFor(0).pushContent(value)));
            }
            continue;
          }

          if (value && typeof value === 'object' && !(value instanceof Uint8Array)) {
            await emitPayload(value);
            continue;
          }

          if (value instanceof Uint8Array) {
            pending += decoder.decode(value, { stream: true });
            if (pending.includes('data:')) await drainSse();
          }
        }

        pending += decoder.decode();
        await drainSse(true);

        const choiceIndices = new Set<number>([
          0,
          ...formatters.keys(),
          ...finishReasons.keys(),
          ...toolCallChoices,
        ]);
        for (const choiceIndex of choiceIndices) {
          const closing = formatterFor(choiceIndex).finish();
          if (closing) {
            await ensureChoiceStarted(choiceIndex);
            await writer.write(session.textChunk(closing, choiceIndex));
          }
          const finishReason =
            finishReasons.get(choiceIndex) ||
            (toolCallChoices.has(choiceIndex) ? 'tool_calls' : 'stop');
          await writer.write(session.finishChunk(finishReason, choiceIndex));
        }
        await writer.write(session.done());
      } catch (error) {
        console.error(
          '[CloudflareAIProvider] Stream failed:',
          error instanceof Error ? error.message : 'Unknown stream error'
        );
        const message = error instanceof Error ? error.message : 'Cloudflare AI stream failed';
        try {
          await writer.write(
            new TextEncoder().encode(
              `data: ${JSON.stringify({ error: { message, type: 'provider_stream_error' } })}\n\n`
            )
          );
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

  // Preserve OpenAI-compatible message blocks instead of flattening away images/tool history.
  private convertMessages(messages: OpenAIMessage[]): Array<Record<string, unknown>> {
    return messages.map((msg) => {
      if (msg.role === 'function') {
        return {
          role: 'tool',
          name: msg.name,
          tool_call_id: msg.tool_call_id,
          content: msg.content ?? '',
        };
      }

      const converted: Record<string, unknown> = {
        role: msg.role === 'developer' ? 'system' : msg.role,
        content: msg.content ?? '',
      };

      if (msg.name) converted.name = msg.name;
      if (msg.tool_calls) converted.tool_calls = msg.tool_calls;
      if (msg.tool_call_id) converted.tool_call_id = msg.tool_call_id;
      return converted;
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
      id: call.id || `call_${generateId(24)}`,
      type: 'function' as const,
      function: {
        name: call.name,
        arguments:
          typeof call.arguments === 'string'
            ? call.arguments
            : JSON.stringify(call.arguments ?? {}),
      },
    }));
  }
}
