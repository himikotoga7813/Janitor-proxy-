import Anthropic from '@anthropic-ai/sdk';
import { BaseProvider } from './base';
import { AnthropicRequest } from '../anthropic-types';
import { OpenAIChatRequest, ProviderResponse, OpenAIMessage, ToolCall } from '../types';
import { createOpenAIResponse, StreamSession } from '../utils/response-mapper';
import { ReasoningStreamFormatter } from '../utils/reasoning';
import { ProxyError } from '../utils/error-handler';

export class AnthropicProvider extends BaseProvider {
  async chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      if (request.n !== undefined && request.n > 1) {
        throw new ProxyError(
          'Anthropic provider supports one completion per request; the router may fall back to another configured provider.',
          501,
          'unsupported_candidate_count'
        );
      }
      const clientOpts: Record<string, unknown> = { apiKey };
      if (this.baseUrl) clientOpts.baseURL = this.baseUrl;
      const client = new Anthropic(clientOpts);

      const { system, messages } = this.convertMessages(request.messages);
      const tools = request.tools?.map((tool) => ({
        name: tool.function.name,
        description: tool.function.description || '',
        input_schema: tool.function.parameters || {},
      }));

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const params: any = {
        model: this.model,
        messages,
        max_tokens: request.max_tokens ?? request.max_completion_tokens ?? 4096,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: request.stream ?? false,
      };

      if (system) params.system = system;
      if (tools?.length) params.tools = tools;
      if (request.stop) {
        params.stop_sequences = Array.isArray(request.stop) ? request.stop : [request.stop];
      }

      if (request.tool_choice) {
        if (request.tool_choice === 'none') {
          params.tool_choice = { type: 'none' };
        } else if (request.tool_choice === 'auto' || request.tool_choice === 'required') {
          params.tool_choice =
            request.tool_choice === 'required' ? { type: 'any' } : { type: 'auto' };
        } else if (
          typeof request.tool_choice === 'object' &&
          request.tool_choice.type === 'function'
        ) {
          params.tool_choice = {
            type: 'tool',
            name: request.tool_choice.function.name,
          };
        }
      }

      if (request.stream) return await this.handleStream(client, params);
      return await this.handleNonStream(client, params);
    } catch (error) {
      return this.handleError(error, 'AnthropicProvider');
    }
  }

  /**
   * Anthropic-native chat — bypasses OpenAI format conversion.
   * The native endpoint deliberately keeps Anthropic's own thinking/event schema.
   */
  async nativeChat(request: AnthropicRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      const clientOpts: Record<string, unknown> = { apiKey };
      if (this.baseUrl) clientOpts.baseURL = this.baseUrl;
      const client = new Anthropic(clientOpts);

      const params: Record<string, unknown> = {
        model: request.model,
        messages: request.messages,
        max_tokens: request.max_tokens ?? 4096,
        stream: request.stream ?? false,
      };

      if (request.system) params.system = request.system;
      if (request.temperature !== undefined) params.temperature = request.temperature;
      if (request.top_p !== undefined) params.top_p = request.top_p;
      if (request.top_k !== undefined) params.top_k = request.top_k;
      if (request.top_k !== undefined) params.top_k = request.top_k;
      if (request.stop_sequences) params.stop_sequences = request.stop_sequences;
      if (request.metadata) params.metadata = request.metadata;
      if (request.tools?.length) params.tools = request.tools;
      if (request.tool_choice) params.tool_choice = request.tool_choice;

      if (request.stream) return await this.handleNativeStream(client, params);
      return await this.handleNativeNonStream(client, params);
    } catch (error) {
      return this.handleError(error, 'AnthropicProvider.nativeChat');
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNonStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    const response = await client.messages.create(params);
    const formatter = new ReasoningStreamFormatter();
    let reasoning = '';
    let content = '';
    let toolCalls: ToolCall[] | undefined;

    for (const block of response.content) {
      const record = block as unknown as Record<string, unknown>;
      if (block.type === 'text') {
        content += block.text;
      } else if (block.type === 'tool_use') {
        if (!toolCalls) toolCalls = [];
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input),
          },
        });
      } else if (record.type === 'thinking' && typeof record.thinking === 'string') {
        reasoning += record.thinking;
      }
    }

    const formattedContent =
      (reasoning ? formatter.pushReasoning(reasoning) : '') +
      formatter.pushContent(content) +
      formatter.finish();
    const finishReason = response.stop_reason === 'tool_use' ? 'tool_calls' : 'stop';

    return {
      success: true,
      response: createOpenAIResponse(formattedContent, this.model, finishReason, toolCalls),
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    const stream = await client.messages.stream(params);
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const session = new StreamSession(this.model);
    const formatter = new ReasoningStreamFormatter();

    (async () => {
      try {
        await writer.write(session.roleChunk());
        let toolCallIndex = -1;
        let hasToolCalls = false;
        let ended = false;

        for await (const event of stream) {
          if (event.type === 'content_block_start') {
            const block = event.content_block as unknown as Record<string, unknown>;
            if (block.type === 'tool_use') {
              toolCallIndex++;
              hasToolCalls = true;
              await writer.write(
                session.toolCallStartChunk(
                  toolCallIndex,
                  String(block.id || ''),
                  String(block.name || '')
                )
              );
            }
          } else if (event.type === 'content_block_delta') {
            const delta = event.delta as unknown as Record<string, unknown>;
            if (delta.type === 'text_delta' && typeof delta.text === 'string') {
              await writer.write(session.textChunk(formatter.pushContent(delta.text)));
            } else if (delta.type === 'thinking_delta' && typeof delta.thinking === 'string') {
              await writer.write(session.textChunk(formatter.pushReasoning(delta.thinking)));
            } else if (
              delta.type === 'input_json_delta' &&
              typeof delta.partial_json === 'string' &&
              toolCallIndex >= 0
            ) {
              await writer.write(session.toolCallArgsChunk(toolCallIndex, delta.partial_json));
            }
          } else if (event.type === 'message_stop') {
            const closing = formatter.finish();
            if (closing) await writer.write(session.textChunk(closing));
            await writer.write(session.finishChunk(hasToolCalls ? 'tool_calls' : 'stop'));
            await writer.write(session.done());
            ended = true;
          }
        }

        if (!ended) {
          const closing = formatter.finish();
          if (closing) await writer.write(session.textChunk(closing));
          await writer.write(session.finishChunk(hasToolCalls ? 'tool_calls' : 'stop'));
          await writer.write(session.done());
        }
      } catch (error) {
        console.error('[AnthropicProvider] Stream error:', error);
        const message = error instanceof Error ? error.message : 'Anthropic stream failed';
        try {
          await writer.write(
            new TextEncoder().encode(
              `data: ${JSON.stringify({ error: { message, type: 'provider_stream_error' } })}\n\n`
            )
          );
          await writer.write(session.done());
        } catch {
          // Downstream may have cancelled.
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

  /** Forward Anthropic SDK Message directly for the Anthropic-native endpoint. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNativeNonStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    const response = await client.messages.create(params);
    return { success: true, rawResponse: response };
  }

  /** Forward native Anthropic stream events unchanged, including thinking blocks and signatures. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNativeStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    const stream = await client.messages.stream(params);
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    (async () => {
      try {
        for await (const event of stream) {
          const line = `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
          await writer.write(encoder.encode(line));
        }
      } catch (error) {
        console.error('[AnthropicProvider] Native stream error:', error);
        const message = error instanceof Error ? error.message : 'Anthropic stream failed';
        try {
          const errorEvent = {
            type: 'error',
            error: { type: 'api_error', message },
          };
          await writer.write(
            encoder.encode(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`)
          );
        } catch {
          // Downstream may have cancelled.
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

  private convertMessages(messages: OpenAIMessage[]): {
    system?: string;
    messages: Anthropic.MessageParam[];
  } {
    const systemParts: string[] = [];
    const convertedMessages: Anthropic.MessageParam[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        const text =
          typeof msg.content === 'string'
            ? msg.content
            : msg.content
              ? msg.content.map((part) => {
                  if (part.type !== 'text') {
                    throw new ProxyError(
                      'Anthropic system messages support text content only.',
                      501,
                      'unsupported_content_part'
                    );
                  }
                  return part.text;
                }).join(' ')
              : '';
        if (text) systemParts.push(text);
      } else if (msg.role === 'user' || msg.role === 'assistant') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const content: any[] = [];

        if (typeof msg.content === 'string') {
          if (msg.content) content.push({ type: 'text', text: msg.content });
        } else if (msg.content) {
          for (const part of msg.content) {
            if (part.type === 'text') {
              content.push({ type: 'text', text: part.text });
            } else if (part.type === 'image_url') {
              if (!part.image_url.url.startsWith('data:')) {
                throw new ProxyError(
                  'Anthropic-compatible image inputs must use a base64 data URL; remote URLs are not fetched by the proxy.',
                  400,
                  'unsupported_image_source'
                );
              }
              const dataUri = part.image_url.url;
              const commaIdx = dataUri.indexOf(',');
              if (commaIdx !== -1) {
                const header = dataUri.slice(0, commaIdx);
                const base64Data = dataUri.slice(commaIdx + 1);
                const mimeMatch = header.match(/^data:([^;]+)/);
                const mediaType = mimeMatch ? mimeMatch[1] : 'image/png';
                content.push({
                  type: 'image',
                  source: { type: 'base64', media_type: mediaType, data: base64Data },
                });
              }
            }
          }
        }

        if (msg.tool_calls) {
          for (const toolCall of msg.tool_calls) {
            let parsedInput: unknown;
            try {
              parsedInput = JSON.parse(toolCall.function.arguments);
            } catch {
              parsedInput = {};
            }
            content.push({
              type: 'tool_use',
              id: toolCall.id,
              name: toolCall.function.name,
              input: parsedInput,
            });
          }
        }

        if (content.length > 0) convertedMessages.push({ role: msg.role, content });
      } else if (msg.role === 'tool') {
        const toolContent =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content.map((part) => (part.type === 'text' ? part.text : '')).join(' ')
              : '';

        if (msg.tool_call_id) {
          convertedMessages.push({
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: msg.tool_call_id,
                content: toolContent,
              },
            ],
          });
        } else {
          convertedMessages.push({ role: 'user', content: toolContent });
        }
      }
    }

    return {
      system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
      messages: convertedMessages,
    };
  }
}
