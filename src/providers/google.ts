import { GoogleGenAI, type Content, type Part } from '@google/genai';
import { BaseProvider } from './base';
import { ProxyError } from '../utils/error-handler';
import { OpenAIChatRequest, ProviderResponse, OpenAIMessage, ToolCall } from '../types';
import { createOpenAIResponse, StreamSession, generateId } from '../utils/response-mapper';
import { extractReasoningText, ReasoningStreamFormatter } from '../utils/reasoning';

/**
 * Thought signatures are required by Gemini 3 for function calling.
 * They are opaque metadata, not user-visible reasoning text.
 *
 * Format: tsig:<base64url_signature>:<randomId>
 */
const TSIG_PREFIX = 'tsig:';

function encodeToolCallId(signature?: string): string {
  const randomPart = generateId(12);
  if (!signature) return `call_${randomPart}`;
  const encoded = btoa(signature).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return `${TSIG_PREFIX}${encoded}:${randomPart}`;
}

function decodeThoughtSignature(toolCallId: string): string | undefined {
  if (!toolCallId.startsWith(TSIG_PREFIX)) return undefined;
  const rest = toolCallId.slice(TSIG_PREFIX.length);
  const colonIdx = rest.lastIndexOf(':');
  if (colonIdx === -1) return undefined;

  try {
    const encoded = rest.slice(0, colonIdx).replace(/-/g, '+').replace(/_/g, '/');
    const padded = encoded + '='.repeat((4 - (encoded.length % 4)) % 4);
    return atob(padded);
  } catch {
    return undefined;
  }
}

function parseToolArguments(value: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    throw new ProxyError(
      'Tool call arguments must contain valid JSON.',
      400,
      'invalid_tool_arguments'
    );
  }
}

export class GoogleProvider extends BaseProvider {
  private grounding: boolean;

  constructor(model: string, baseUrl?: string, grounding = false) {
    super(model, baseUrl);
    this.grounding = grounding;
  }

  async chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      if (request.n !== undefined && request.n > 1) {
        throw new ProxyError(
          'Google provider currently supports one candidate per request; the router may fall back to another configured provider.',
          501,
          'unsupported_candidate_count'
        );
      }
      const unsupportedOptions = [
        'logprobs',
        'top_logprobs',
        'user',
        'parallel_tool_calls',
        'stream_options',
        'reasoning_effort',
        'verbosity',
      ] as const;
      const unsupportedOption = unsupportedOptions.find(
        (key) => request[key] !== undefined
      );
      if (unsupportedOption) {
        throw new ProxyError(
          `Google provider does not support the "${unsupportedOption}" chat option through this adapter.`,
          501,
          'unsupported_request_option'
        );
      }

      const ai = new GoogleGenAI({ apiKey });

      const { systemInstruction, contents } = this.convertMessages(request.messages);

      const tools: Record<string, unknown>[] = [];

      if (request.tools && request.tools.length > 0) {
        tools.push({
          functionDeclarations: request.tools.map((tool) => ({
            name: tool.function.name,
            description: tool.function.description || '',
            parameters: tool.function.parameters || {},
          })),
        });
      }

      if (this.grounding) tools.push({ googleSearch: {} });

      const config: Record<string, unknown> = {};
      if (request.temperature !== undefined) config.temperature = request.temperature;
      if (request.max_tokens !== undefined || request.max_completion_tokens !== undefined) {
        config.maxOutputTokens = request.max_tokens ?? request.max_completion_tokens;
      }
      if (request.top_p !== undefined) config.topP = request.top_p;
      if (request.top_k !== undefined) config.topK = request.top_k;
      if (request.n !== undefined) config.candidateCount = request.n;
      if (request.frequency_penalty !== undefined) {
        config.frequencyPenalty = request.frequency_penalty;
      }
      if (request.presence_penalty !== undefined) {
        config.presencePenalty = request.presence_penalty;
      }
      if (request.seed !== undefined) config.seed = request.seed;
      if (request.response_format !== undefined) {
        const formatType = request.response_format.type;
        if (formatType === 'json_object') {
          config.responseMimeType = 'application/json';
        } else if (
          formatType === 'json_schema' &&
          request.response_format.json_schema &&
          typeof request.response_format.json_schema === 'object'
        ) {
          const schema = (
            request.response_format.json_schema as Record<string, unknown>
          ).schema;
          if (!schema || typeof schema !== 'object') {
            throw new ProxyError(
              'Google json_schema response_format requires a schema object.',
              400,
              'invalid_response_format'
            );
          }
          config.responseMimeType = 'application/json';
          config.responseSchema = schema;
        } else {
          throw new ProxyError(
            'Google supports response_format types json_object and json_schema only.',
            501,
            'unsupported_response_format'
          );
        }
      }
      if (request.stop) {
        config.stopSequences = Array.isArray(request.stop) ? request.stop : [request.stop];
      }

      if (systemInstruction) config.systemInstruction = systemInstruction;
      if (tools.length > 0) config.tools = tools;

      if (request.tool_choice) {
        const mode =
          request.tool_choice === 'none'
            ? 'NONE'
            : request.tool_choice === 'required'
              ? 'ANY'
              : 'AUTO';
        config.toolConfig = {
          functionCallingConfig: {
            mode,
            ...(typeof request.tool_choice === 'object' && request.tool_choice.type === 'function'
              ? { allowedFunctionNames: [request.tool_choice.function.name] }
              : {}),
          },
        };
      }

      if (request.stream) {
        return await this.handleStream(ai, contents, config);
      }
      return await this.handleNonStream(ai, contents, config);
    } catch (error) {
      return this.handleError(error, 'GoogleProvider');
    }
  }

  private async handleNonStream(
    ai: GoogleGenAI,
    contents: Content[],
    config: Record<string, unknown>
  ): Promise<ProviderResponse> {
    const response = await ai.models.generateContent({
      model: this.model,
      contents,
      config,
    });

    const parts = response.candidates?.[0]?.content?.parts as
      | (Part & Record<string, unknown>)[]
      | undefined;
    const reasoning = parts
      ?.filter((part) => part.thought === true && typeof part.text === 'string')
      .map((part) => part.text as string)
      .join('');
    const visibleText = parts
      ?.filter((part) => part.thought !== true && typeof part.text === 'string')
      .map((part) => part.text as string)
      .join('');
    const content = parts ? visibleText || '' : response.text || '';

    const toolParts = parts?.filter((part) => part.functionCall);
    let toolCalls: ToolCall[] | undefined;
    if (toolParts && toolParts.length > 0) {
      toolCalls = toolParts.map((part) => ({
        id: encodeToolCallId(part.thoughtSignature as string | undefined),
        type: 'function' as const,
        function: {
          name: part.functionCall?.name || '',
          arguments: JSON.stringify(part.functionCall?.args || {}),
        },
      }));
    }

    const formatter = new ReasoningStreamFormatter();
    const formattedContent =
      (reasoning ? formatter.pushReasoning(reasoning) : '') +
      formatter.pushContent(content) +
      formatter.finish();

    return {
      success: true,
      response: createOpenAIResponse(
        formattedContent,
        this.model,
        toolCalls ? 'tool_calls' : 'stop',
        toolCalls
      ),
    };
  }

  private async handleStream(
    ai: GoogleGenAI,
    contents: Content[],
    config: Record<string, unknown>
  ): Promise<ProviderResponse> {
    const response = await ai.models.generateContentStream({
      model: this.model,
      contents,
      config,
    });

    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const session = new StreamSession(this.model);
    const formatter = new ReasoningStreamFormatter();

    (async () => {
      try {
        await writer.write(session.roleChunk());
        let hasToolCalls = false;
        let toolCallIndex = 0;

        for await (const chunk of response) {
          const parts = chunk.candidates?.[0]?.content?.parts as
            | (Part & Record<string, unknown>)[]
            | undefined;

          if (parts) {
            const reasoning = extractReasoningText(
              parts
                .filter((part) => part.thought === true && typeof part.text === 'string')
                .map((part) => part.text)
            );
            const visibleText = extractReasoningText(
              parts
                .filter((part) => part.thought !== true && typeof part.text === 'string')
                .map((part) => part.text)
            );

            if (reasoning) {
              await writer.write(session.textChunk(formatter.pushReasoning(reasoning)));
            }
            if (visibleText) {
              await writer.write(session.textChunk(formatter.pushContent(visibleText)));
            }

            for (const part of parts) {
              if (!part.functionCall) continue;
              hasToolCalls = true;
              const callId = encodeToolCallId(part.thoughtSignature as string | undefined);
              await writer.write(
                session.toolCallStartChunk(toolCallIndex, callId, part.functionCall.name || '')
              );
              await writer.write(
                session.toolCallArgsChunk(
                  toolCallIndex,
                  JSON.stringify(part.functionCall.args || {})
                )
              );
              toolCallIndex++;
            }
          } else if (chunk.text) {
            await writer.write(session.textChunk(formatter.pushContent(chunk.text)));
          }
        }

        const closing = formatter.finish();
        if (closing) await writer.write(session.textChunk(closing));
        await writer.write(session.finishChunk(hasToolCalls ? 'tool_calls' : 'stop'));
        await writer.write(session.done());
      } catch (error) {
        console.error('[GoogleProvider] Stream error:', error);
        const message = error instanceof Error ? error.message : 'Google stream failed';
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

  /**
   * Convert OpenAI messages to Google GenAI Content format.
   * Preserves Gemini thought signatures from encoded tool call IDs.
   */
  private convertMessages(messages: OpenAIMessage[]): {
    systemInstruction?: string;
    contents: Content[];
  } {
    const systemParts: string[] = [];
    const contents: Content[] = [];

    const toolCallNameMap = new Map<string, string>();
    for (const message of messages) {
      for (const toolCall of message.tool_calls || []) {
        toolCallNameMap.set(toolCall.id, toolCall.function.name);
      }
    }

    for (const msg of messages) {
      if (msg.role === 'system' || msg.role === 'developer') {
        const text =
          typeof msg.content === 'string'
            ? msg.content
            : msg.content
              ? msg.content
                  .map((part) => {
                    if (part.type !== 'text') {
                      throw new ProxyError(
                        'Google system messages support text content only.',
                        501,
                        'unsupported_content_part'
                      );
                    }
                    return part.text;
                  })
                  .join(' ')
              : '';
        if (text) systemParts.push(text);
      } else if (msg.role === 'user') {
        const parts: Part[] = [];
        if (typeof msg.content === 'string') {
          parts.push({ text: msg.content || '' });
        } else if (msg.content) {
          for (const part of msg.content) {
            if (part.type === 'text') {
              parts.push({ text: part.text });
            } else if (part.type === 'image_url') {
              if (!part.image_url.url.startsWith('data:')) {
                throw new ProxyError(
                  'Google provider image inputs must use a base64 data URL; remote URLs are not fetched by the proxy.',
                  501,
                  'unsupported_image_source'
                );
              }
              const dataUri = part.image_url.url;
              const commaIdx = dataUri.indexOf(',');
              const base64Data = dataUri.slice(commaIdx + 1);
              if (commaIdx === -1 || !base64Data) {
                throw new ProxyError(
                  'Google image input contains an invalid base64 data URL.',
                  400,
                  'invalid_image_data'
                );
              }
              const header = dataUri.slice(0, commaIdx);
              const mimeMatch = header.match(/^data:([^;]+)/);
              const mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
              parts.push({ inlineData: { mimeType, data: base64Data } } as Part);
            } else {
              throw new ProxyError(
                `Google provider does not support content part type: ${String((part as { type?: unknown }).type)}`,
                501,
                'unsupported_content_part'
              );
            }
          }
        }
        if (parts.length > 0) contents.push({ role: 'user', parts });
      } else if (msg.role === 'assistant') {
        const parts: Part[] = [];

        if (msg.content) {
          if (typeof msg.content === 'string') {
            parts.push({ text: msg.content });
          } else {
            for (const part of msg.content) {
              if (part.type !== 'text') {
                throw new ProxyError(
                  'Google assistant messages support text content and function calls only.',
                  501,
                  'unsupported_content_part'
                );
              }
              parts.push({ text: part.text });
            }
          }
        }

        for (const toolCall of msg.tool_calls || []) {
          const part: Record<string, unknown> = {
            functionCall: {
              name: toolCall.function.name,
              args: parseToolArguments(toolCall.function.arguments),
            },
          };
          const signature = decodeThoughtSignature(toolCall.id);
          if (signature) part.thoughtSignature = signature;
          parts.push(part as Part);
        }

        if (parts.length > 0) contents.push({ role: 'model', parts });
      } else if (msg.role === 'tool' || msg.role === 'function') {
        const functionName =
          msg.role === 'function'
            ? msg.name
            : toolCallNameMap.get(msg.tool_call_id || '');
        const toolContent =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content
                  .map((part) => {
                    if (part.type !== 'text') {
                      throw new ProxyError(
                        'Google tool results support text content only.',
                        501,
                        'unsupported_content_part'
                      );
                    }
                    return part.text;
                  })
                  .join(' ')
              : '';

        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: functionName || 'unknown',
                response: { content: toolContent },
              },
            },
          ],
        });
      }
    }

    return {
      systemInstruction: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
      contents,
    };
  }
}
