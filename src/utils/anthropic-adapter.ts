import { AnthropicRequest, AnthropicResponse, AnthropicContentBlock } from '../anthropic-types';
import { ProxyError } from './error-handler';
import {
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIMessage,
  ContentPart,
  ToolCall,
} from '../types';

/** Convert an Anthropic request into the common OpenAI-compatible request format. */
function convertAnthropicImage(block: AnthropicContentBlock): ContentPart {
  const source = block.source;
  if (!source) {
    throw new ProxyError('Anthropic image block is missing its source.', 400, 'invalid_image_block');
  }

  if (source.type === 'base64') {
    if (!source.data || !source.media_type) {
      throw new ProxyError('Anthropic base64 image source is incomplete.', 400, 'invalid_image_block');
    }
    return {
      type: 'image_url',
      image_url: {
        url: `data:${source.media_type};base64,${source.data}`,
      },
    };
  }

  if (source.type === 'url') {
    if (!source.url) {
      throw new ProxyError('Anthropic URL image source is empty.', 400, 'invalid_image_block');
    }
    return {
      type: 'image_url',
      image_url: { url: source.url },
    };
  }

  throw new ProxyError('Unsupported Anthropic image source type.', 501, 'unsupported_image_source');
}

function convertAnthropicToolResult(
  value: string | AnthropicContentBlock[] | undefined
): OpenAIMessage['content'] {
  if (typeof value === 'string') return value;
  if (!value) return '';

  const parts: ContentPart[] = [];
  for (const block of value) {
    if (block.type === 'text') {
      if (typeof block.text !== 'string') {
        throw new ProxyError('Anthropic tool result contains an invalid text block.', 400, 'invalid_tool_result');
      }
      parts.push({ type: 'text', text: block.text });
    } else if (block.type === 'image') {
      parts.push(convertAnthropicImage(block));
    } else {
      throw new ProxyError(
        `Anthropic tool result block "${String(block.type)}" cannot be represented as an OpenAI tool result.`,
        501,
        'unsupported_tool_result_content'
      );
    }
  }

  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text;
  return parts;
}

/** Convert an Anthropic request into the common OpenAI-compatible request format. */
export function convertAnthropicRequestToOpenAI(anthropicReq: AnthropicRequest): OpenAIChatRequest {
  const messages: OpenAIMessage[] = [];

  const systemText =
    typeof anthropicReq.system === 'string'
      ? anthropicReq.system
      : anthropicReq.system?.map((block) => block.text).join('\n\n');
  if (systemText) messages.push({ role: 'system', content: systemText });

  for (const msg of anthropicReq.messages) {
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
      continue;
    }

    const contentParts: ContentPart[] = [];
    const toolCalls: ToolCall[] = [];
    const toolResults: OpenAIMessage[] = [];
    let reasoningText = '';

    for (const block of msg.content) {
      if (block.type === 'text') {
        if (typeof block.text !== 'string') {
          throw new ProxyError('Anthropic text block is missing text.', 400, 'invalid_text_block');
        }
        if (block.text) contentParts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image') {
        contentParts.push(convertAnthropicImage(block));
      } else if (block.type === 'tool_result') {
        if (!block.tool_use_id) {
          throw new ProxyError(
            'Anthropic tool_result blocks require tool_use_id.',
            400,
            'invalid_tool_result'
          );
        }
        const resultContent = convertAnthropicToolResult(block.content);
        const prefix =
          block.is_error === true
            ? '[Tool returned an error]\n'
            : '';
        const content =
          prefix && typeof resultContent === 'string'
            ? prefix + resultContent
            : prefix && Array.isArray(resultContent)
              ? [{ type: 'text', text: prefix } as ContentPart, ...resultContent]
              : resultContent;
        toolResults.push({
          role: 'tool',
          tool_call_id: block.tool_use_id,
          content,
        });
      } else if (block.type === 'tool_use') {
        if (!block.id || !block.name) {
          throw new ProxyError(
            'Anthropic tool_use blocks require both an id and a name.',
            400,
            'invalid_tool_use'
          );
        }
        if (
          block.input !== undefined &&
          (!block.input || typeof block.input !== 'object' || Array.isArray(block.input))
        ) {
          throw new ProxyError('Anthropic tool_use input must be a JSON object.', 400, 'invalid_tool_use');
        }
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      } else if (block.type === 'thinking') {
        if (msg.role !== 'assistant' || typeof block.thinking !== 'string') {
          throw new ProxyError('Invalid Anthropic thinking block.', 400, 'invalid_thinking_block');
        }
        reasoningText += block.thinking;
      } else if (block.type === 'redacted_thinking') {
        throw new ProxyError(
          'Redacted Anthropic thinking blocks cannot be converted to an OpenAI-compatible provider.',
          501,
          'unsupported_thinking_block'
        );
      } else {
        throw new ProxyError(
          `Anthropic content block type "${String(block.type)}" cannot be converted to OpenAI chat format.`,
          501,
          'unsupported_content_block'
        );
      }
    }

    // Anthropic tool results must immediately follow the assistant tool-call message.
    // Emit those tool messages before any extra user text carried in the same content array.
    if (toolResults.length > 0) messages.push(...toolResults);

    let content: OpenAIMessage['content'] = null;
    if (contentParts.length === 1 && contentParts[0].type === 'text') {
      content = contentParts[0].text;
    } else if (contentParts.length > 0) {
      content = contentParts;
    }

    if (content !== null || toolCalls.length > 0 || reasoningText) {
      messages.push({
        role: msg.role,
        content,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        ...(reasoningText ? { reasoning_content: reasoningText } : {}),
      });
    }
  }

  const openAIReq: OpenAIChatRequest = {
    model: anthropicReq.model,
    messages,
    max_tokens: anthropicReq.max_tokens ?? 4096,
    stream: anthropicReq.stream ?? false,
  };

  if (anthropicReq.temperature !== undefined) openAIReq.temperature = anthropicReq.temperature;
  if (anthropicReq.top_p !== undefined) openAIReq.top_p = anthropicReq.top_p;
  if (anthropicReq.top_k !== undefined) openAIReq.top_k = anthropicReq.top_k;
  if (anthropicReq.stop_sequences) openAIReq.stop = anthropicReq.stop_sequences;

  if (anthropicReq.tools?.length) {
    openAIReq.tools = anthropicReq.tools.map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description || '',
        parameters: tool.input_schema || {},
      },
    }));
  }

  if (anthropicReq.tool_choice) {
    if (anthropicReq.tool_choice.type === 'auto') {
      openAIReq.tool_choice = 'auto';
    } else if (anthropicReq.tool_choice.type === 'any') {
      openAIReq.tool_choice = 'required';
    } else if (anthropicReq.tool_choice.type === 'tool' && anthropicReq.tool_choice.name) {
      openAIReq.tool_choice = {
        type: 'function',
        function: { name: anthropicReq.tool_choice.name },
      };
    }
  }

  return openAIReq;
}

function mapFinishReason(
  reason: string | null | undefined
): 'end_turn' | 'max_tokens' | 'tool_use' | null {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
      return 'tool_use';
    default:
      return null;
  }
}

/** Convert a non-streaming OpenAI completion into an Anthropic message. */
export function convertOpenAIResponseToAnthropic(
  openaiResp: OpenAIChatResponse,
  model: string
): AnthropicResponse {
  const choice = openaiResp.choices[0];
  if (!choice) {
    throw new Error('OpenAI-compatible provider returned no completion choices');
  }

  const message = choice.message;
  const content: AnthropicContentBlock[] = [];

  if (message.content) {
    const textContent =
      typeof message.content === 'string'
        ? message.content
        : message.content.map((part) => (part.type === 'text' ? part.text : '')).join(' ');
    if (textContent) content.push({ type: 'text', text: textContent });
  }

  for (const toolCall of message.tool_calls || []) {
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

  return {
    id: openaiResp.id,
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: mapFinishReason(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: openaiResp.usage?.prompt_tokens || 0,
      output_tokens: openaiResp.usage?.completion_tokens || 0,
    },
  };
}

/**
 * Transform OpenAI SSE chunks into Anthropic SSE events.
 * Includes a stateful UTF-8 decoder and always closes the message cleanly.
 */
export class AnthropicStreamAdapter {
  private encoder = new TextEncoder();
  private blockIndex = 0;
  private activeTextBlockIndex: number | null = null;
  private started = false;
  private finished = false;
  private inputTokens = 0;
  private outputTokens = 0;
  private toolCalls = new Map<
    number,
    { id: string; name: string; arguments: string }
  >();

  constructor(
    private model: string,
    private messageId: string
  ) {}

  private encodeSSE(event: string, data: unknown): Uint8Array {
    return this.encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  private startMessage(controller: TransformStreamDefaultController<Uint8Array>): void {
    if (this.started) return;
    this.started = true;
    controller.enqueue(
      this.encodeSSE('message_start', {
        type: 'message_start',
        message: {
          id: this.messageId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: this.inputTokens, output_tokens: 0 },
        },
      })
    );
  }

  private startTextBlock(
    controller: TransformStreamDefaultController<Uint8Array>
  ): void {
    if (this.activeTextBlockIndex !== null) return;
    const index = this.blockIndex++;
    this.activeTextBlockIndex = index;
    controller.enqueue(
      this.encodeSSE('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      })
    );
  }

  private stopTextBlock(
    controller: TransformStreamDefaultController<Uint8Array>
  ): void {
    if (this.activeTextBlockIndex === null) return;
    controller.enqueue(
      this.encodeSSE('content_block_stop', {
        type: 'content_block_stop',
        index: this.activeTextBlockIndex,
      })
    );
    this.activeTextBlockIndex = null;
  }

  /**
   * Tool-call deltas may be interleaved in one OpenAI chunk.
   * Buffer by provider tool index so arguments never leak into another tool block.
   */
  private emitToolBlocks(
    controller: TransformStreamDefaultController<Uint8Array>
  ): void {
    const orderedTools = [...this.toolCalls.entries()].sort(([left], [right]) => left - right);
    for (const [, tool] of orderedTools) {
      const index = this.blockIndex++;
      controller.enqueue(
        this.encodeSSE('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: tool.id,
            name: tool.name,
            input: {},
          },
        })
      );

      if (tool.arguments) {
        controller.enqueue(
          this.encodeSSE('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: { type: 'input_json_delta', partial_json: tool.arguments },
          })
        );
      }

      controller.enqueue(
        this.encodeSSE('content_block_stop', {
          type: 'content_block_stop',
          index,
        })
      );
    }
  }

  private finishMessage(
    controller: TransformStreamDefaultController<Uint8Array>,
    reason: 'end_turn' | 'max_tokens' | 'tool_use' | null
  ): void {
    if (this.finished) return;
    this.startMessage(controller);
    this.stopTextBlock(controller);
    this.emitToolBlocks(controller);

    const stopReason = this.toolCalls.size > 0 ? 'tool_use' : reason || 'end_turn';
    controller.enqueue(
      this.encodeSSE('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: this.outputTokens },
      })
    );
    controller.enqueue(this.encodeSSE('message_stop', { type: 'message_stop' }));
    this.finished = true;
  }

  createTransformStream(): TransformStream<Uint8Array, Uint8Array> {
    let lineBuffer = '';
    const decoder = new TextDecoder();

    return new TransformStream({
      transform: (chunk, controller) => {
        const text = lineBuffer + decoder.decode(chunk, { stream: true });
        const lines = text.split(/\r?\n/);
        lineBuffer = lines.pop() || '';

        for (const line of lines) this.processSSELine(line, controller);
      },
      flush: (controller) => {
        const remaining = lineBuffer + decoder.decode();
        if (remaining) this.processSSELine(remaining, controller);
        if (!this.finished) this.finishMessage(controller, 'end_turn');
        lineBuffer = '';
      },
    });
  }

  private processSSELine(
    line: string,
    controller: TransformStreamDefaultController<Uint8Array>
  ): void {
    if (!line.startsWith('data:')) return;
    const payload = line.slice(5).trim();
    if (!payload || payload === '[DONE]') return;

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      return;
    }

    if (parsed.error && typeof parsed.error === 'object') {
      this.startMessage(controller);
      controller.enqueue(
        this.encodeSSE('error', {
          type: 'error',
          error: parsed.error,
        })
      );
      this.finished = true;
      return;
    }

    const usage = parsed.usage as Record<string, unknown> | undefined;
    if (usage) {
      if (typeof usage.prompt_tokens === 'number') this.inputTokens = usage.prompt_tokens;
      if (typeof usage.completion_tokens === 'number') {
        this.outputTokens = usage.completion_tokens;
      }
    }

    const choices = parsed.choices;
    const choice = Array.isArray(choices) ? choices[0] : undefined;
    if (!choice || typeof choice !== 'object') return;

    const choiceRecord = choice as Record<string, unknown>;
    const delta = (choiceRecord.delta || {}) as Record<string, unknown>;
    const finishReason = choiceRecord.finish_reason as string | null | undefined;

    this.startMessage(controller);

    if (typeof delta.content === 'string' && delta.content.length > 0) {
      this.startTextBlock(controller);
      controller.enqueue(
        this.encodeSSE('content_block_delta', {
          type: 'content_block_delta',
          index: this.activeTextBlockIndex,
          delta: { type: 'text_delta', text: delta.content },
        })
      );
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const item of delta.tool_calls) {
        if (!item || typeof item !== 'object') continue;
        const toolCall = item as Record<string, unknown>;
        const fn = (toolCall.function || {}) as Record<string, unknown>;
        const index = Number.isInteger(toolCall.index) ? Number(toolCall.index) : 0;
        const tool = this.toolCalls.get(index) || { id: '', name: '', arguments: '' };

        if (typeof toolCall.id === 'string' && toolCall.id.length > 0) {
          tool.id = toolCall.id;
        }
        if (typeof fn.name === 'string') tool.name += fn.name;
        if (typeof fn.arguments === 'string') tool.arguments += fn.arguments;
        this.toolCalls.set(index, tool);
      }
    }

    if (finishReason) {
      const stopReason = mapFinishReason(finishReason);
      this.finishMessage(controller, stopReason);
    }
  }
}
