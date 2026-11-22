import { AnthropicRequest, AnthropicResponse, AnthropicContentBlock } from '../anthropic-types';
import {
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIMessage,
  ContentPart,
  ToolCall,
} from '../types';

/** Convert an Anthropic request into the common OpenAI-compatible request format. */
export function convertAnthropicRequestToOpenAI(anthropicReq: AnthropicRequest): OpenAIChatRequest {
  const messages: OpenAIMessage[] = [];

  if (anthropicReq.system) messages.push({ role: 'system', content: anthropicReq.system });

  for (const msg of anthropicReq.messages) {
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
      continue;
    }

    const contentParts: ContentPart[] = [];
    const toolCalls: ToolCall[] = [];
    const toolResults: OpenAIMessage[] = [];

    for (const block of msg.content) {
      if (block.type === 'text' && block.text) {
        contentParts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image' && block.source) {
        contentParts.push({
          type: 'image_url',
          image_url: {
            url: `data:${block.source.media_type};base64,${block.source.data}`,
          },
        });
      } else if (block.type === 'tool_result') {
        const toolContent =
          typeof block.content === 'string'
            ? block.content
            : block.content
              ? block.content
                  .filter((part) => part.type === 'text')
                  .map((part) => part.text || '')
                  .join(' ')
              : '';
        if (block.tool_use_id) {
          toolResults.push({
            role: 'tool',
            tool_call_id: block.tool_use_id,
            content: toolContent,
          });
        } else {
          contentParts.push({ type: 'text', text: toolContent });
        }
      } else if (block.type === 'tool_use' && block.name) {
        toolCalls.push({
          id: block.id || '',
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      }
    }

    let content: OpenAIMessage['content'] = null;
    if (contentParts.length === 1 && contentParts[0].type === 'text') {
      content = contentParts[0].text;
    } else if (contentParts.length > 0) {
      content = contentParts;
    }

    if (content !== null || toolCalls.length > 0) {
      messages.push({
        role: msg.role,
        content,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
    }
    messages.push(...toolResults);
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
  private activeBlockIndex = 0;
  private started = false;
  private textBlockActive = false;
  private toolBlockActive = false;
  private finished = false;
  private outputTokens = 0;

  constructor(
    private model: string,
    private messageId: string
  ) {}

  private encodeSSE(event: string, data: unknown): Uint8Array {
    return this.encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  private startMessage(controller: TransformStreamDefaultController): void {
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
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      })
    );
  }

  private startTextBlock(controller: TransformStreamDefaultController): void {
    this.activeBlockIndex = this.blockIndex++;
    this.textBlockActive = true;
    controller.enqueue(
      this.encodeSSE('content_block_start', {
        type: 'content_block_start',
        index: this.activeBlockIndex,
        content_block: { type: 'text', text: '' },
      })
    );
  }

  private stopActiveBlock(controller: TransformStreamDefaultController): void {
    if (!this.textBlockActive && !this.toolBlockActive) return;
    controller.enqueue(
      this.encodeSSE('content_block_stop', {
        type: 'content_block_stop',
        index: this.activeBlockIndex,
      })
    );
    this.textBlockActive = false;
    this.toolBlockActive = false;
  }

  private finishMessage(
    controller: TransformStreamDefaultController,
    reason: 'end_turn' | 'max_tokens' | 'tool_use' | null
  ): void {
    if (this.finished) return;
    this.startMessage(controller);
    this.stopActiveBlock(controller);
    controller.enqueue(
      this.encodeSSE('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: reason || 'end_turn', stop_sequence: null },
        usage: { output_tokens: this.outputTokens },
      })
    );
    controller.enqueue(this.encodeSSE('message_stop', { type: 'message_stop' }));
    this.finished = true;
  }

  createTransformStream(): TransformStream<Uint8Array, Uint8Array> {
    let lineBuffer = '';
    let toolIndex = -1;
    const decoder = new TextDecoder();

    return new TransformStream({
      transform: (chunk, controller) => {
        const text = lineBuffer + decoder.decode(chunk, { stream: true });
        const lines = text.split(/\r?\n/);
        lineBuffer = lines.pop() || '';

        for (const line of lines) {
          toolIndex = this.processSSELine(line, controller, toolIndex);
        }
      },
      flush: (controller) => {
        const tail = decoder.decode();
        const remaining = lineBuffer + tail;
        if (remaining) this.processSSELine(remaining, controller, toolIndex);
        if (!this.finished) this.finishMessage(controller, 'end_turn');
        lineBuffer = '';
      },
    });
  }

  private processSSELine(
    line: string,
    controller: TransformStreamDefaultController,
    toolIndex: number
  ): number {
    if (!line.startsWith('data:')) return toolIndex;
    const payload = line.slice(5).trim();
    if (!payload || payload === '[DONE]') return toolIndex;

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      return toolIndex;
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
      return toolIndex;
    }

    const usage = parsed.usage as Record<string, unknown> | undefined;
    if (usage && typeof usage.completion_tokens === 'number') {
      this.outputTokens = usage.completion_tokens;
    }

    const choices = parsed.choices;
    const choice = Array.isArray(choices) ? choices[0] : undefined;
    if (!choice || typeof choice !== 'object') return toolIndex;

    const choiceRecord = choice as Record<string, unknown>;
    const delta = (choiceRecord.delta || {}) as Record<string, unknown>;
    const finishReason = choiceRecord.finish_reason as string | null | undefined;

    this.startMessage(controller);

    if (typeof delta.content === 'string' && delta.content.length > 0) {
      if (!this.textBlockActive && !this.toolBlockActive) this.startTextBlock(controller);
      controller.enqueue(
        this.encodeSSE('content_block_delta', {
          type: 'content_block_delta',
          index: this.activeBlockIndex,
          delta: { type: 'text_delta', text: delta.content },
        })
      );
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const item of delta.tool_calls) {
        if (!item || typeof item !== 'object') continue;
        const toolCall = item as Record<string, unknown>;
        const fn = (toolCall.function || {}) as Record<string, unknown>;

        if (toolCall.id) {
          this.stopActiveBlock(controller);
          toolIndex++;
          this.activeBlockIndex = this.blockIndex++;
          this.toolBlockActive = true;
          controller.enqueue(
            this.encodeSSE('content_block_start', {
              type: 'content_block_start',
              index: this.activeBlockIndex,
              content_block: {
                type: 'tool_use',
                id: toolCall.id,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            })
          );
        }

        if (typeof fn.arguments === 'string' && this.toolBlockActive) {
          controller.enqueue(
            this.encodeSSE('content_block_delta', {
              type: 'content_block_delta',
              index: this.activeBlockIndex,
              delta: { type: 'input_json_delta', partial_json: fn.arguments },
            })
          );
        }
      }
    }

    if (finishReason) {
      const stopReason = mapFinishReason(finishReason);
      this.finishMessage(controller, stopReason);
    }

    return toolIndex;
  }
}
