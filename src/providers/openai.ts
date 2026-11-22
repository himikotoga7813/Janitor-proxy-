import OpenAI from 'openai';
import { BaseProvider } from './base';
import { OpenAIChatRequest, ProviderResponse } from '../types';
import {
  extractReasoningText,
  normalizeOpenAIResponse,
  ReasoningStreamFormatter,
} from '../utils/reasoning';

export class OpenAIProvider extends BaseProvider {
  async chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      const client = new OpenAI({
        apiKey,
        baseURL: this.baseUrl,
      });

      // Preserve compatible request options (response_format, seed, reasoning_effort, etc.)
      // while always routing through the configured upstream model name.
      const params = {
        ...request,
        model: this.model,
        messages: request.messages,
        stream: request.stream ?? false,
      };

      if (request.stream) {
        return this.handleStream(
          client,
          params as unknown as OpenAI.Chat.ChatCompletionCreateParamsStreaming
        );
      }
      return this.handleNonStream(
        client,
        params as unknown as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
      );
    } catch (error) {
      return this.handleError(error, 'OpenAIProvider');
    }
  }

  private async handleNonStream(
    client: OpenAI,
    params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
  ): Promise<ProviderResponse> {
    const response = await client.chat.completions.create(params);
    return {
      success: true,
      response: normalizeOpenAIResponse(response) as unknown as ProviderResponse['response'],
    };
  }

  private async handleStream(
    client: OpenAI,
    params: OpenAI.Chat.ChatCompletionCreateParamsStreaming
  ): Promise<ProviderResponse> {
    const stream = await client.chat.completions.create(params);
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    const formatters = new Map<number, ReasoningStreamFormatter>();

    const formatterFor = (index: number): ReasoningStreamFormatter => {
      let formatter = formatters.get(index);
      if (!formatter) {
        formatter = new ReasoningStreamFormatter();
        formatters.set(index, formatter);
      }
      return formatter;
    };

    (async () => {
      try {
        for await (const chunk of stream) {
          const normalizedChoices = chunk.choices.map((choice) => {
            const delta = choice.delta as unknown as Record<string, unknown>;
            const formatter = formatterFor(choice.index);
            const reasoning = extractReasoningText(
              delta.reasoning_content,
              delta.reasoning,
              delta.thinking
            );
            const nextDelta: Record<string, unknown> = { ...delta };

            delete nextDelta.reasoning_content;
            delete nextDelta.reasoning;
            delete nextDelta.thinking;

            let normalizedText = '';
            if (reasoning) normalizedText += formatter.pushReasoning(reasoning);
            if (typeof delta.content === 'string') {
              normalizedText += formatter.pushContent(delta.content);
            }
            if (choice.finish_reason) normalizedText += formatter.finish();

            if (normalizedText) nextDelta.content = normalizedText;
            else if (typeof delta.content !== 'string') delete nextDelta.content;

            return {
              ...choice,
              delta: nextDelta,
            };
          });

          const normalizedChunk = { ...chunk, choices: normalizedChoices };
          await writer.write(encoder.encode(`data: ${JSON.stringify(normalizedChunk)}\n\n`));
        }

        // Some compatible APIs end without a finish_reason chunk.
        for (const [index, formatter] of formatters) {
          const closing = formatter.finish();
          if (closing) {
            const closingChunk = {
              id: 'chatcmpl-reasoning-close',
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: params.model,
              choices: [{ index, delta: { content: closing }, finish_reason: null }],
            };
            await writer.write(encoder.encode(`data: ${JSON.stringify(closingChunk)}\n\n`));
          }
        }
        await writer.write(encoder.encode('data: [DONE]\n\n'));
      } catch (error) {
        console.error('[OpenAIProvider] Stream error:', error);
        const message = error instanceof Error ? error.message : 'Upstream stream failed';
        const errorPayload = {
          error: {
            message,
            type: 'provider_stream_error',
          },
        };
        try {
          await writer.write(encoder.encode(`data: ${JSON.stringify(errorPayload)}\n\n`));
          await writer.write(encoder.encode('data: [DONE]\n\n'));
        } catch {
          // The downstream client may have cancelled the stream.
        }
      } finally {
        try {
          await writer.close();
        } catch {
          // Already closed or cancelled.
        }
      }
    })();

    return { success: true, stream: readable };
  }
}
