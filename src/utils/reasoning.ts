/**
 * Normalizes provider-specific reasoning into the OpenAI-compatible message content.
 *
 * Reasoning is only emitted when the upstream provider explicitly supplies it.
 * Opaque thought signatures and other metadata are never treated as reasoning text.
 */
export function extractReasoningText(...values: unknown[]): string {
  const parts: string[] = [];

  for (const value of values) {
    if (typeof value === 'string' && value.length > 0) {
      parts.push(value);
    } else if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === 'string' && item.length > 0) {
          parts.push(item);
        } else if (
          item &&
          typeof item === 'object' &&
          typeof (item as Record<string, unknown>).text === 'string'
        ) {
          parts.push((item as Record<string, string>).text);
        }
      }
    } else if (value && typeof value === 'object') {
      const record = value as Record<string, unknown>;
      if (typeof record.text === 'string') parts.push(record.text);
      else if (typeof record.content === 'string') parts.push(record.content);
    }
  }

  return parts.join('');
}

/** Stateful formatter used for streams where reasoning and final text arrive separately. */
export class ReasoningStreamFormatter {
  private opened = false;

  pushReasoning(text: string): string {
    if (!text) return '';
    if (this.opened) return text;
    this.opened = true;
    return `<think>\n${text}`;
  }

  pushContent(text: string): string {
    if (!text) return '';
    if (!this.opened) return text;
    this.opened = false;
    return `\n</think>\n${text}`;
  }

  /** Closes a reasoning block if the stream ends before final content starts. */
  finish(): string {
    if (!this.opened) return '';
    this.opened = false;
    return '\n</think>';
  }
}

/**
 * Convert a non-streaming OpenAI-compatible assistant message.
 * The upstream-only reasoning fields are removed after being represented in content.
 */
export function normalizeAssistantMessage(message: unknown): unknown {
  if (!message || typeof message !== 'object' || Array.isArray(message)) return message;

  const source = message as Record<string, unknown>;
  const reasoning = extractReasoningText(
    source.reasoning_content,
    source.reasoning,
    source.thinking
  );
  const formatter = new ReasoningStreamFormatter();
  const reasoningPrefix = reasoning ? formatter.pushReasoning(reasoning) : '';
  const content =
    typeof source.content === 'string'
      ? source.content
      : source.content === null || source.content === undefined
        ? ''
        : '';

  const normalizedContent = reasoningPrefix
    ? reasoningPrefix + formatter.pushContent(content) + formatter.finish()
    : source.content;

  const normalized: Record<string, unknown> = { ...source };
  delete normalized.reasoning_content;
  delete normalized.reasoning;
  delete normalized.thinking;

  if (reasoningPrefix) normalized.content = normalizedContent;
  return normalized;
}

/** Normalize all assistant messages in an OpenAI-compatible completion response. */
export function normalizeOpenAIResponse<T>(response: T): T {
  if (!response || typeof response !== 'object') return response;

  const source = response as Record<string, unknown>;
  if (!Array.isArray(source.choices)) return response;

  const choices = source.choices.map((choice) => {
    if (!choice || typeof choice !== 'object') return choice;
    const choiceRecord = choice as Record<string, unknown>;
    const message = choiceRecord.message;
    if (!message || typeof message !== 'object') return choice;
    return {
      ...choiceRecord,
      message: normalizeAssistantMessage(message),
    };
  });

  return { ...source, choices } as T;
}
