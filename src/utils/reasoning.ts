/**
 * Normalizes provider-specific reasoning into OpenAI-compatible assistant content.
 * Only explicit human-readable thought text is exposed; signatures and opaque metadata are not.
 */
export function extractReasoningText(...values: unknown[]): string {
  const parts: string[] = [];

  const visit = (value: unknown, depth: number): void => {
    if (depth > 4 || value === null || value === undefined) return;

    if (typeof value === 'string') {
      if (value.length > 0) parts.push(value);
      return;
    }

    if (Array.isArray(value)) {
      for (const item of value) visit(item, depth + 1);
      return;
    }

    if (typeof value !== 'object') return;
    const record = value as Record<string, unknown>;

    // Accept documented text-bearing reasoning representations only.
    if (typeof record.text === 'string') {
      visit(record.text, depth + 1);
      return;
    }
    if (typeof record.thinking === 'string') {
      visit(record.thinking, depth + 1);
      return;
    }
    if (typeof record.summary === 'string') {
      visit(record.summary, depth + 1);
      return;
    }
    if (typeof record.content === 'string') {
      visit(record.content, depth + 1);
    }
  };

  for (const value of values) visit(value, 0);
  return parts.join('');
}

/** Stateful formatter for streams where reasoning and final text arrive separately. */
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

  finish(): string {
    if (!this.opened) return '';
    this.opened = false;
    return '\n</think>';
  }
}

/**
 * Normalize one non-streaming assistant message without discarding multimodal content.
 * For block content, the reasoning prefix is inserted as a text block before the original blocks.
 */
export function normalizeAssistantMessage(message: unknown): unknown {
  if (!message || typeof message !== 'object' || Array.isArray(message)) return message;

  const source = message as Record<string, unknown>;
  const reasoning = extractReasoningText(
    source.reasoning_content,
    source.reasoning,
    source.thinking,
    source.reasoning_details
  );

  const normalized: Record<string, unknown> = { ...source };
  delete normalized.reasoning_content;
  delete normalized.reasoning;
  delete normalized.thinking;
  delete normalized.reasoning_details;

  if (!reasoning) return normalized;

  const prefix = `<think>\n${reasoning}\n</think>\n`;
  if (typeof source.content === 'string') {
    normalized.content = prefix + source.content;
  } else if (Array.isArray(source.content)) {
    normalized.content = [{ type: 'text', text: prefix }, ...source.content];
  } else {
    normalized.content = prefix;
  }

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
