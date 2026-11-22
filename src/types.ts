// OpenAI API types

export interface TextContentPart {
  type: 'text';
  text: string;
}

export interface ImageContentPart {
  type: 'image_url';
  image_url: {
    url: string;
    detail?: 'low' | 'high' | 'auto';
  };
}

export type ContentPart = TextContentPart | ImageContentPart;

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool' | 'function';
  content: string | ContentPart[] | null;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  /** Provider-specific reasoning fields; normalized into content when present. */
  reasoning_content?: string;
  reasoning?: string;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

/** Tool call delta in streaming — MUST include `index` per OpenAI spec */
export interface StreamToolCallDelta {
  index: number;
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    arguments?: string;
  };
}

export interface Tool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: Tool[];
  tool_choice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string | string[];
  n?: number;
  /** Common OpenAI-compatible extension fields passed through to compatible providers. */
  response_format?: Record<string, unknown>;
  seed?: number;
  logprobs?: boolean;
  top_logprobs?: number;
  user?: string;
  parallel_tool_calls?: boolean;
  stream_options?: Record<string, unknown>;
  reasoning_effort?: string;
  verbosity?: string;
  [key: string]: unknown;
}

export interface OpenAIChatResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: OpenAIMessage;
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAIStreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string | null;
      reasoning_content?: string | null;
      reasoning?: string | null;
      tool_calls?: StreamToolCallDelta[];
    };
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  }>;
}

// Provider configuration
export interface ProviderConfig {
  provider:
    | 'anthropic'
    | 'anthropic-compatible'
    | 'google'
    | 'openai'
    | 'openai-compatible'
    | 'cloudflare-ai';
  model: string;
  apiKeys: string[];
  baseUrl?: string;
  /** Enable Google Search grounding (Google provider only) */
  grounding?: boolean;
}

export interface RouteConfig {
  [route: string]: ProviderConfig[];
}

// Environment bindings
export interface Env {
  AI?: any; // eslint-disable-line @typescript-eslint/no-explicit-any
  PROXY_AUTH_TOKEN: string;
  ROUTES_CONFIG: string;
  /** Optional override for the time allowed to establish an upstream response. */
  PROVIDER_TIMEOUT_MS?: string;
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

// Provider response
export interface ProviderResponse {
  success: boolean;
  response?: OpenAIChatResponse;
  /** Anthropic-native raw response (used by nativeChat path) */
  rawResponse?: unknown;
  stream?: ReadableStream<Uint8Array>;
  error?: string;
  statusCode?: number;
}
