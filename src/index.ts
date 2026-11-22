import { Env, OpenAIChatRequest, ProviderConfig } from './types';
import { AnthropicRequest } from './anthropic-types';
import { Router } from './router';
import {
  ProxyError,
  createErrorResponse,
  getProviderTimeoutMs,
  withTimeout,
} from './utils/error-handler';
import { createProvider } from './providers';
import { AnthropicProvider } from './providers/anthropic';
import {
  convertAnthropicRequestToOpenAI,
  convertOpenAIResponseToAnthropic,
  AnthropicStreamAdapter,
} from './utils/anthropic-adapter';
import { generateId } from './utils/response-mapper';

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers':
    'Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta',
  'Access-Control-Max-Age': '86400',
  'X-Content-Type-Options': 'nosniff',
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    try {
      const url = new URL(request.url);
      const path = url.pathname;

      if ((path === '/health' || path === '/') && request.method === 'GET') {
        return json({
          status: 'ok',
          service: 'ai-worker-proxy',
          timestamp: new Date().toISOString(),
        });
      }

      if ((path === '/models' || path === '/v1/models') && request.method === 'GET') {
        const router = new Router(env);
        return json({
          object: 'list',
          data: router.getAvailableModels(),
        });
      }

      if (!env.PROXY_AUTH_TOKEN || !env.PROXY_AUTH_TOKEN.trim()) {
        throw new ProxyError(
          'Proxy authentication is not configured on this Worker',
          503,
          'auth_not_configured'
        );
      }

      if (!verifyAuth(request, env)) {
        throw new ProxyError('Unauthorized', 401, 'invalid_auth');
      }

      if (
        request.method === 'POST' &&
        (path === '/anthropic/v1/messages' || path === '/anthropic/messages')
      ) {
        return await handleAnthropicChat(request, env);
      }

      if (
        request.method === 'POST' &&
        (path === '/' || path === '/v1/chat/completions' || path === '/chat/completions')
      ) {
        return await handleChatCompletion(request, env);
      }

      throw new ProxyError('Not found', 404, 'not_found');
    } catch (error) {
      console.error('[Worker] Request failed:', error);
      const errorResponse = createErrorResponse(error);
      const headers = new Headers(errorResponse.headers);
      for (const [key, value] of Object.entries(CORS_HEADERS)) {
        headers.set(key, value);
      }
      return new Response(errorResponse.body, {
        status: errorResponse.status,
        headers,
      });
    }
  },
};

async function handleChatCompletion(request: Request, env: Env): Promise<Response> {
  const body = await parseJsonBody(request);
  const chatRequest = validateOpenAIRequest(body);

  console.log(
    `[Worker] model=${chatRequest.model} stream=${chatRequest.stream ?? false} messages=${chatRequest.messages.length}`
  );

  const router = new Router(env);
  const response = await router.executeWithFallback(chatRequest, 'openai');

  if (!response.success) {
    throw new ProxyError(
      response.error || 'All providers failed',
      response.statusCode || 502,
      'provider_error'
    );
  }

  if (response.stream) {
    return new Response(response.stream, {
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        ...CORS_HEADERS,
      },
    });
  }

  if (!response.response) {
    throw new ProxyError('Provider returned success but no response data', 502);
  }
  return json(response.response);
}

async function handleAnthropicChat(request: Request, env: Env): Promise<Response> {
  const body = validateAnthropicRequest(await parseJsonBody(request));

  console.log(
    `[AnthropicHandler] model=${body.model} stream=${body.stream ?? false} messages=${body.messages.length}`
  );

  const router = new Router(env);
  const providers = router.getProvidersForModel(body.model);
  const hasAnthropicNative = providers.some(
    (provider) =>
      provider.provider === 'anthropic' || provider.provider === 'anthropic-compatible'
  );

  if (hasAnthropicNative) return handleAnthropicNativePath(body, providers, env);
  return handleAnthropicConversionPath(body, router);
}

/** Call native Anthropic providers without changing Anthropic's request/response schema. */
async function handleAnthropicNativePath(
  body: AnthropicRequest,
  providers: ProviderConfig[],
  env: Env
): Promise<Response> {
  let lastError: unknown = null;
  let lastStatusCode = 502;

  for (const config of providers) {
    if (config.provider !== 'anthropic' && config.provider !== 'anthropic-compatible') {
      continue;
    }

    const provider = createProvider(config, env) as AnthropicProvider;
    const apiKeys = resolveApiKeys(config, env);

    if (apiKeys.length === 0) {
      lastError = new Error(`No API keys configured for ${config.provider}/${config.model}`);
      lastStatusCode = 500;
      continue;
    }

    for (let index = 0; index < apiKeys.length; index++) {
      try {
        const result = await withTimeout(
          provider.nativeChat(body, apiKeys[index]),
          getProviderTimeoutMs(env)
        );
        if (!result.success) {
          lastError = result.error || 'Anthropic provider failed';
          lastStatusCode = result.statusCode || 502;
          if (![401, 403, 429, 502, 503, 504, 529].includes(lastStatusCode)) break;
          continue;
        }

        if (result.stream) {
          return new Response(result.stream, {
            headers: {
              'Content-Type': 'text/event-stream; charset=utf-8',
              'Cache-Control': 'no-cache, no-transform',
              Connection: 'keep-alive',
              ...CORS_HEADERS,
            },
          });
        }

        if (result.rawResponse === undefined) {
          throw new ProxyError('Anthropic provider returned no response data', 502);
        }
        return json(result.rawResponse);
      } catch (error) {
        lastError = error;
        lastStatusCode =
          error && typeof error === 'object' && 'statusCode' in error &&
          typeof (error as { statusCode?: unknown }).statusCode === 'number'
            ? (error as { statusCode: number }).statusCode
            : 502;
      }
    }
  }

  // If native providers failed, try remaining providers through the format adapter.
  const otherProviders = providers.filter(
    (provider) => provider.provider !== 'anthropic' && provider.provider !== 'anthropic-compatible'
  );
  if (otherProviders.length > 0) {
    const router = new Router(env);
    return handleAnthropicConversionPath(body, router, otherProviders);
  }

  const message = lastError instanceof Error ? lastError.message : String(lastError || 'Unknown error');
  throw new ProxyError(`All Anthropic providers failed: ${message}`, lastStatusCode);
}

async function handleAnthropicConversionPath(
  body: AnthropicRequest,
  router: Router,
  overrideProviders?: ProviderConfig[]
): Promise<Response> {
  const openaiRequest = convertAnthropicRequestToOpenAI(body);
  const response = await router.executeWithFallback(openaiRequest, 'anthropic', overrideProviders);

  if (!response.success) {
    throw new ProxyError(
      response.error || 'All providers failed',
      response.statusCode || 502,
      'provider_error'
    );
  }

  if (response.stream) {
    const adapter = new AnthropicStreamAdapter(body.model, `msg_${generateId()}`);
    const anthropicStream = response.stream.pipeThrough(adapter.createTransformStream());
    return new Response(anthropicStream, {
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        ...CORS_HEADERS,
      },
    });
  }

  if (!response.response) {
    throw new ProxyError('Provider returned success but no response data', 502);
  }
  return json(convertOpenAIResponseToAnthropic(response.response, body.model));
}

async function parseJsonBody(request: Request): Promise<unknown> {
  const contentType = request.headers.get('content-type') || '';
  if (contentType && !contentType.toLowerCase().includes('application/json')) {
    throw new ProxyError('Content-Type must be application/json', 415, 'unsupported_media_type');
  }

  try {
    return await request.json();
  } catch {
    throw new ProxyError('Request body must contain valid JSON', 400, 'invalid_json');
  }
}

function validateOpenAIRequest(value: unknown): OpenAIChatRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProxyError('Request body must be a JSON object', 400, 'invalid_request');
  }

  const body = value as Record<string, unknown>;
  if (typeof body.model !== 'string' || !body.model.trim()) {
    throw new ProxyError('Invalid request: model is required', 400, 'invalid_request');
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw new ProxyError('Invalid request: a non-empty messages array is required', 400, 'invalid_request');
  }

  for (let index = 0; index < body.messages.length; index++) {
    const message = body.messages[index];
    if (!message || typeof message !== 'object' || Array.isArray(message)) {
      throw new ProxyError(`Invalid message at index ${index}`, 400, 'invalid_request');
    }
    const record = message as Record<string, unknown>;
    if (!['system', 'user', 'assistant', 'tool', 'function'].includes(String(record.role))) {
      throw new ProxyError(`Invalid role in message at index ${index}`, 400, 'invalid_request');
    }
    const content = record.content;
    if (
      content !== null &&
      content !== undefined &&
      typeof content !== 'string' &&
      !Array.isArray(content)
    ) {
      throw new ProxyError(`Invalid content in message at index ${index}`, 400, 'invalid_request');
    }
  }

  for (const field of ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty'] as const) {
    const fieldValue = body[field];
    if (fieldValue !== undefined && (typeof fieldValue !== 'number' || !Number.isFinite(fieldValue))) {
      throw new ProxyError(`${field} must be a finite number`, 400, 'invalid_request');
    }
  }
  if (
    body.max_tokens !== undefined &&
    (typeof body.max_tokens !== 'number' || !Number.isInteger(body.max_tokens) || body.max_tokens < 1)
  ) {
    throw new ProxyError('max_tokens must be a positive integer', 400, 'invalid_request');
  }
  if (body.n !== undefined && (typeof body.n !== 'number' || !Number.isInteger(body.n) || body.n < 1)) {
    throw new ProxyError('n must be a positive integer', 400, 'invalid_request');
  }
  if (body.stream !== undefined && typeof body.stream !== 'boolean') {
    throw new ProxyError('stream must be a boolean', 400, 'invalid_request');
  }
  if (body.tools !== undefined && !Array.isArray(body.tools)) {
    throw new ProxyError('tools must be an array', 400, 'invalid_request');
  }

  return body as unknown as OpenAIChatRequest;
}

function validateAnthropicRequest(value: unknown): AnthropicRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProxyError('Request body must be a JSON object', 400, 'invalid_request');
  }

  const body = value as Record<string, unknown>;
  if (typeof body.model !== 'string' || !body.model.trim()) {
    throw new ProxyError('Invalid request: model is required', 400, 'invalid_request');
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw new ProxyError('Invalid request: a non-empty messages array is required', 400, 'invalid_request');
  }
  if (
    body.max_tokens !== undefined &&
    (typeof body.max_tokens !== 'number' || !Number.isInteger(body.max_tokens) || body.max_tokens < 1)
  ) {
    throw new ProxyError('max_tokens must be a positive integer', 400, 'invalid_request');
  }
  if (body.stream !== undefined && typeof body.stream !== 'boolean') {
    throw new ProxyError('stream must be a boolean', 400, 'invalid_request');
  }

  return body as unknown as AnthropicRequest;
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...CORS_HEADERS },
  });
}

function verifyAuth(request: Request, env: Env): boolean {
  const expectedToken = env.PROXY_AUTH_TOKEN;
  if (!expectedToken) return false;

  const authorization = request.headers.get('Authorization') || '';
  const bearerMatch = authorization.match(/^Bearer\s+(.+)$/i);
  const authorizationToken = bearerMatch ? bearerMatch[1].trim() : authorization.trim();
  if (authorizationToken && authorizationToken === expectedToken) return true;

  const apiKey = request.headers.get('x-api-key');
  return Boolean(apiKey && apiKey === expectedToken);
}

function resolveApiKeys(config: ProviderConfig, env: Env): string[] {
  const keys = new Set<string>();
  for (const keyName of config.apiKeys) {
    const value = env[keyName];
    if (typeof value === 'string' && value.trim()) {
      keys.add(value);
    } else {
      console.warn(`[resolveApiKeys] API key environment variable is unavailable: ${keyName}`);
    }
  }
  return [...keys];
}
