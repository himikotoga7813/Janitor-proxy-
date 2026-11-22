import { RouteConfig, ProviderConfig, Env, OpenAIChatRequest, ProviderResponse } from './types';
import { TokenManager } from './token-manager';
import { getErrorStatusCode, ProxyError } from './utils/error-handler';

const SUPPORTED_PROVIDERS = new Set([
  'anthropic',
  'anthropic-compatible',
  'google',
  'openai',
  'openai-compatible',
  'cloudflare-ai',
]);

export class Router {
  private routes: RouteConfig;

  constructor(private env: Env) {
    this.routes = this.parseRoutesConfig();
  }

  /** Get list of available public model aliases. */
  getAvailableModels(): Array<{
    id: string;
    object: string;
    owned_by: string;
    permission: string[];
  }> {
    return Object.keys(this.routes).map((model) => ({
      id: model,
      object: 'model',
      owned_by: 'ai-worker-proxy',
      permission: [],
    }));
  }

  /** Get provider configurations for a given model alias. */
  getProvidersForModel(model: string): ProviderConfig[] {
    const providers = this.routes[model];
    if (!providers || providers.length === 0) {
      throw new ProxyError(`No providers configured for model: ${model}`, 404, 'model_not_found');
    }
    return providers;
  }

  /**
   * Execute request with provider and credential fallback.
   * Invalid client requests are not retried against every configured upstream.
   */
  async executeWithFallback(
    request: OpenAIChatRequest,
    preferredType?: 'openai' | 'anthropic',
    overrideProviders?: ProviderConfig[]
  ): Promise<ProviderResponse> {
    const model = request.model;
    if (!model) throw new ProxyError('Model name is required', 400, 'invalid_request');

    const providers = overrideProviders ?? this.getProvidersForModel(model);
    if (!Array.isArray(providers) || providers.length === 0) {
      throw new ProxyError(`No providers configured for model: ${model}`, 404, 'model_not_found');
    }

    const orderedProviders = preferredType
      ? this.sortByPreferredType(providers, preferredType)
      : providers;

    console.log(
      `[Router] Model "${model}" has ${orderedProviders.length} provider(s) configured`
    );

    let lastError = 'Unknown provider error';
    let lastStatusCode = 502;

    for (let i = 0; i < orderedProviders.length; i++) {
      const config = orderedProviders[i];
      console.log(
        `[Router] Trying provider ${i + 1}/${orderedProviders.length}: ${config.provider}/${config.model}`
      );

      try {
        const manager = new TokenManager(config, this.env);
        const response = await manager.executeWithRotation(request);

        if (response.success) {
          console.log(`[Router] Success with provider: ${config.provider}/${config.model}`);
          return response;
        }

        lastError = response.error || 'Provider request failed';
        lastStatusCode = response.statusCode || 502;
        console.warn(
          `[Router] Provider ${config.provider}/${config.model} failed with status ${lastStatusCode}`
        );

        if ([400, 413, 422].includes(lastStatusCode)) break;
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        lastStatusCode = getErrorStatusCode(error) || 502;
        console.error(
          `[Router] Provider ${config.provider}/${config.model} failed with status ${lastStatusCode}`
        );

        if ([400, 413, 422].includes(lastStatusCode)) break;
      }
    }

    return {
      success: false,
      error: lastError,
      statusCode: lastStatusCode,
    };
  }

  private sortByPreferredType(
    providers: ProviderConfig[],
    preferredType: 'openai' | 'anthropic'
  ): ProviderConfig[] {
    const preferred = providers.filter(
      (provider) =>
        provider.provider === preferredType ||
        provider.provider === `${preferredType}-compatible`
    );
    const rest = providers.filter(
      (provider) =>
        provider.provider !== preferredType &&
        provider.provider !== `${preferredType}-compatible`
    );
    return [...preferred, ...rest];
  }

  /** Parse and validate routing configuration once per request. */
  private parseRoutesConfig(): RouteConfig {
    try {
      if (!this.env.ROUTES_CONFIG) {
        throw new Error('ROUTES_CONFIG is missing');
      }

      const parsed: unknown = JSON.parse(this.env.ROUTES_CONFIG);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('ROUTES_CONFIG must be a JSON object keyed by model alias');
      }

      const entries = Object.entries(parsed as Record<string, unknown>);
      if (entries.length === 0) {
        throw new Error('ROUTES_CONFIG must contain at least one model route');
      }
      if (entries.length > 500) {
        throw new Error('ROUTES_CONFIG contains too many model routes (maximum 500)');
      }

      const routes: RouteConfig = {};
      for (const [alias, rawProviders] of entries) {
        if (!alias.trim() || alias.length > 200) {
          throw new Error('Model aliases must be non-empty and at most 200 characters');
        }
        if (!Array.isArray(rawProviders) || rawProviders.length === 0) {
          throw new Error(`Route "${alias}" must contain at least one provider`);
        }
        if (rawProviders.length > 20) {
          throw new Error(`Route "${alias}" contains too many providers (maximum 20)`);
        }

        routes[alias] = rawProviders.map((rawProvider, index) => {
          if (!rawProvider || typeof rawProvider !== 'object' || Array.isArray(rawProvider)) {
            throw new Error(`Invalid provider ${index + 1} in route "${alias}"`);
          }

          const provider = rawProvider as Record<string, unknown>;
          if (
            typeof provider.provider !== 'string' ||
            !SUPPORTED_PROVIDERS.has(provider.provider)
          ) {
            throw new Error(`Unsupported provider in route "${alias}" at index ${index}`);
          }
          if (typeof provider.model !== 'string' || !provider.model.trim()) {
            throw new Error(`Provider model is missing in route "${alias}" at index ${index}`);
          }

          const apiKeys = provider.apiKeys === undefined && provider.provider === 'cloudflare-ai'
            ? []
            : provider.apiKeys;
          if (
            !Array.isArray(apiKeys) ||
            !apiKeys.every(
              (key) => typeof key === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(key)
            )
          ) {
            throw new Error(`apiKeys must be an array of environment variable names in route "${alias}"`);
          }
          if (provider.provider !== 'cloudflare-ai' && apiKeys.length === 0) {
            throw new Error(`At least one API key variable is required for route "${alias}"`);
          }

          const compatible =
            provider.provider === 'openai-compatible' ||
            provider.provider === 'anthropic-compatible';
          let baseUrl: string | undefined;
          if (provider.baseUrl !== undefined) {
            if (typeof provider.baseUrl !== 'string') {
              throw new Error(`baseUrl must be a string in route "${alias}"`);
            }
            let parsedUrl: URL;
            try {
              parsedUrl = new URL(provider.baseUrl);
            } catch {
              throw new Error(`baseUrl is not a valid URL in route "${alias}"`);
            }
            if (!['http:', 'https:'].includes(parsedUrl.protocol) || parsedUrl.username || parsedUrl.password) {
              throw new Error(`baseUrl must be an HTTP(S) URL without embedded credentials in route "${alias}"`);
            }
            baseUrl = parsedUrl.toString().replace(/\/$/, '');
          }
          if (compatible && !baseUrl) {
            throw new Error(`baseUrl is required for ${provider.provider} in route "${alias}"`);
          }
          if (provider.grounding !== undefined && typeof provider.grounding !== 'boolean') {
            throw new Error(`grounding must be a boolean in route "${alias}"`);
          }

          return {
            provider: provider.provider as ProviderConfig['provider'],
            model: provider.model,
            apiKeys: apiKeys as string[],
            ...(baseUrl ? { baseUrl } : {}),
            ...(provider.grounding !== undefined ? { grounding: provider.grounding } : {}),
          };
        });
      }

      console.log('[Router] Validated routes:', Object.keys(routes).length);
      return routes;
    } catch (error) {
      console.error(
        '[Router] Failed to parse ROUTES_CONFIG:',
        error instanceof Error ? error.message : 'Unknown configuration error'
      );
      throw new ProxyError(
        `Invalid ROUTES_CONFIG: ${error instanceof Error ? error.message : 'unknown error'}`,
        500,
        'invalid_routes_config'
      );
    }
  }
}
