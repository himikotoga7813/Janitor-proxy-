import { ProviderConfig, Env, OpenAIChatRequest, ProviderResponse } from './types';
import { createProvider } from './providers';
import {
  getErrorStatusCode,
  getProviderTimeoutMs,
  isRetryableError,
  withTimeout,
} from './utils/error-handler';

export class TokenManager {
  constructor(
    private config: ProviderConfig,
    private env: Env
  ) {}

  /** Execute against configured keys in order, rotating on authentication and transient failures. */
  async executeWithRotation(request: OpenAIChatRequest): Promise<ProviderResponse> {
    const provider = createProvider(this.config, this.env);
    const apiKeys = this.getApiKeys();

    if (this.config.provider === 'cloudflare-ai') {
      return provider.chat(request, '');
    }

    if (apiKeys.length === 0) {
      return {
        success: false,
        error: `No usable API keys configured for ${this.config.provider}/${this.config.model}`,
        statusCode: 500,
      };
    }

    let lastError: unknown = null;
    let lastStatusCode = 500;

    for (let i = 0; i < apiKeys.length; i++) {
      const apiKey = apiKeys[i];
      try {
        console.log(
          `[TokenManager] Trying ${this.config.provider}/${this.config.model} with key ${i + 1}/${apiKeys.length}`
        );

        const response = await withTimeout(
          provider.chat(request, apiKey),
          getProviderTimeoutMs(this.env)
        );

        if (response.success) {
          console.log(`[TokenManager] Success with key ${i + 1}/${apiKeys.length}`);
          return response;
        }

        lastError = response.error || 'Provider request failed';
        lastStatusCode = response.statusCode || 500;
        console.warn(
          `[TokenManager] Key ${i + 1}/${apiKeys.length} failed with status ${lastStatusCode}`
        );

        if (
          ![401, 403].includes(lastStatusCode) &&
          !isRetryableError({ statusCode: lastStatusCode })
        ) {
          break;
        }
      } catch (error) {
        lastError = error;
        lastStatusCode = getErrorStatusCode(error) || 500;
        console.error(
          `[TokenManager] Key ${i + 1}/${apiKeys.length} failed with status ${lastStatusCode}`
        );

        if (!isRetryableError(error) && ![401, 403].includes(lastStatusCode)) {
          break;
        }
      }
    }

    return {
      success: false,
      error:
        lastError instanceof Error ? lastError.message : String(lastError || 'All API keys failed'),
      statusCode: lastStatusCode,
    };
  }

  private getApiKeys(): string[] {
    const keys = new Set<string>();

    for (const keyName of this.config.apiKeys) {
      const keyValue = this.env[keyName];
      if (typeof keyValue === 'string' && keyValue.trim()) {
        keys.add(keyValue);
      } else {
        console.warn(`[TokenManager] API key environment variable is unavailable: ${keyName}`);
      }
    }

    return [...keys];
  }
}
