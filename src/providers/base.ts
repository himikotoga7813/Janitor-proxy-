import { OpenAIChatRequest, ProviderResponse } from '../types';
import { getErrorStatusCode } from '../utils/error-handler';

export interface AIProvider {
  /**
   * Send a chat completion request.
   * @param request OpenAI-format request
   * @param apiKey API key to use
   * @returns Provider response with either response object or stream
   */
  chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse>;
}

export abstract class BaseProvider implements AIProvider {
  constructor(
    protected model: string,
    protected baseUrl?: string
  ) {}

  abstract chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse>;

  protected handleError(error: unknown, context: string): ProviderResponse {
    console.error(`[${context}] Error:`, error);

    const message =
      error instanceof Error
        ? error.message
        : error && typeof error === 'object' && 'message' in error &&
            typeof (error as { message?: unknown }).message === 'string'
          ? (error as { message: string }).message
          : typeof error === 'string'
            ? error
            : 'Unknown provider error';

    return {
      success: false,
      error: message,
      statusCode: getErrorStatusCode(error) || 500,
    };
  }
}
