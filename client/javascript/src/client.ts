/**
 * AI Principles Gym JavaScript/TypeScript Client
 */

import {
  APIKeyResponse,
  AgentConfig,
  AgentRegistration,
  TrainingSessionResponse,
  TrainingStatus,
  PrinciplesReport,
  ClientOptions,
  RetryOptions,
  ProgressCallback,
  AuthenticationError,
  RateLimitError,
  ResourceNotFoundError,
  TrainingError,
  PrinciplesGymError
} from './types';

export class PrinciplesGymClient {
  private baseURL: string;
  private apiKey?: string;
  private timeout: number;
  private retryOptions: RetryOptions;
  private headers: Record<string, string>;

  constructor(options: ClientOptions = {}) {
    this.baseURL = options.baseURL || 'http://localhost:8000';
    this.apiKey = options.apiKey;
    this.timeout = options.timeout || 30000;
    this.retryOptions = {
      attempts: options.retryAttempts || 3,
      delay: options.retryDelay || 1000,
      maxDelay: 10000,
      factor: 2
    };

    this.headers = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      this.headers['X-API-Key'] = this.apiKey;
    }
  }

  /**
   * Generate a new API key
   */
  async generateApiKey(
    userId: string,
    usageLimit: number = 10000,
    expiresInDays: number = 30
  ): Promise<string> {
    const response = await this.request<APIKeyResponse>('/api/keys', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        usage_limit: usageLimit,
        expires_in_days: expiresInDays
      })
    });

    return response.api_key;
  }

  /**
   * Register a new agent
   */
  async registerAgent(
    agentId: string,
    framework: string,
    config: AgentConfig = {}
  ): Promise<AgentRegistration> {
    return await this.request<AgentRegistration>('/api/agents/register', {
      method: 'POST',
      body: JSON.stringify({
        agent_id: agentId,
        framework,
        config
      })
    });
  }

  /**
   * Start a training session
   */
  async startTraining(
    agentId: string,
    numScenarios: number = 20
  ): Promise<string> {
    const response = await this.request<TrainingSessionResponse>('/api/training/start', {
      method: 'POST',
      body: JSON.stringify({
        agent_id: agentId,
        num_scenarios: numScenarios
      })
    });

    return response.session_id;
  }

  /**
   * Get training status
   */
  async getTrainingStatus(sessionId: string): Promise<TrainingStatus> {
    return await this.request<TrainingStatus>(`/api/training/status/${sessionId}`);
  }

  /**
   * Wait for training completion with progress updates
   */
  async waitForCompletion(
    sessionId: string,
    pollInterval: number = 1000,
    progressCallback?: ProgressCallback
  ): Promise<void> {
    while (true) {
      const status = await this.getTrainingStatus(sessionId);

      if (progressCallback && status.progress) {
        const { completed, total } = status.progress;
        const progress = total > 0 ? (completed / total) * 100 : 0;
        progressCallback(progress, completed, total);
      }

      if (status.status === 'completed') {
        return;
      }

      if (status.status === 'failed') {
        throw new TrainingError(
          status.error || 'Training failed',
          { session_id: sessionId, status }
        );
      }

      // Wait before next poll
      await this.sleep(pollInterval);
    }
  }

  /**
   * Get principles report
   */
  async getReport(sessionId: string): Promise<PrinciplesReport> {
    return await this.request<PrinciplesReport>(`/api/reports/${sessionId}`);
  }

  /**
   * List registered agents
   */
  async listAgents(): Promise<AgentRegistration[]> {
    return await this.request<AgentRegistration[]>('/api/agents');
  }

  /**
   * List training sessions
   */
  async listTrainingSessions(status?: string): Promise<TrainingSessionResponse[]> {
    const params = status ? `?status=${status}` : '';
    return await this.request<TrainingSessionResponse[]>(`/api/training/sessions${params}`);
  }

  /**
   * Update API key for authenticated requests
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
    this.headers['X-API-Key'] = apiKey;
  }

  /**
   * Make HTTP request with retry logic
   */
  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    const requestOptions: RequestInit = {
      ...options,
      headers: {
        ...this.headers,
        ...options.headers
      },
      signal: controller.signal
    };

    let lastError: Error | null = null;
    let retryDelay = this.retryOptions.delay;

    for (let attempt = 0; attempt < this.retryOptions.attempts; attempt++) {
      try {
        const response = await fetch(url, requestOptions);
        clearTimeout(timeoutId);

        if (!response.ok) {
          await this.handleErrorResponse(response);
        }

        return await response.json();
      } catch (error) {
        clearTimeout(timeoutId);
        lastError = error as Error;

        // Don't retry on certain errors
        if (
          error instanceof AuthenticationError ||
          error instanceof ResourceNotFoundError ||
          (error instanceof PrinciplesGymError && error.statusCode === 400)
        ) {
          throw error;
        }

        // Handle rate limiting
        if (error instanceof RateLimitError) {
          retryDelay = error.retryAfter * 1000;
        }

        // Don't retry on last attempt
        if (attempt === this.retryOptions.attempts - 1) {
          throw error;
        }

        // Wait before retry
        await this.sleep(retryDelay);

        // Exponential backoff
        retryDelay = Math.min(
          retryDelay * (this.retryOptions.factor || 2),
          this.retryOptions.maxDelay || 10000
        );
      }
    }

    throw lastError || new Error('Request failed');
  }

  /**
   * Handle error responses
   */
  private async handleErrorResponse(response: Response): Promise<never> {
    let errorData: any;
    try {
      errorData = await response.json();
    } catch {
      errorData = { detail: response.statusText };
    }

    const message = errorData.detail || errorData.message || 'Request failed';

    switch (response.status) {
      case 401:
        throw new AuthenticationError(message);
      case 404:
        const match = response.url.match(/\/([\w-]+)\/([\w-]+)$/);
        if (match) {
          throw new ResourceNotFoundError(match[1], match[2]);
        }
        throw new ResourceNotFoundError('Resource', 'unknown');
      case 429:
        const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
        throw new RateLimitError(retryAfter, message);
      default:
        throw new PrinciplesGymError(message, response.status, errorData);
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Support for both CommonJS and ES modules
export default PrinciplesGymClient;
