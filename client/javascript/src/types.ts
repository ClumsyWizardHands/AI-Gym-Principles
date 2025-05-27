/**
 * TypeScript interfaces for AI Principles Gym client
 */

export interface APIKeyResponse {
  api_key: string;
  expires_at: string;
  usage_limit: number;
}

export interface AgentConfig {
  [key: string]: any;
}

export interface AgentRegistration {
  agent_id: string;
  framework: string;
  config: AgentConfig;
  registered_at: string;
}

export interface TrainingSessionResponse {
  session_id: string;
  agent_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  num_scenarios: number;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface TrainingStatus {
  session_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: {
    completed: number;
    total: number;
    current_scenario?: string;
  };
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface Principle {
  id: string;
  description: string;
  strength: number;
  evidence_count: number;
  context_distribution: Record<string, number>;
  first_observed: string;
  last_observed: string;
  contradictions?: string[];
}

export interface PrinciplesReport {
  session_id: string;
  agent_id: string;
  total_actions: number;
  behavioral_entropy: number;
  principles: Principle[];
  personality_analysis?: {
    traits: Record<string, number>;
    decision_style: string;
    social_orientation: string;
    concerns?: string[];
  };
  temporal_patterns?: Array<{
    pattern: string;
    frequency: number;
    contexts: string[];
  }>;
  generated_at: string;
}

export interface WebSocketMessage {
  type: 'training_progress' | 'principle_discovered' | 'scenario_completed' | 'error';
  session_id: string;
  data: any;
  timestamp: string;
}

export interface ClientOptions {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  onError?: (error: Error) => void;
}

export interface RetryOptions {
  attempts: number;
  delay: number;
  maxDelay?: number;
  factor?: number;
}

export type ProgressCallback = (progress: number, completed: number, total: number) => void;

export class PrinciplesGymError extends Error {
  constructor(message: string, public statusCode?: number, public details?: any) {
    super(message);
    this.name = 'PrinciplesGymError';
  }
}

export class AuthenticationError extends PrinciplesGymError {
  constructor(message: string = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends PrinciplesGymError {
  constructor(public retryAfter: number, message: string = 'Rate limit exceeded') {
    super(message, 429);
    this.name = 'RateLimitError';
  }
}

export class ResourceNotFoundError extends PrinciplesGymError {
  constructor(resource: string, id: string) {
    super(`${resource} not found: ${id}`, 404);
    this.name = 'ResourceNotFoundError';
  }
}

export class TrainingError extends PrinciplesGymError {
  constructor(message: string, details?: any) {
    super(message, 500, details);
    this.name = 'TrainingError';
  }
}
