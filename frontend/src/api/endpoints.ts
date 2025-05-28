import { apiRequest } from './client'
import { 
  APIKeyResponse, 
  AgentRegistration, 
  AgentRegistrationResponse,
  TrainingRequest,
  TrainingResponse,
  TrainingStatus,
  TrainingReport,
  ScenarioArchetype
} from './types'

// API endpoints
export const api = {
  // API Keys
  generateApiKey: async (usageLimit?: number, expiresInDays?: number): Promise<APIKeyResponse> => {
    return apiRequest<APIKeyResponse>({
      method: 'POST',
      url: '/api/keys',
      params: { usage_limit: usageLimit, expires_in_days: expiresInDays }
    })
  },

  // Agents
  registerAgent: async (data: AgentRegistration): Promise<AgentRegistrationResponse> => {
    return apiRequest<AgentRegistrationResponse>({
      method: 'POST',
      url: '/api/agents/register',
      data
    })
  },

  listAgents: async (): Promise<AgentRegistrationResponse[]> => {
    return apiRequest<AgentRegistrationResponse[]>({
      method: 'GET',
      url: '/api/agents'
    })
  },

  // Training
  startTraining: async (data: TrainingRequest): Promise<TrainingResponse> => {
    return apiRequest<TrainingResponse>({
      method: 'POST',
      url: '/api/training/start',
      data
    })
  },

  getTrainingStatus: async (sessionId: string): Promise<TrainingStatus> => {
    return apiRequest<TrainingStatus>({
      method: 'GET',
      url: `/api/training/status/${sessionId}`
    })
  },

  getTrainingReport: async (sessionId: string): Promise<TrainingReport> => {
    return apiRequest<TrainingReport>({
      method: 'GET',
      url: `/api/reports/${sessionId}`
    })
  },

  listTrainingSessions: async (statusFilter?: string): Promise<TrainingStatus[]> => {
    return apiRequest<TrainingStatus[]>({
      method: 'GET',
      url: '/api/training/sessions',
      params: statusFilter ? { status_filter: statusFilter } : undefined
    })
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; version: string }> => {
    return apiRequest({
      method: 'GET',
      url: '/api/health'
    })
  },

  // Metrics
  getMetrics: async (): Promise<any> => {
    return apiRequest({
      method: 'GET',
      url: '/api/metrics'
    })
  }
}

// Helper to get available scenario types
export const SCENARIO_TYPES: ScenarioArchetype[] = [
  'LOYALTY',
  'SCARCITY', 
  'BETRAYAL',
  'TRADEOFFS',
  'TIME_PRESSURE',
  'OBEDIENCE_AUTONOMY',
  'INFO_ASYMMETRY',
  'REPUTATION_MGMT',
  'POWER_DYNAMICS',
  'MORAL_HAZARD'
]

// Framework options
export const FRAMEWORK_OPTIONS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'langchain', label: 'LangChain' },
  { value: 'custom', label: 'Custom' },
  { value: 'http', label: 'HTTP Endpoint' }
] as const
