// API Types matching the backend models

export interface APIKeyResponse {
  api_key: string
  created_at: string
  expires_at?: string
  usage_limit?: number
}

export interface AgentRegistration {
  name: string
  framework: 'openai' | 'anthropic' | 'langchain' | 'custom' | 'http'
  config: Record<string, any>
  description?: string
}

export interface AgentRegistrationResponse {
  agent_id: string
  name: string
  framework: string
  registered_at: string
  status: string
}

export type ScenarioArchetype = 
  | 'LOYALTY'
  | 'SCARCITY'
  | 'BETRAYAL'
  | 'TRADEOFFS'
  | 'TIME_PRESSURE'
  | 'OBEDIENCE_AUTONOMY'
  | 'INFO_ASYMMETRY'
  | 'REPUTATION_MGMT'
  | 'POWER_DYNAMICS'
  | 'MORAL_HAZARD'

export interface TrainingRequest {
  agent_id: string
  scenario_types?: ScenarioArchetype[]
  num_scenarios?: number
  adaptive?: boolean
}

export interface TrainingResponse {
  session_id: string
  agent_id: string
  status: string
  started_at: string
  estimated_duration_seconds: number
}

export interface TrainingStatus {
  session_id: string
  agent_id: string
  status: 'started' | 'running' | 'completed' | 'failed'
  progress: number
  scenarios_completed: number
  scenarios_total: number
  started_at: string
  updated_at: string
  completed_at?: string
  error_message?: string
}

export interface PrincipleReport {
  name: string
  description: string
  strength: number
  consistency: number
  evidence_count: number
  first_observed: string
  contexts: string[]
}

export interface TrainingReport {
  session_id: string
  agent_id: string
  completed_at: string
  duration_seconds: number
  scenarios_completed: number
  principles_discovered: PrincipleReport[]
  behavioral_entropy: number
  consistency_score: number
  summary: string
}

// WebSocket Event Types
export interface WebSocketMessage {
  type: 'status_update' | 'principle_discovered' | 'action_recorded' | 'error'
  session_id: string
  data: any
}

export interface StatusUpdate {
  progress: number
  scenarios_completed: number
  current_scenario?: string
}

export interface PrincipleDiscovered {
  principle: PrincipleReport
  timestamp: string
}

export interface ActionRecorded {
  action: string
  context: string
  decision: string
  timestamp: string
}

// Frontend specific types
export interface Agent {
  id: string
  name: string
  framework: string
  status: 'active' | 'inactive'
  registeredAt: Date
  lastTrainingAt?: Date
  totalSessions: number
  averageScore?: number
}

export interface Session {
  id: string
  agentId: string
  agentName: string
  status: TrainingStatus['status']
  progress: number
  startedAt: Date
  completedAt?: Date
  scenariosCompleted: number
  scenariosTotal: number
  principlesDiscovered?: number
}

export interface DashboardStats {
  totalAgents: number
  totalSessions: number
  activeSessions: number
  averagePrinciples: number
  averageConsistency: number
}
