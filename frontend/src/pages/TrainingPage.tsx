import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { api, SCENARIO_TYPES } from '@/api/endpoints'
import { TrainingRequest } from '@/api/types'
import toast from 'react-hot-toast'

export function TrainingPage() {
  const navigate = useNavigate()
  const [selectedAgent, setSelectedAgent] = useState('')
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([])
  const [numScenarios, setNumScenarios] = useState(10)
  const [adaptive, setAdaptive] = useState(true)

  const { data: agents, isLoading: agentsLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: api.listAgents
  })

  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.listTrainingSessions()
  })

  const startTrainingMutation = useMutation({
    mutationFn: api.startTraining,
    onSuccess: (data) => {
      toast.success('Training session started!')
      navigate(`/training/${data.session_id}`)
    },
    onError: () => {
      toast.error('Failed to start training session')
    }
  })

  const handleStartTraining = () => {
    if (!selectedAgent) {
      toast.error('Please select an agent')
      return
    }

    const request: TrainingRequest = {
      agent_id: selectedAgent,
      scenario_types: selectedScenarios.length > 0 ? selectedScenarios as any : undefined,
      num_scenarios: numScenarios,
      adaptive
    }

    startTrainingMutation.mutate(request)
  }

  const toggleScenario = (scenario: string) => {
    setSelectedScenarios(prev =>
      prev.includes(scenario)
        ? prev.filter(s => s !== scenario)
        : [...prev, scenario]
    )
  }

  const activeSessions = sessions?.filter(s => s.status === 'running' || s.status === 'started') || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Training Sessions</h1>
        <p className="mt-1 text-sm text-gray-600">
          Configure and start training sessions for your agents
        </p>
      </div>

      {/* Active Sessions Alert */}
      {activeSessions.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <InformationCircleIcon className="h-5 w-5 text-blue-400" />
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">
                Active Sessions
              </h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>You have {activeSessions.length} active training session{activeSessions.length > 1 ? 's' : ''}.</p>
                <div className="mt-2 space-y-1">
                  {activeSessions.map(session => (
                    <button
                      key={session.session_id}
                      onClick={() => navigate(`/training/${session.session_id}`)}
                      className="text-blue-600 hover:text-blue-500 underline text-sm"
                    >
                      View session {session.session_id.slice(0, 8)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Training Configuration */}
      <div className="bg-white shadow-soft rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-6">New Training Session</h2>
        
        <div className="space-y-6">
          {/* Agent Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Agent
            </label>
            {agentsLoading ? (
              <div className="h-10 bg-gray-200 rounded animate-pulse" />
            ) : agents && agents.length > 0 ? (
              <select
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
              >
                <option value="">Choose an agent...</option>
                {agents.map(agent => (
                  <option key={agent.agent_id} value={agent.agent_id}>
                    {agent.name} ({agent.framework})
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-sm text-gray-500">
                No agents registered. <button onClick={() => navigate('/agents')} className="text-primary-600 hover:text-primary-500">Create one first →</button>
              </p>
            )}
          </div>

          {/* Scenario Types */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Scenario Types
            </label>
            <p className="text-xs text-gray-500 mb-3">
              Select specific scenarios or leave empty for all types
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {SCENARIO_TYPES.map(scenario => (
                <label
                  key={scenario}
                  className={`
                    relative flex items-center px-3 py-2 rounded-md border cursor-pointer
                    ${selectedScenarios.includes(scenario)
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-300 hover:bg-gray-50'
                    }
                  `}
                >
                  <input
                    type="checkbox"
                    checked={selectedScenarios.includes(scenario)}
                    onChange={() => toggleScenario(scenario)}
                    className="sr-only"
                  />
                  <span className="text-sm font-medium">
                    {scenario.split('_').map(word => 
                      word.charAt(0) + word.slice(1).toLowerCase()
                    ).join(' ')}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Number of Scenarios */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Scenarios
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="1"
                max="100"
                value={numScenarios}
                onChange={(e) => setNumScenarios(Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm font-medium text-gray-900 w-12 text-right">
                {numScenarios}
              </span>
            </div>
          </div>

          {/* Adaptive Mode */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="adaptive"
              checked={adaptive}
              onChange={(e) => setAdaptive(e.target.checked)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="adaptive" className="ml-2 block text-sm text-gray-900">
              Enable adaptive scenario generation
            </label>
          </div>

          {/* Start Button */}
          <div className="pt-4">
            <button
              onClick={handleStartTraining}
              disabled={!selectedAgent || startTrainingMutation.isPending}
              className="w-full sm:w-auto inline-flex justify-center items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {startTrainingMutation.isPending ? (
                <>
                  <SpinnerIcon className="animate-spin -ml-1 mr-3 h-5 w-5" />
                  Starting...
                </>
              ) : (
                <>
                  <PlayIcon className="h-5 w-5 mr-2" />
                  Start Training
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Recent Sessions */}
      <div className="bg-white shadow-soft rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Recent Sessions</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {sessionsLoading ? (
            <div className="px-6 py-4">
              <div className="animate-pulse space-y-3">
                <div className="h-4 bg-gray-200 rounded w-3/4" />
                <div className="h-3 bg-gray-200 rounded w-1/2" />
              </div>
            </div>
          ) : sessions && sessions.length > 0 ? (
            sessions.slice(0, 10).map((session) => (
              <div key={session.session_id} className="px-6 py-4 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center">
                      <p className="text-sm font-medium text-gray-900">
                        Session {session.session_id.slice(0, 8)}
                      </p>
                      <StatusBadge status={session.status} className="ml-3" />
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      {session.scenarios_completed} / {session.scenarios_total} scenarios • 
                      Started {new Date(session.started_at).toLocaleString()}
                    </p>
                  </div>
                  <div className="ml-4 flex items-center space-x-2">
                    {session.status === 'completed' ? (
                      <button
                        onClick={() => navigate(`/reports/${session.session_id}`)}
                        className="text-sm font-medium text-primary-600 hover:text-primary-500"
                      >
                        View Report
                      </button>
                    ) : (
                      <button
                        onClick={() => navigate(`/training/${session.session_id}`)}
                        className="text-sm font-medium text-primary-600 hover:text-primary-500"
                      >
                        Monitor
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="px-6 py-12 text-center">
              <p className="text-sm text-gray-500">No training sessions yet</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Helper Components
function StatusBadge({ status, className = '' }: { status: string; className?: string }) {
  const statusClasses = {
    started: 'bg-blue-100 text-blue-800',
    running: 'bg-green-100 text-green-800',
    completed: 'bg-gray-100 text-gray-800',
    failed: 'bg-red-100 text-red-800'
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusClasses[status as keyof typeof statusClasses] || statusClasses.started} ${className}`}>
      {status}
    </span>
  )
}

// Icons
function InformationCircleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function PlayIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  )
}
