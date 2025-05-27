import React, { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/endpoints'
import { useWebSocket } from '@/hooks/useWebSocket'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export function TrainingSessionPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  
  const { data: status, isLoading } = useQuery({
    queryKey: ['training-status', sessionId],
    queryFn: () => api.getTrainingStatus(sessionId!),
    refetchInterval: 2000, // Poll every 2 seconds
    enabled: !!sessionId
  })

  const {
    isConnected,
    statusUpdates,
    principles,
    actions,
  } = useWebSocket(sessionId || null)

  useEffect(() => {
    if (status?.status === 'completed') {
      navigate(`/reports/${sessionId}`)
    }
  }, [status?.status, sessionId, navigate])

  if (!sessionId) {
    return <div>Invalid session ID</div>
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <SpinnerIcon className="animate-spin h-8 w-8 text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading session...</p>
        </div>
      </div>
    )
  }

  if (!status) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-gray-600">Session not found</p>
          <button
            onClick={() => navigate('/training')}
            className="mt-4 text-primary-600 hover:text-primary-500"
          >
            Back to Training
          </button>
        </div>
      </div>
    )
  }

  const progressData = statusUpdates.map((update, index) => ({
    time: index,
    progress: update.progress * 100,
    scenarios: update.scenarios_completed
  }))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Session Monitor</h1>
          <p className="mt-1 text-sm text-gray-600">
            Session ID: {sessionId.slice(0, 8)}
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <ConnectionStatus isConnected={isConnected} />
          <StatusBadge status={status.status} />
        </div>
      </div>

      {/* Progress Overview */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Progress</h3>
          <div className="mt-2">
            <div className="flex items-baseline">
              <p className="text-2xl font-semibold text-gray-900">
                {Math.round(status.progress * 100)}%
              </p>
              <p className="ml-2 text-sm text-gray-500">
                {status.scenarios_completed} / {status.scenarios_total}
              </p>
            </div>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${status.progress * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Principles Discovered</h3>
          <p className="mt-2 text-2xl font-semibold text-gray-900">
            {principles.length}
          </p>
          <p className="mt-1 text-sm text-gray-500">
            {principles.length > 0 ? 'Analyzing patterns...' : 'Gathering data...'}
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Actions Recorded</h3>
          <p className="mt-2 text-2xl font-semibold text-gray-900">
            {actions.length}
          </p>
          <p className="mt-1 text-sm text-gray-500">
            Real-time tracking
          </p>
        </div>
      </div>

      {/* Progress Chart */}
      {progressData.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Progress Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={progressData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="progress" 
                stroke="#0ea5e9" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Live Feed */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Actions */}
        <div className="bg-white rounded-lg shadow-soft">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Recent Actions</h3>
          </div>
          <div className="max-h-96 overflow-y-auto">
            {actions.length > 0 ? (
              <div className="divide-y divide-gray-200">
                {actions.slice(-10).reverse().map((action, index) => (
                  <div key={index} className="px-6 py-3">
                    <p className="text-sm font-medium text-gray-900">{action.action}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Context: {action.context} • Decision: {action.decision}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="px-6 py-12 text-center">
                <p className="text-sm text-gray-500">Waiting for actions...</p>
              </div>
            )}
          </div>
        </div>

        {/* Discovered Principles */}
        <div className="bg-white rounded-lg shadow-soft">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Discovered Principles</h3>
          </div>
          <div className="max-h-96 overflow-y-auto">
            {principles.length > 0 ? (
              <div className="divide-y divide-gray-200">
                {principles.map((principle, index) => (
                  <div key={index} className="px-6 py-3">
                    <p className="text-sm font-medium text-gray-900">
                      {principle.principle.name}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Strength: {(principle.principle.strength * 100).toFixed(0)}% • 
                      Evidence: {principle.principle.evidence_count}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="px-6 py-12 text-center">
                <p className="text-sm text-gray-500">No principles discovered yet</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      {status.status === 'failed' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <ExclamationTriangleIcon className="h-5 w-5 text-red-400" />
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Training Failed</h3>
              <p className="mt-2 text-sm text-red-700">
                {status.error_message || 'An unexpected error occurred'}
              </p>
              <div className="mt-4">
                <button
                  onClick={() => navigate('/training')}
                  className="text-sm font-medium text-red-600 hover:text-red-500"
                >
                  Start New Session →
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Helper Components
function ConnectionStatus({ isConnected }: { isConnected: boolean }) {
  return (
    <div className="flex items-center">
      <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-gray-400'}`} />
      <span className="ml-2 text-sm text-gray-600">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const statusClasses = {
    started: 'bg-blue-100 text-blue-800',
    running: 'bg-green-100 text-green-800',
    completed: 'bg-gray-100 text-gray-800',
    failed: 'bg-red-100 text-red-800'
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusClasses[status as keyof typeof statusClasses] || statusClasses.started}`}>
      {status}
    </span>
  )
}

// Icons
function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  )
}

function ExclamationTriangleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  )
}
