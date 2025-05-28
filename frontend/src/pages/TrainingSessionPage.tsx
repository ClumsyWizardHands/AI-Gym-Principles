import { useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/endpoints'
import { useWebSocket } from '@/hooks/useWebSocket'
import { TrainingProgressDashboard } from '@/components/visualizations'

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
    principles,
    actions,
  } = useWebSocket(sessionId || null)

  useEffect(() => {
    if (status?.status === 'completed') {
      navigate(`/reports/${sessionId}`)
    }
  }, [status?.status, sessionId, navigate])

  // Calculate current behavioral entropy from latest actions
  const currentBehavioralEntropy = useMemo(() => {
    if (actions.length === 0) return 0
    
    // Simple entropy calculation based on action diversity
    const actionCounts: Record<string, number> = {}
    actions.forEach(action => {
      const key = `${action.action}-${action.decision}`
      actionCounts[key] = (actionCounts[key] || 0) + 1
    })
    
    const total = actions.length
    let entropy = 0
    Object.values(actionCounts).forEach(count => {
      const p = count / total
      if (p > 0) entropy -= p * Math.log2(p)
    })
    
    // Normalize to 0-1 range (assuming max entropy of 5 bits)
    return Math.min(entropy / 5, 1)
  }, [actions])

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

      {/* Training Progress Dashboard */}
      <TrainingProgressDashboard 
        status={status}
        actions={actions}
        principlesDiscovered={principles}
        behavioralEntropy={currentBehavioralEntropy}
      />

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
