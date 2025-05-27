import React from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/endpoints'
import { 
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

export function ReportPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()

  const { data: report, isLoading, error } = useQuery({
    queryKey: ['training-report', sessionId],
    queryFn: () => api.getTrainingReport(sessionId!),
    enabled: !!sessionId
  })

  if (!sessionId) {
    return <div>Invalid session ID</div>
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <SpinnerIcon className="animate-spin h-8 w-8 text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading report...</p>
        </div>
      </div>
    )
  }

  if (error || !report) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-gray-600">Failed to load report</p>
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

  // Prepare data for charts
  const principlesData = report.principles_discovered.map(p => ({
    name: p.name.length > 20 ? p.name.substring(0, 20) + '...' : p.name,
    strength: p.strength * 100,
    consistency: p.consistency * 100,
    evidence: p.evidence_count
  }))

  const radarData = report.principles_discovered.slice(0, 6).map(p => ({
    principle: p.name.split(' ').slice(0, 2).join(' '),
    value: p.strength * 100
  }))

  const overallScore = Math.round(
    (report.consistency_score * 0.5 + (1 - report.behavioral_entropy) * 0.5) * 100
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Report</h1>
          <p className="mt-1 text-sm text-gray-600">
            Session completed on {new Date(report.completed_at).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => window.print()}
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            <PrintIcon className="h-4 w-4 mr-2" />
            Print Report
          </button>
          <button
            onClick={() => navigate('/training')}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            New Session
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Overall Score</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-3xl font-bold text-primary-600">{overallScore}</p>
            <p className="ml-1 text-lg text-gray-500">/100</p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Scenarios Completed</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">
            {report.scenarios_completed}
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Principles Discovered</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">
            {report.principles_discovered.length}
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-sm font-medium text-gray-500">Training Duration</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">
            {Math.round(report.duration_seconds / 60)}m
          </p>
        </div>
      </div>

      {/* Summary Text */}
      <div className="bg-white p-6 rounded-lg shadow-soft">
        <h3 className="text-lg font-medium text-gray-900 mb-3">Executive Summary</h3>
        <p className="text-gray-700 leading-relaxed">{report.summary}</p>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Principles Strength Chart */}
        {principlesData.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-soft">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Principle Strength Analysis
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={principlesData.slice(0, 5)} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" domain={[0, 100]} stroke="#6b7280" />
                <YAxis dataKey="name" type="category" stroke="#6b7280" width={120} />
                <Tooltip />
                <Bar dataKey="strength" fill="#0ea5e9" name="Strength %" />
                <Bar dataKey="consistency" fill="#10b981" name="Consistency %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Radar Chart */}
        {radarData.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-soft">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Behavioral Profile
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="principle" stroke="#6b7280" />
                <PolarRadiusAxis domain={[0, 100]} stroke="#6b7280" />
                <Radar
                  name="Strength"
                  dataKey="value"
                  stroke="#0ea5e9"
                  fill="#0ea5e9"
                  fillOpacity={0.3}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Detailed Principles */}
      <div className="bg-white rounded-lg shadow-soft">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">
            Discovered Principles Detail
          </h3>
        </div>
        <div className="divide-y divide-gray-200">
          {report.principles_discovered.length > 0 ? (
            report.principles_discovered.map((principle, index) => (
              <div key={index} className="px-6 py-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-gray-900">
                      {principle.name}
                    </h4>
                    <p className="mt-1 text-sm text-gray-600">
                      {principle.description}
                    </p>
                    <div className="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                      <span>Strength: {(principle.strength * 100).toFixed(0)}%</span>
                      <span>Consistency: {(principle.consistency * 100).toFixed(0)}%</span>
                      <span>Evidence: {principle.evidence_count} actions</span>
                      <span>First seen: {new Date(principle.first_observed).toLocaleString()}</span>
                    </div>
                    <div className="mt-2">
                      <span className="text-xs font-medium text-gray-500">Contexts: </span>
                      {principle.contexts.map((context, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 ml-1"
                        >
                          {context}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="px-6 py-12 text-center">
              <p className="text-sm text-gray-500">No principles discovered</p>
            </div>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="bg-white p-6 rounded-lg shadow-soft">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Key Metrics</h3>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Behavioral Entropy</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {(report.behavioral_entropy * 100).toFixed(2)}% 
              <span className="text-xs text-gray-500 ml-1">
                ({report.behavioral_entropy < 0.3 ? 'Low - Consistent' : 
                  report.behavioral_entropy < 0.7 ? 'Medium - Variable' : 
                  'High - Unpredictable'})
              </span>
            </dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Consistency Score</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {(report.consistency_score * 100).toFixed(2)}%
              <span className="text-xs text-gray-500 ml-1">
                ({report.consistency_score > 0.8 ? 'Excellent' : 
                  report.consistency_score > 0.6 ? 'Good' : 
                  report.consistency_score > 0.4 ? 'Fair' : 'Poor'})
              </span>
            </dd>
          </div>
        </dl>
      </div>
    </div>
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

function PrintIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
    </svg>
  )
}
