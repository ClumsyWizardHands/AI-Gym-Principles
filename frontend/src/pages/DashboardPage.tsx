import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '@/api/endpoints'
import { DashboardStats } from '@/api/types'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts'

export function DashboardPage() {
  const { data: agents, isLoading: agentsLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: api.listAgents
  })

  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.listTrainingSessions()
  })

  // Calculate stats
  const stats: DashboardStats = React.useMemo(() => {
    if (!agents || !sessions) return {
      totalAgents: 0,
      totalSessions: 0,
      activeSessions: 0,
      averagePrinciples: 0,
      averageConsistency: 0
    }

    const activeSessions = sessions.filter(s => s.status === 'running').length
    
    return {
      totalAgents: agents.length,
      totalSessions: sessions.length,
      activeSessions,
      averagePrinciples: 0, // Would need to calculate from reports
      averageConsistency: 0 // Would need to calculate from reports
    }
  }, [agents, sessions])

  // Mock data for charts
  const progressData = [
    { name: 'Mon', sessions: 4 },
    { name: 'Tue', sessions: 3 },
    { name: 'Wed', sessions: 7 },
    { name: 'Thu', sessions: 5 },
    { name: 'Fri', sessions: 9 },
    { name: 'Sat', sessions: 6 },
    { name: 'Sun', sessions: 8 },
  ]

  const principlesData = [
    { name: 'Loyalty', count: 12 },
    { name: 'Scarcity', count: 8 },
    { name: 'Betrayal', count: 15 },
    { name: 'Tradeoffs', count: 10 },
    { name: 'Time Pressure', count: 7 },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600">
          Overview of your AI training activities
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Agents"
          value={stats.totalAgents}
          icon={RobotIcon}
          loading={agentsLoading}
        />
        <StatCard
          title="Total Sessions"
          value={stats.totalSessions}
          icon={ChartBarIcon}
          loading={sessionsLoading}
        />
        <StatCard
          title="Active Sessions"
          value={stats.activeSessions}
          icon={SparklesIcon}
          loading={sessionsLoading}
          variant="success"
        />
        <StatCard
          title="Avg. Consistency"
          value={`${(stats.averageConsistency * 100).toFixed(1)}%`}
          icon={TrendingUpIcon}
          loading={false}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Sessions Over Time */}
        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Training Sessions This Week
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={progressData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="sessions" 
                stroke="#0ea5e9" 
                fill="#0ea5e9" 
                fillOpacity={0.1}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Principles Distribution */}
        <div className="bg-white p-6 rounded-lg shadow-soft">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Top Discovered Principles
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={principlesData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" stroke="#6b7280" />
              <YAxis dataKey="name" type="category" stroke="#6b7280" width={100} />
              <Tooltip />
              <Bar dataKey="count" fill="#0ea5e9" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Sessions */}
      <div className="bg-white shadow-soft rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">Recent Sessions</h3>
            <Link
              to="/training"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all →
            </Link>
          </div>
        </div>
        <div className="divide-y divide-gray-200">
          {sessions?.slice(0, 5).map((session) => (
            <div key={session.session_id} className="px-6 py-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    Session {session.session_id.slice(0, 8)}
                  </p>
                  <p className="text-sm text-gray-500">
                    {session.scenarios_completed} / {session.scenarios_total} scenarios
                  </p>
                </div>
                <div className="flex items-center space-x-4">
                  <StatusBadge status={session.status} />
                  <Link
                    to={`/training/${session.session_id}`}
                    className="text-sm font-medium text-primary-600 hover:text-primary-500"
                  >
                    View
                  </Link>
                </div>
              </div>
            </div>
          ))}
          {(!sessions || sessions.length === 0) && (
            <div className="px-6 py-12 text-center">
              <p className="text-sm text-gray-500">No training sessions yet</p>
              <Link
                to="/training"
                className="mt-2 inline-flex items-center text-sm font-medium text-primary-600 hover:text-primary-500"
              >
                Start your first session →
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Helper Components
interface StatCardProps {
  title: string
  value: string | number
  icon: React.FC<{ className?: string }>
  loading?: boolean
  variant?: 'default' | 'success' | 'warning'
}

function StatCard({ title, value, icon: Icon, loading, variant = 'default' }: StatCardProps) {
  const variantClasses = {
    default: 'text-gray-900',
    success: 'text-success',
    warning: 'text-warning'
  }

  return (
    <div className="bg-white overflow-hidden shadow-soft rounded-lg">
      <div className="p-5">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <Icon className={`h-6 w-6 ${variantClasses[variant]}`} />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
              <dd className={`text-2xl font-semibold ${variantClasses[variant]}`}>
                {loading ? (
                  <div className="h-8 w-20 bg-gray-200 rounded animate-pulse" />
                ) : (
                  value
                )}
              </dd>
            </dl>
          </div>
        </div>
      </div>
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
function RobotIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  )
}

function ChartBarIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  )
}

function SparklesIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
    </svg>
  )
}

function TrendingUpIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  )
}
