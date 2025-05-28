import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, FRAMEWORK_OPTIONS } from '@/api/endpoints'
import { AgentRegistration } from '@/api/types'
import toast from 'react-hot-toast'

export function AgentsPage() {
  const queryClient = useQueryClient()
  const [showModal, setShowModal] = useState(false)
  const [formData, setFormData] = useState<AgentRegistration>({
    name: '',
    framework: 'openai',
    config: {},
    description: ''
  })

  const { data: agents, isLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: api.listAgents
  })

  const registerMutation = useMutation({
    mutationFn: api.registerAgent,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] })
      toast.success('Agent registered successfully')
      setShowModal(false)
      resetForm()
    },
    onError: () => {
      toast.error('Failed to register agent')
    }
  })

  const resetForm = () => {
    setFormData({
      name: '',
      framework: 'openai',
      config: {},
      description: ''
    })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    // Basic validation
    if (!formData.name.trim()) {
      toast.error('Agent name is required')
      return
    }

    // Framework-specific config validation
    if (formData.framework === 'openai' && !formData.config.api_key) {
      toast.error('OpenAI API key is required')
      return
    }

    if (formData.framework === 'anthropic' && !formData.config.api_key) {
      toast.error('Anthropic API key is required')
      return
    }

    registerMutation.mutate(formData)
  }

  const getFrameworkConfigFields = () => {
    switch (formData.framework) {
      case 'openai':
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700">API Key</label>
              <input
                type="password"
                value={formData.config.api_key || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  config: { ...formData.config, api_key: e.target.value }
                })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="sk-..."
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Model</label>
              <select
                value={formData.config.model || 'gpt-4'}
                onChange={(e) => setFormData({
                  ...formData,
                  config: { ...formData.config, model: e.target.value }
                })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
              >
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              </select>
            </div>
          </>
        )
      case 'anthropic':
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700">API Key</label>
              <input
                type="password"
                value={formData.config.api_key || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  config: { ...formData.config, api_key: e.target.value }
                })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="sk-ant-..."
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Model</label>
              <select
                value={formData.config.model || 'claude-3-opus-20240229'}
                onChange={(e) => setFormData({
                  ...formData,
                  config: { ...formData.config, model: e.target.value }
                })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
              >
                <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
              </select>
            </div>
          </>
        )
      case 'langchain':
        return (
          <div>
            <label className="block text-sm font-medium text-gray-700">Configuration JSON</label>
            <textarea
              value={JSON.stringify(formData.config, null, 2)}
              onChange={(e) => {
                try {
                  const config = JSON.parse(e.target.value)
                  setFormData({ ...formData, config })
                } catch {
                  // Invalid JSON, don't update
                }
              }}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm font-mono"
              rows={4}
              placeholder='{"llm": {...}, "memory": {...}}'
            />
          </div>
        )
      case 'custom':
        return (
          <div>
            <label className="block text-sm font-medium text-gray-700">Custom Implementation</label>
            <textarea
              value={formData.config.function_code || ''}
              onChange={(e) => setFormData({
                ...formData,
                config: { ...formData.config, function_code: e.target.value }
              })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm font-mono"
              rows={6}
              placeholder="def agent_function(scenario):\n    # Your implementation here\n    return decision"
            />
            <p className="mt-1 text-xs text-gray-500">
              Provide a Python function that takes a scenario and returns a decision
            </p>
          </div>
        )
      case 'http':
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700">Endpoint URL</label>
              <input
                type="text"
                value={formData.config.endpoint_url || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  config: { ...formData.config, endpoint_url: e.target.value }
                })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="http://localhost:8000/chat"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Method</label>
                <select
                  value={formData.config.method || 'POST'}
                  onChange={(e) => setFormData({
                    ...formData,
                    config: { ...formData.config, method: e.target.value }
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                >
                  <option value="POST">POST</option>
                  <option value="GET">GET</option>
                  <option value="PUT">PUT</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Timeout (seconds)</label>
                <input
                  type="number"
                  value={formData.config.timeout || 30}
                  onChange={(e) => setFormData({
                    ...formData,
                    config: { ...formData.config, timeout: parseInt(e.target.value) }
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                  min="1"
                  max="300"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Request Format</label>
                <select
                  value={formData.config.request_format || 'json'}
                  onChange={(e) => setFormData({
                    ...formData,
                    config: { ...formData.config, request_format: e.target.value }
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                >
                  <option value="json">JSON</option>
                  <option value="form">Form Data</option>
                  <option value="text">Plain Text</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Response Format</label>
                <select
                  value={formData.config.response_format || 'json'}
                  onChange={(e) => setFormData({
                    ...formData,
                    config: { ...formData.config, response_format: e.target.value }
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                >
                  <option value="json">JSON</option>
                  <option value="text">Plain Text</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Authorization Token (optional)</label>
              <input
                type="text"
                value={formData.config.auth_token || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  config: { ...formData.config, auth_token: e.target.value }
                })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="Bearer token..."
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Custom Headers (optional)</label>
              <textarea
                value={JSON.stringify(formData.config.headers || {}, null, 2)}
                onChange={(e) => {
                  try {
                    const headers = JSON.parse(e.target.value)
                    setFormData({ ...formData, config: { ...formData.config, headers } })
                  } catch {
                    // Invalid JSON, don't update
                  }
                }}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm font-mono"
                rows={3}
                placeholder='{"X-Custom-Header": "value"}'
              />
            </div>
          </>
        )
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Agents</h1>
          <p className="mt-1 text-sm text-gray-600">
            Manage your AI agents and their configurations
          </p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
        >
          <PlusIcon className="h-4 w-4 mr-2" />
          Register Agent
        </button>
      </div>

      {/* Agents Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white rounded-lg shadow-soft p-6 animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
              <div className="h-3 bg-gray-200 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : agents && agents.length > 0 ? (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {agents.map((agent) => (
            <div key={agent.agent_id} className="bg-white rounded-lg shadow-soft p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">{agent.name}</h3>
                <StatusBadge status={agent.status} />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex items-center text-gray-500">
                  <CpuChipIcon className="h-4 w-4 mr-2" />
                  {FRAMEWORK_OPTIONS.find(f => f.value === agent.framework)?.label || agent.framework}
                </div>
                <div className="flex items-center text-gray-500">
                  <CalendarIcon className="h-4 w-4 mr-2" />
                  Registered {new Date(agent.registered_at).toLocaleDateString()}
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-200">
                <button className="text-sm font-medium text-primary-600 hover:text-primary-500">
                  View Details →
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-white rounded-lg shadow-soft">
          <RobotIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No agents</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by registering your first agent.
          </p>
          <div className="mt-6">
            <button
              onClick={() => setShowModal(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              Register Agent
            </button>
          </div>
        </div>
      )}

      {/* Registration Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-end justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" onClick={() => setShowModal(false)} />
            
            <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
              <form onSubmit={handleSubmit}>
                <div>
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    Register New Agent
                  </h3>
                  <div className="mt-6 space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Name</label>
                      <input
                        type="text"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                        placeholder="My AI Agent"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700">Framework</label>
                      <select
                        value={formData.framework}
                        onChange={(e) => setFormData({ 
                          ...formData, 
                          framework: e.target.value as any,
                          config: {} // Reset config when framework changes
                        })}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                      >
                        {FRAMEWORK_OPTIONS.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {getFrameworkConfigFields()}

                    <div>
                      <label className="block text-sm font-medium text-gray-700">Description (optional)</label>
                      <textarea
                        value={formData.description || ''}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                        rows={3}
                        placeholder="Describe your agent's purpose..."
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-5 sm:mt-6 sm:grid sm:grid-cols-2 sm:gap-3 sm:grid-flow-row-dense">
                  <button
                    type="submit"
                    disabled={registerMutation.isPending}
                    className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-primary-600 text-base font-medium text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 sm:col-start-2 sm:text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {registerMutation.isPending ? 'Registering...' : 'Register'}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowModal(false)
                      resetForm()
                    }}
                    className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 sm:mt-0 sm:col-start-1 sm:text-sm"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Helper Components
function StatusBadge({ status }: { status: string }) {
  const statusClasses = {
    active: 'bg-green-100 text-green-800',
    inactive: 'bg-gray-100 text-gray-800'
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusClasses[status as keyof typeof statusClasses] || statusClasses.inactive}`}>
      {status}
    </span>
  )
}

// Icons
function PlusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  )
}

function RobotIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  )
}

function CpuChipIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  )
}

function CalendarIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  )
}
