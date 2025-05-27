import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/authStore'
import { api } from '@/api/endpoints'
import toast from 'react-hot-toast'

export function HomePage() {
  const navigate = useNavigate()
  const { setApiKey } = useAuthStore()
  const [isLoading, setIsLoading] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [showApiKeyInput, setShowApiKeyInput] = useState(false)

  const handleGenerateApiKey = async () => {
    setIsLoading(true)
    try {
      const response = await api.generateApiKey()
      setApiKey(response.api_key)
      toast.success('API key generated successfully!')
      navigate('/dashboard')
    } catch (error) {
      toast.error('Failed to generate API key')
    } finally {
      setIsLoading(false)
    }
  }

  const handleUseExistingKey = () => {
    if (!apiKeyInput.trim()) {
      toast.error('Please enter an API key')
      return
    }
    setApiKey(apiKeyInput)
    toast.success('API key saved')
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100">
      <div className="flex min-h-screen flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <div className="flex justify-center">
            <div className="h-16 w-16 rounded-full bg-primary-600 flex items-center justify-center">
              <svg className="h-10 w-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
          </div>
          <h2 className="mt-6 text-center text-3xl font-bold tracking-tight text-gray-900">
            AI Principles Gym
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Train AI agents to develop behavioral principles through experience
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-white py-8 px-4 shadow-soft sm:rounded-lg sm:px-10">
            {!showApiKeyInput ? (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">Get Started</h3>
                  <p className="mt-1 text-sm text-gray-600">
                    Choose how you want to access the platform
                  </p>
                </div>

                <div className="space-y-3">
                  <button
                    onClick={handleGenerateApiKey}
                    disabled={isLoading}
                    className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isLoading ? 'Generating...' : 'Generate New API Key'}
                  </button>

                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-gray-300" />
                    </div>
                    <div className="relative flex justify-center text-sm">
                      <span className="px-2 bg-white text-gray-500">Or</span>
                    </div>
                  </div>

                  <button
                    onClick={() => setShowApiKeyInput(true)}
                    className="w-full flex justify-center py-3 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
                  >
                    Use Existing API Key
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <button
                    onClick={() => setShowApiKeyInput(false)}
                    className="flex items-center text-sm text-gray-600 hover:text-gray-900"
                  >
                    <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Back
                  </button>
                </div>

                <div>
                  <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700">
                    API Key
                  </label>
                  <div className="mt-1">
                    <input
                      id="apiKey"
                      type="password"
                      value={apiKeyInput}
                      onChange={(e) => setApiKeyInput(e.target.value)}
                      placeholder="sk-..."
                      className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>
                </div>

                <button
                  onClick={handleUseExistingKey}
                  className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
                >
                  Continue
                </button>
              </div>
            )}

            <div className="mt-6">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-white text-gray-500">Features</span>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-2 gap-3 text-sm">
                <div className="flex items-center">
                  <svg className="h-4 w-4 text-success mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-gray-700">Real-time monitoring</span>
                </div>
                <div className="flex items-center">
                  <svg className="h-4 w-4 text-success mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-gray-700">Multi-framework</span>
                </div>
                <div className="flex items-center">
                  <svg className="h-4 w-4 text-success mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-gray-700">10 scenario types</span>
                </div>
                <div className="flex items-center">
                  <svg className="h-4 w-4 text-success mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-gray-700">Principle analysis</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
