import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/authStore'
import toast from 'react-hot-toast'

export function SettingsPage() {
  const navigate = useNavigate()
  const { apiKey, clearAuth } = useAuthStore()
  const [showApiKey, setShowApiKey] = useState(false)
  const [apiUrl, setApiUrl] = useState(
    import.meta.env.VITE_API_URL || 'http://localhost:8000'
  )
  const [wsUrl, setWsUrl] = useState(
    import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
  )

  const handleSaveSettings = () => {
    // In a real app, you'd save these to environment variables or a config file
    toast.success('Settings saved (in memory only for this demo)')
  }

  const handleLogout = () => {
    clearAuth()
    navigate('/')
    toast.success('Logged out successfully')
  }

  const handleCopyApiKey = () => {
    if (apiKey) {
      navigator.clipboard.writeText(apiKey)
      toast.success('API key copied to clipboard')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="mt-1 text-sm text-gray-600">
          Manage your application settings and preferences
        </p>
      </div>

      {/* Settings Sections */}
      <div className="space-y-6">
        {/* API Configuration */}
        <div className="bg-white shadow-soft rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">API Configuration</h3>
          </div>
          <div className="px-6 py-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                API URL
              </label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                The base URL for the AI Principles Gym API
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                WebSocket URL
              </label>
              <input
                type="text"
                value={wsUrl}
                onChange={(e) => setWsUrl(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                The WebSocket URL for real-time updates
              </p>
            </div>

            <div className="pt-2">
              <button
                onClick={handleSaveSettings}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>

        {/* Authentication */}
        <div className="bg-white shadow-soft rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Authentication</h3>
          </div>
          <div className="px-6 py-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Current API Key
              </label>
              <div className="mt-1 flex rounded-md shadow-sm">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={apiKey || ''}
                  readOnly
                  className="flex-1 block w-full rounded-none rounded-l-md border-gray-300 bg-gray-50 sm:text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="inline-flex items-center px-3 rounded-r-md border border-l-0 border-gray-300 bg-gray-50 text-gray-500 text-sm hover:bg-gray-100"
                >
                  {showApiKey ? (
                    <EyeSlashIcon className="h-4 w-4" />
                  ) : (
                    <EyeIcon className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={handleCopyApiKey}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                <ClipboardCopyIcon className="h-4 w-4 mr-2" />
                Copy API Key
              </button>
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-red-700 bg-red-100 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                <LogoutIcon className="h-4 w-4 mr-2" />
                Logout
              </button>
            </div>
          </div>
        </div>

        {/* About */}
        <div className="bg-white shadow-soft rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">About</h3>
          </div>
          <div className="px-6 py-4">
            <dl className="space-y-4">
              <div>
                <dt className="text-sm font-medium text-gray-500">Version</dt>
                <dd className="mt-1 text-sm text-gray-900">1.0.0</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Environment</dt>
                <dd className="mt-1 text-sm text-gray-900">
                  {import.meta.env.MODE === 'production' ? 'Production' : 'Development'}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Documentation</dt>
                <dd className="mt-1 text-sm text-primary-600 hover:text-primary-500">
                  <a href="https://github.com/yourusername/ai-principles-gym" target="_blank" rel="noopener noreferrer">
                    View on GitHub →
                  </a>
                </dd>
              </div>
            </dl>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="bg-white shadow-soft rounded-lg border-2 border-red-200">
          <div className="px-6 py-4 border-b border-red-200 bg-red-50">
            <h3 className="text-lg font-medium text-red-900">Danger Zone</h3>
          </div>
          <div className="px-6 py-4">
            <p className="text-sm text-gray-600 mb-4">
              These actions are irreversible. Please be certain.
            </p>
            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all local data?')) {
                  localStorage.clear()
                  sessionStorage.clear()
                  clearAuth()
                  navigate('/')
                  toast.success('All local data cleared')
                }
              }}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              Clear All Local Data
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Icons
function EyeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  )
}

function EyeSlashIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
    </svg>
  )
}

function ClipboardCopyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
    </svg>
  )
}

function LogoutIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
    </svg>
  )
}
