import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { Layout } from '@/components/Layout'

// Pages
import { HomePage } from '@/pages/HomePage'
import { DashboardPage } from '@/pages/DashboardPage'
import { AgentsPage } from '@/pages/AgentsPage'
import { TrainingPage } from '@/pages/TrainingPage'
import { TrainingSessionPage } from '@/pages/TrainingSessionPage'
import { ReportPage } from '@/pages/ReportPage'
import { SettingsPage } from '@/pages/SettingsPage'

// Stores
import { useAuthStore } from '@/stores/authStore'

function App() {
  const { apiKey } = useAuthStore()

  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<HomePage />} />
          
          {/* Protected routes */}
          <Route
            path="/*"
            element={
              apiKey ? (
                <Layout>
                  <Routes>
                    <Route path="/dashboard" element={<DashboardPage />} />
                    <Route path="/agents" element={<AgentsPage />} />
                    <Route path="/training" element={<TrainingPage />} />
                    <Route path="/training/:sessionId" element={<TrainingSessionPage />} />
                    <Route path="/reports/:sessionId" element={<ReportPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                  </Routes>
                </Layout>
              ) : (
                <Navigate to="/" replace />
              )
            }
          />
        </Routes>
      </Router>
    </ErrorBoundary>
  )
}

export default App
