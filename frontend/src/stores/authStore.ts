import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { setApiKey, clearApiKey, getApiKey } from '@/api/client'

interface AuthState {
  apiKey: string | null
  isAuthenticated: boolean
  
  // Actions
  setApiKey: (apiKey: string) => void
  clearAuth: () => void
  checkAuth: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      apiKey: null,
      isAuthenticated: false,

      setApiKey: (apiKey: string) => {
        setApiKey(apiKey)
        set({ apiKey, isAuthenticated: true })
      },

      clearAuth: () => {
        clearApiKey()
        set({ apiKey: null, isAuthenticated: false })
      },

      checkAuth: () => {
        const apiKey = getApiKey()
        set({ 
          apiKey, 
          isAuthenticated: !!apiKey 
        })
      }
    }),
    {
      name: 'auth-storage',
      onRehydrateStorage: () => (state) => {
        // Sync with localStorage on rehydration
        if (state) {
          const apiKey = getApiKey()
          if (apiKey && !state.apiKey) {
            state.setApiKey(apiKey)
          } else if (!apiKey && state.apiKey) {
            state.clearAuth()
          }
        }
      }
    }
  )
)
