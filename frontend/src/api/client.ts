import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios'
import toast from 'react-hot-toast'

// Types
export interface ApiError {
  message: string
  statusCode?: number
  details?: any
}

export interface ApiResponse<T = any> {
  data: T
  status: number
  headers: Record<string, string>
}

// Create axios instance
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: '',  // Empty baseURL since endpoints already include /api
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  })

  // Request interceptor
  client.interceptors.request.use(
    (config) => {
      // Add API key if available
      const apiKey = localStorage.getItem('apiKey')
      if (apiKey && config.headers) {
        config.headers['X-API-Key'] = apiKey
      }

      // Add request ID
      const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      if (config.headers) {
        config.headers['X-Request-ID'] = requestId
      }

      return config
    },
    (error) => {
      return Promise.reject(error)
    }
  )

  // Response interceptor
  client.interceptors.response.use(
    (response) => {
      return response
    },
    async (error: AxiosError) => {
      const config = error.config as AxiosRequestConfig & { _retry?: boolean }
      
      // Handle network errors
      if (!error.response) {
        toast.error('Network error. Please check your connection.')
        return Promise.reject({
          message: 'Network error',
          statusCode: 0,
        })
      }

      // Handle specific status codes
      const { status, data } = error.response

      switch (status) {
        case 401:
          // Unauthorized - clear auth and redirect
          localStorage.removeItem('apiKey')
          window.location.href = '/'
          toast.error('Authentication failed. Please sign in again.')
          break
          
        case 403:
          toast.error('Access denied. You do not have permission to perform this action.')
          break
          
        case 404:
          // Don't show toast for 404s, let the component handle it
          break
          
        case 429:
          // Rate limit - show retry after if available
          const retryAfter = error.response.headers['retry-after']
          if (retryAfter) {
            toast.error(`Rate limit exceeded. Please try again in ${retryAfter} seconds.`)
          } else {
            toast.error('Rate limit exceeded. Please try again later.')
          }
          break
          
        case 500:
        case 502:
        case 503:
        case 504:
          // Server errors - retry once
          if (!config._retry) {
            config._retry = true
            return client(config)
          }
          toast.error('Server error. Please try again later.')
          break
          
        default:
          if (status >= 400 && status < 500) {
            // Client errors
            const message = (data as any)?.detail || (data as any)?.message || 'Request failed'
            toast.error(message)
          }
      }

      return Promise.reject({
        message: (data as any)?.detail || (data as any)?.message || error.message,
        statusCode: status,
        details: data,
      })
    }
  )

  return client
}

// Export singleton instance
export const apiClient = createApiClient()

// Helper functions
export const setApiKey = (apiKey: string) => {
  localStorage.setItem('apiKey', apiKey)
}

export const clearApiKey = () => {
  localStorage.removeItem('apiKey')
}

export const getApiKey = () => {
  return localStorage.getItem('apiKey')
}

// Generic request function with better error handling
export async function apiRequest<T>(
  config: AxiosRequestConfig
): Promise<T> {
  try {
    const response = await apiClient(config)
    return response.data
  } catch (error) {
    throw error as ApiError
  }
}
