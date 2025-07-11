import { useEffect, useRef, useCallback, useState } from 'react'
import toast from 'react-hot-toast'

interface WebSocketMessage {
  type: string
  timestamp: string
  data: any
}

interface UseWebSocketOptions {
  autoConnect?: boolean
  reconnectAttempts?: number
  reconnectDelay?: number
}

interface WebSocketState {
  isConnected: boolean
  error: Error | null
  reconnectCount: number
}

export function useWebSocket(sessionId: string | null, options: UseWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 3,
    reconnectDelay = 1000
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number>()
  const reconnectCountRef = useRef(0)
  
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    error: null,
    reconnectCount: 0
  })

  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [statusUpdates, setStatusUpdates] = useState<any[]>([])
  const [principles, setPrinciples] = useState<any[]>([])
  const [actions, setActions] = useState<any[]>([])
  const [entropy, setEntropy] = useState<number>(0)

  const connect = useCallback(() => {
    if (!sessionId || wsRef.current?.readyState === WebSocket.OPEN) return

    const apiKey = localStorage.getItem('apiKey')
    if (!apiKey) {
      toast.error('API key not found. Please login again.')
      return
    }

    // Construct WebSocket URL with API key as query parameter
    // Use relative path to leverage Vite's proxy configuration
    const wsUrl = `/ws/training/${sessionId}?api_key=${apiKey}`

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setState({ isConnected: true, error: null, reconnectCount: reconnectCountRef.current })
        reconnectCountRef.current = 0
        console.log('WebSocket connected')
      }

      ws.onclose = (event) => {
        setState(prev => ({ ...prev, isConnected: false }))
        wsRef.current = null

        // Attempt reconnection if not a normal closure
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++
          const delay = reconnectDelay * Math.pow(2, reconnectCountRef.current - 1) // Exponential backoff
          
          console.log(`WebSocket closed. Reconnecting in ${delay}ms... (attempt ${reconnectCountRef.current}/${reconnectAttempts})`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        } else if (reconnectCountRef.current >= reconnectAttempts) {
          toast.error('Failed to connect to real-time updates after multiple attempts')
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        setState(prev => ({ ...prev, error: new Error('WebSocket connection error') }))
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLastMessage(message)

          // Handle different message types
          switch (message.type) {
            case 'connection_established':
              toast.success('Connected to real-time updates')
              break

            case 'progress':
              // Update progress in status updates
              setStatusUpdates(prev => [...prev, {
                type: 'progress',
                ...message.data
              }])
              break

            case 'principle_discovered':
              setPrinciples(prev => [...prev, message.data])
              toast.success(`New principle discovered: ${message.data.principle_name}`)
              break

            case 'action_recorded':
              setActions(prev => {
                const newActions = [...prev, message.data]
                // Keep only last 50 actions to prevent memory issues
                return newActions.slice(-50)
              })
              break

            case 'entropy_update':
              setEntropy(message.data.entropy)
              break

            case 'status_change':
              setStatusUpdates(prev => [...prev, {
                type: 'status',
                ...message.data
              }])
              if (message.data.new_status === 'completed') {
                toast.success('Training completed successfully!')
              } else if (message.data.new_status === 'failed') {
                toast.error('Training failed')
              }
              break

            case 'error':
              toast.error(message.data.error_message || 'An error occurred')
              if (!message.data.recoverable) {
                // Close connection on non-recoverable errors
                ws.close()
              }
              break

            case 'pong':
              // Response to ping, connection is healthy
              break

            default:
              console.log('Unknown message type:', message.type)
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

    } catch (error) {
      console.error('Failed to create WebSocket:', error)
      setState(prev => ({ ...prev, error: error as Error }))
    }
  }, [sessionId, reconnectAttempts, reconnectDelay])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected')
      wsRef.current = null
    }
    
    setState({ isConnected: false, error: null, reconnectCount: 0 })
    reconnectCountRef.current = 0
  }, [])

  const sendMessage = useCallback((type: string, data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, data }))
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])

  // Send periodic pings to keep connection alive
  useEffect(() => {
    if (!state.isConnected) return

    const pingInterval = setInterval(() => {
      sendMessage('ping', {})
    }, 30000) // Ping every 30 seconds

    return () => clearInterval(pingInterval)
  }, [state.isConnected, sendMessage])

  // Auto connect/disconnect based on sessionId
  useEffect(() => {
    if (autoConnect && sessionId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [sessionId, autoConnect, connect, disconnect])

  // Clear data when session changes
  useEffect(() => {
    setLastMessage(null)
    setStatusUpdates([])
    setPrinciples([])
    setActions([])
    setEntropy(0)
  }, [sessionId])

  return {
    isConnected: state.isConnected,
    error: state.error,
    reconnectCount: state.reconnectCount,
    lastMessage,
    statusUpdates,
    principles,
    actions,
    entropy,
    connect,
    disconnect,
    sendMessage
  }
}
