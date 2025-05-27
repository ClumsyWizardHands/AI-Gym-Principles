import { useEffect, useRef, useCallback, useState } from 'react'
import { io, Socket } from 'socket.io-client'
import toast from 'react-hot-toast'
import { WebSocketMessage, StatusUpdate, PrincipleDiscovered, ActionRecorded } from '@/api/types'

interface UseWebSocketOptions {
  autoConnect?: boolean
  reconnectAttempts?: number
  reconnectDelay?: number
}

interface WebSocketState {
  isConnected: boolean
  error: Error | null
}

export function useWebSocket(sessionId: string | null, options: UseWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 3,
    reconnectDelay = 1000
  } = options

  const socketRef = useRef<Socket | null>(null)
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    error: null
  })

  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [statusUpdates, setStatusUpdates] = useState<StatusUpdate[]>([])
  const [principles, setPrinciples] = useState<PrincipleDiscovered[]>([])
  const [actions, setActions] = useState<ActionRecorded[]>([])

  const connect = useCallback(() => {
    if (!sessionId || socketRef.current?.connected) return

    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
    const apiKey = localStorage.getItem('apiKey')

    socketRef.current = io(wsUrl, {
      auth: {
        apiKey,
        sessionId
      },
      reconnectionAttempts: reconnectAttempts,
      reconnectionDelay: reconnectDelay,
      transports: ['websocket']
    })

    const socket = socketRef.current

    socket.on('connect', () => {
      setState({ isConnected: true, error: null })
      toast.success('Connected to real-time updates')
    })

    socket.on('disconnect', () => {
      setState({ isConnected: false, error: null })
    })

    socket.on('connect_error', (error) => {
      setState({ isConnected: false, error })
      toast.error('Failed to connect to real-time updates')
    })

    socket.on('message', (message: WebSocketMessage) => {
      setLastMessage(message)

      switch (message.type) {
        case 'status_update':
          setStatusUpdates(prev => [...prev, message.data as StatusUpdate])
          break
        case 'principle_discovered':
          setPrinciples(prev => [...prev, message.data as PrincipleDiscovered])
          toast.success('New principle discovered!')
          break
        case 'action_recorded':
          setActions(prev => [...prev, message.data as ActionRecorded])
          break
        case 'error':
          toast.error(message.data.message || 'An error occurred')
          break
      }
    })

    // Join session room
    socket.emit('join_session', { sessionId })

  }, [sessionId, reconnectAttempts, reconnectDelay])

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
      setState({ isConnected: false, error: null })
    }
  }, [])

  const sendMessage = useCallback((type: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('message', { type, data })
    }
  }, [])

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
  }, [sessionId])

  return {
    isConnected: state.isConnected,
    error: state.error,
    lastMessage,
    statusUpdates,
    principles,
    actions,
    connect,
    disconnect,
    sendMessage
  }
}
