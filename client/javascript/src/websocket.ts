/**
 * WebSocket support for real-time updates
 */

import { WebSocketMessage, TrainingStatus, Principle } from './types';

type WebSocketEventHandlers = {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onProgress?: (progress: number, completed: number, total: number) => void;
  onPrincipleDiscovered?: (principle: Principle) => void;
  onScenarioCompleted?: (scenarioName: string, index: number, total: number) => void;
  onStatusChange?: (status: TrainingStatus) => void;
  onMessage?: (message: WebSocketMessage) => void;
};

export class PrinciplesGymWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private handlers: WebSocketEventHandlers = {};
  private sessionId: string | null = null;
  private isClosing = false;

  constructor(
    private wsURL: string,
    private apiKey?: string
  ) {}

  /**
   * Connect to WebSocket server for a specific training session
   */
  connect(sessionId: string, handlers: WebSocketEventHandlers = {}): void {
    this.sessionId = sessionId;
    this.handlers = handlers;
    this.isClosing = false;
    this.establishConnection();
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.isClosing = true;
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.sessionId = null;
    this.reconnectAttempts = 0;
  }

  /**
   * Send a message through the WebSocket connection
   */
  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  private establishConnection(): void {
    if (!this.sessionId) return;

    // Construct WebSocket URL with authentication
    const url = new URL(`${this.wsURL}/ws/training/${this.sessionId}`);
    if (this.apiKey) {
      url.searchParams.set('api_key', this.apiKey);
    }

    try {
      this.ws = new WebSocket(url.toString());
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.handlers.onError?.(new Event('WebSocket creation failed'));
      this.scheduleReconnect();
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.handlers.onOpen?.();
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      this.handlers.onClose?.(event);
      
      if (!this.isClosing && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.handlers.onError?.(error);
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
  }

  private handleMessage(message: WebSocketMessage): void {
    // Call generic message handler
    this.handlers.onMessage?.(message);

    // Call specific handlers based on message type
    switch (message.type) {
      case 'training_progress':
        const { completed, total } = message.data;
        const progress = total > 0 ? (completed / total) * 100 : 0;
        this.handlers.onProgress?.(progress, completed, total);
        break;

      case 'principle_discovered':
        this.handlers.onPrincipleDiscovered?.(message.data as Principle);
        break;

      case 'scenario_completed':
        const { scenario_name, index, total: scenarioTotal } = message.data;
        this.handlers.onScenarioCompleted?.(scenario_name, index, scenarioTotal);
        break;

      case 'error':
        console.error('Training error:', message.data);
        break;

      default:
        console.log('Unknown message type:', message.type);
    }

    // Update status if provided
    if (message.data.status) {
      this.handlers.onStatusChange?.(message.data as TrainingStatus);
    }
  }

  private scheduleReconnect(): void {
    if (this.isClosing) return;

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      if (!this.isClosing && this.sessionId) {
        this.establishConnection();
      }
    }, delay);
  }
}

// Export for environments that don't have native WebSocket
export interface WebSocketProvider {
  new(url: string | URL, protocols?: string | string[]): WebSocket;
}

/**
 * Factory function to create WebSocket instance with custom provider
 */
export function createWebSocketClient(
  wsURL: string,
  apiKey?: string,
  WebSocketImpl?: WebSocketProvider
): PrinciplesGymWebSocket {
  // Use provided WebSocket implementation or global WebSocket
  if (WebSocketImpl && typeof globalThis !== 'undefined') {
    (globalThis as any).WebSocket = WebSocketImpl;
  }

  return new PrinciplesGymWebSocket(wsURL, apiKey);
}
