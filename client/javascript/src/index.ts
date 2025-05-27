/**
 * AI Principles Gym JavaScript/TypeScript Client
 * 
 * Main entry point for the library
 */

// Export main client
export { PrinciplesGymClient } from './client';
export { default } from './client';

// Export WebSocket support
export { 
  PrinciplesGymWebSocket, 
  createWebSocketClient,
  WebSocketProvider 
} from './websocket';

// Export all types
export * from './types';

// Version info
export const VERSION = '1.0.0';
