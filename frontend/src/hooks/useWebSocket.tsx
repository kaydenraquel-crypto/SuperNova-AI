import { useState, useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuth } from './useAuth';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export interface MarketDataUpdate {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
  high: number;
  low: number;
  open: number;
}

export interface NotificationMessage {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
}

export interface ChatMessage {
  id: string;
  sessionId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    confidence?: number;
    suggestions?: string[];
    charts?: any[];
  };
}

interface UseWebSocketOptions {
  enabled?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  timeout?: number;
}

interface WebSocketHookReturn {
  connectionStatus: ConnectionStatus;
  marketData: Record<string, MarketDataUpdate>;
  notifications: NotificationMessage[];
  lastMessage: any;
  sendMessage: (type: string, data: any) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
  connect: () => void;
  disconnect: () => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): WebSocketHookReturn => {
  const {
    enabled = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    timeout = 30000,
  } = options;

  const { tokens, isAuthenticated } = useAuth();
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [marketData, setMarketData] = useState<Record<string, MarketDataUpdate>>({});
  const [notifications, setNotifications] = useState<NotificationMessage[]>([]);
  const [lastMessage, setLastMessage] = useState<any>(null);

  const socketRef = useRef<Socket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const subscribedChannelsRef = useRef<Set<string>>(new Set());

  // Get WebSocket URL based on environment
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.NODE_ENV === 'development' 
      ? 'localhost:8081' 
      : window.location.host;
    return `${protocol}//${host}`;
  }, []);

  // Clear reconnect timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  // Attempt to reconnect
  const attemptReconnect = useCallback(() => {
    if (reconnectCountRef.current >= reconnectAttempts) {
      console.log('Max reconnection attempts reached');
      setConnectionStatus('error');
      return;
    }

    clearReconnectTimeout();
    reconnectCountRef.current += 1;
    
    console.log(`Attempting to reconnect (${reconnectCountRef.current}/${reconnectAttempts})`);
    setConnectionStatus('reconnecting');

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, reconnectInterval);
  }, [reconnectAttempts, reconnectInterval]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled || !isAuthenticated || socketRef.current?.connected) {
      return;
    }

    console.log('Connecting to WebSocket...');
    setConnectionStatus('connecting');

    try {
      const socket = io(getWebSocketUrl(), {
        auth: {
          token: tokens?.accessToken,
        },
        timeout,
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true,
        forceNew: true,
      });

      socketRef.current = socket;

      // Connection successful
      socket.on('connect', () => {
        console.log('WebSocket connected:', socket.id);
        setConnectionStatus('connected');
        reconnectCountRef.current = 0;
        clearReconnectTimeout();

        // Re-subscribe to channels after reconnection
        subscribedChannelsRef.current.forEach(channel => {
          socket.emit('subscribe', { channel });
        });
      });

      // Connection error
      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnectionStatus('error');
        
        if (reconnectCountRef.current < reconnectAttempts) {
          attemptReconnect();
        }
      });

      // Disconnection
      socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setConnectionStatus('disconnected');
        
        // Attempt reconnection unless it was a manual disconnect
        if (reason !== 'io client disconnect' && reconnectCountRef.current < reconnectAttempts) {
          attemptReconnect();
        }
      });

      // Market data updates
      socket.on('market_data', (data: MarketDataUpdate) => {
        setMarketData(prev => ({
          ...prev,
          [data.symbol]: data,
        }));
        setLastMessage({ type: 'market_data', data, timestamp: Date.now() });
      });

      // Notifications
      socket.on('notification', (notification: NotificationMessage) => {
        setNotifications(prev => [notification, ...prev.slice(0, 99)]); // Keep last 100
        setLastMessage({ type: 'notification', data: notification, timestamp: Date.now() });
      });

      // Chat messages
      socket.on('chat_message', (message: ChatMessage) => {
        setLastMessage({ type: 'chat_message', data: message, timestamp: Date.now() });
      });

      // System messages
      socket.on('system_message', (message: any) => {
        console.log('System message:', message);
        setLastMessage({ type: 'system_message', data: message, timestamp: Date.now() });
      });

      // Portfolio updates
      socket.on('portfolio_update', (update: any) => {
        setLastMessage({ type: 'portfolio_update', data: update, timestamp: Date.now() });
      });

      // Error handling
      socket.on('error', (error: any) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      });

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('error');
      
      if (reconnectCountRef.current < reconnectAttempts) {
        attemptReconnect();
      }
    }
  }, [enabled, isAuthenticated, tokens, timeout, getWebSocketUrl, attemptReconnect, reconnectAttempts]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    clearReconnectTimeout();
    reconnectCountRef.current = reconnectAttempts; // Prevent auto-reconnection
    
    if (socketRef.current) {
      console.log('Disconnecting WebSocket...');
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    setConnectionStatus('disconnected');
    subscribedChannelsRef.current.clear();
  }, [reconnectAttempts, clearReconnectTimeout]);

  // Send message
  const sendMessage = useCallback((type: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(type, data);
    } else {
      console.warn('WebSocket not connected. Cannot send message:', type, data);
    }
  }, []);

  // Subscribe to a channel
  const subscribe = useCallback((channel: string) => {
    subscribedChannelsRef.current.add(channel);
    
    if (socketRef.current?.connected) {
      socketRef.current.emit('subscribe', { channel });
      console.log(`Subscribed to channel: ${channel}`);
    }
  }, []);

  // Unsubscribe from a channel
  const unsubscribe = useCallback((channel: string) => {
    subscribedChannelsRef.current.delete(channel);
    
    if (socketRef.current?.connected) {
      socketRef.current.emit('unsubscribe', { channel });
      console.log(`Unsubscribed from channel: ${channel}`);
    }
  }, []);

  // Effect to manage connection lifecycle
  useEffect(() => {
    if (enabled && isAuthenticated) {
      connect();
    } else {
      disconnect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [enabled, isAuthenticated, connect, disconnect]);

  // Effect to handle authentication changes
  useEffect(() => {
    if (!isAuthenticated && socketRef.current) {
      disconnect();
    }
  }, [isAuthenticated, disconnect]);

  // Effect for cleanup on unmount
  useEffect(() => {
    return () => {
      clearReconnectTimeout();
      if (socketRef.current) {
        socketRef.current.removeAllListeners();
        socketRef.current.disconnect();
      }
    };
  }, [clearReconnectTimeout]);

  // Auto-subscribe to default channels
  useEffect(() => {
    if (connectionStatus === 'connected' && isAuthenticated) {
      // Subscribe to user-specific channels
      subscribe('notifications');
      subscribe('portfolio_updates');
      
      // Subscribe to general market data (could be made configurable)
      subscribe('market_data');
    }
  }, [connectionStatus, isAuthenticated, subscribe]);

  return {
    connectionStatus,
    marketData,
    notifications,
    lastMessage,
    sendMessage,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  };
};

// Helper hook for market data subscription
export const useMarketData = (symbols: string[] = []) => {
  const { connectionStatus, marketData, subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    if (connectionStatus === 'connected') {
      symbols.forEach(symbol => {
        subscribe(`market_data:${symbol.toUpperCase()}`);
      });
    }

    return () => {
      symbols.forEach(symbol => {
        unsubscribe(`market_data:${symbol.toUpperCase()}`);
      });
    };
  }, [symbols, connectionStatus, subscribe, unsubscribe]);

  return {
    connectionStatus,
    data: symbols.reduce((acc, symbol) => {
      acc[symbol] = marketData[symbol.toUpperCase()];
      return acc;
    }, {} as Record<string, MarketDataUpdate>),
  };
};

// Helper hook for chat WebSocket
export const useChatWebSocket = (sessionId?: string) => {
  const { connectionStatus, lastMessage, sendMessage, subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    if (connectionStatus === 'connected' && sessionId) {
      subscribe(`chat:${sessionId}`);
    }

    return () => {
      if (sessionId) {
        unsubscribe(`chat:${sessionId}`);
      }
    };
  }, [sessionId, connectionStatus, subscribe, unsubscribe]);

  const sendChatMessage = useCallback((message: string, metadata?: any) => {
    if (sessionId) {
      sendMessage('chat_message', {
        sessionId,
        message,
        metadata,
        timestamp: new Date().toISOString(),
      });
    }
  }, [sessionId, sendMessage]);

  const sendTypingIndicator = useCallback((isTyping: boolean) => {
    if (sessionId) {
      sendMessage('typing', {
        sessionId,
        isTyping,
      });
    }
  }, [sessionId, sendMessage]);

  return {
    connectionStatus,
    lastChatMessage: lastMessage?.type === 'chat_message' ? lastMessage.data : null,
    sendChatMessage,
    sendTypingIndicator,
  };
};

export default useWebSocket;