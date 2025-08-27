/**
 * Tests for useWebSocket custom hook
 */
import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { useWebSocket } from '../useWebSocket';

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: WebSocket.CONNECTING,
};

global.WebSocket = jest.fn(() => mockWebSocket) as any;

describe('useWebSocket', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWebSocket.readyState = WebSocket.CONNECTING;
  });

  test('initializes WebSocket connection', () => {
    renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    expect(WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws');
  });

  test('returns initial state correctly', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    expect(result.current.isConnected).toBe(false);
    expect(result.current.lastMessage).toBe(null);
    expect(typeof result.current.sendMessage).toBe('function');
  });

  test('updates connection status when WebSocket opens', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    // Simulate WebSocket opening
    const openHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'open'
    )?.[1];

    act(() => {
      mockWebSocket.readyState = WebSocket.OPEN;
      openHandler?.();
    });

    expect(result.current.isConnected).toBe(true);
  });

  test('updates connection status when WebSocket closes', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    // First open the connection
    const openHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'open'
    )?.[1];

    act(() => {
      mockWebSocket.readyState = WebSocket.OPEN;
      openHandler?.();
    });

    expect(result.current.isConnected).toBe(true);

    // Then close it
    const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'close'
    )?.[1];

    act(() => {
      mockWebSocket.readyState = WebSocket.CLOSED;
      closeHandler?.();
    });

    expect(result.current.isConnected).toBe(false);
  });

  test('receives and stores messages', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];

    const testMessage = { type: 'test', data: 'hello' };

    act(() => {
      messageHandler?.({ 
        data: JSON.stringify(testMessage) 
      } as MessageEvent);
    });

    expect(result.current.lastMessage).toEqual(testMessage);
  });

  test('sends messages when connected', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    // Open connection
    const openHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'open'
    )?.[1];

    act(() => {
      mockWebSocket.readyState = WebSocket.OPEN;
      openHandler?.();
    });

    const testMessage = { type: 'test', content: 'hello' };

    act(() => {
      result.current.sendMessage(testMessage);
    });

    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify(testMessage));
  });

  test('does not send messages when disconnected', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    const testMessage = { type: 'test', content: 'hello' };

    act(() => {
      result.current.sendMessage(testMessage);
    });

    expect(mockWebSocket.send).not.toHaveBeenCalled();
  });

  test('handles connection errors', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    const errorHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'error'
    )?.[1];

    act(() => {
      errorHandler?.({ error: 'Connection failed' } as any);
    });

    expect(result.current.isConnected).toBe(false);
    expect(result.current.connectionError).toBeTruthy();
  });

  test('attempts to reconnect on connection loss', () => {
    jest.useFakeTimers();
    
    const { result } = renderHook(() => 
      useWebSocket('ws://localhost:8000/ws', { 
        reconnectAttempts: 3,
        reconnectInterval: 1000 
      })
    );

    // Simulate connection close
    const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'close'
    )?.[1];

    act(() => {
      closeHandler?.();
    });

    // Fast-forward time
    act(() => {
      jest.advanceTimersByTime(1000);
    });

    // Should attempt to reconnect
    expect(WebSocket).toHaveBeenCalledTimes(2);

    jest.useRealTimers();
  });

  test('stops reconnecting after max attempts', () => {
    jest.useFakeTimers();
    
    renderHook(() => 
      useWebSocket('ws://localhost:8000/ws', { 
        reconnectAttempts: 2,
        reconnectInterval: 1000 
      })
    );

    // Simulate multiple connection failures
    const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'close'
    )?.[1];

    for (let i = 0; i < 3; i++) {
      act(() => {
        closeHandler?.();
      });

      act(() => {
        jest.advanceTimersByTime(1000);
      });
    }

    // Should stop trying after max attempts (initial + 2 reconnects = 3 total)
    expect(WebSocket).toHaveBeenCalledTimes(3);

    jest.useRealTimers();
  });

  test('cleans up WebSocket on unmount', () => {
    const { unmount } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    unmount();

    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  test('handles invalid JSON messages gracefully', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];

    act(() => {
      messageHandler?.({ 
        data: 'invalid json' 
      } as MessageEvent);
    });

    // Should not crash and lastMessage should remain null
    expect(result.current.lastMessage).toBe(null);
  });

  test('supports custom options', () => {
    const options = {
      reconnectAttempts: 5,
      reconnectInterval: 2000,
      protocols: ['echo-protocol'],
    };

    renderHook(() => useWebSocket('ws://localhost:8000/ws', options));

    expect(WebSocket).toHaveBeenCalledWith(
      'ws://localhost:8000/ws',
      ['echo-protocol']
    );
  });

  test('provides connection state history', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'));

    // Initially connecting
    expect(result.current.connectionState).toBe('connecting');

    // Open connection
    const openHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'open'
    )?.[1];

    act(() => {
      mockWebSocket.readyState = WebSocket.OPEN;
      openHandler?.();
    });

    expect(result.current.connectionState).toBe('connected');

    // Close connection
    const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'close'
    )?.[1];

    act(() => {
      mockWebSocket.readyState = WebSocket.CLOSED;
      closeHandler?.();
    });

    expect(result.current.connectionState).toBe('disconnected');
  });

  test('tracks message history when enabled', () => {
    const { result } = renderHook(() => 
      useWebSocket('ws://localhost:8000/ws', { keepHistory: true })
    );

    const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];

    const messages = [
      { type: 'test', data: 'message1' },
      { type: 'test', data: 'message2' },
      { type: 'test', data: 'message3' },
    ];

    messages.forEach(message => {
      act(() => {
        messageHandler?.({ 
          data: JSON.stringify(message) 
        } as MessageEvent);
      });
    });

    expect(result.current.messageHistory).toHaveLength(3);
    expect(result.current.messageHistory[2]).toEqual(messages[2]);
  });

  test('limits message history size', () => {
    const { result } = renderHook(() => 
      useWebSocket('ws://localhost:8000/ws', { 
        keepHistory: true,
        maxHistorySize: 2 
      })
    );

    const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];

    // Send 3 messages but expect only the last 2 to be kept
    for (let i = 1; i <= 3; i++) {
      act(() => {
        messageHandler?.({ 
          data: JSON.stringify({ type: 'test', data: `message${i}` })
        } as MessageEvent);
      });
    }

    expect(result.current.messageHistory).toHaveLength(2);
    expect(result.current.messageHistory[0].data).toBe('message2');
    expect(result.current.messageHistory[1].data).toBe('message3');
  });
});