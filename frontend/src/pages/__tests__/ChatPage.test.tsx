/**
 * Tests for ChatPage component
 */
import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatPage } from '../ChatPage';
import { mockChatMessages, mockApiResponses } from '../../test-utils';
import { server } from '../../test-utils/mocks/server';
import { rest } from 'msw';

describe('ChatPage', () => {
  test('renders chat interface correctly', () => {
    render(<ChatPage />);

    expect(screen.getByText(/supernova ai assistant/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/ask me anything/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  test('displays welcome message on initial load', () => {
    render(<ChatPage />);

    expect(screen.getByText(/hello! i'm your supernova ai assistant/i)).toBeInTheDocument();
    expect(screen.getByText(/how can i help you today/i)).toBeInTheDocument();
  });

  test('sends message when form is submitted', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    const sendButton = screen.getByRole('button', { name: /send/i });

    await user.type(input, 'What do you think about AAPL?');
    await user.click(sendButton);

    // User message should appear
    expect(screen.getByText('What do you think about AAPL?')).toBeInTheDocument();

    // Wait for AI response
    await waitFor(() => {
      expect(screen.getByText(/based on the current market conditions/i)).toBeInTheDocument();
    });
  });

  test('sends message when Enter is pressed', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    await user.type(input, 'Tell me about market trends');
    await user.keyboard('{Enter}');

    expect(screen.getByText('Tell me about market trends')).toBeInTheDocument();
  });

  test('prevents sending empty messages', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const sendButton = screen.getByRole('button', { name: /send/i });
    
    // Button should be disabled initially
    expect(sendButton).toBeDisabled();

    const input = screen.getByPlaceholderText(/ask me anything/i);
    await user.type(input, '   '); // Only whitespace
    
    // Button should still be disabled
    expect(sendButton).toBeDisabled();

    await user.clear(input);
    await user.type(input, 'Real message');
    
    // Button should now be enabled
    expect(sendButton).not.toBeDisabled();
  });

  test('displays loading indicator while waiting for response', async () => {
    // Mock delayed response
    server.use(
      rest.post('/api/chat', (req, res, ctx) => {
        return res(
          ctx.delay(1000),
          ctx.json(mockApiResponses.getChatResponse)
        );
      })
    );

    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    const sendButton = screen.getByRole('button', { name: /send/i });

    await user.type(input, 'Test message');
    await user.click(sendButton);

    // Should show typing indicator
    expect(screen.getByText(/typing/i)).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('handles API errors gracefully', async () => {
    // Mock error response
    server.use(
      rest.post('/api/chat', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ error: 'Server error' }));
      })
    );

    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    const sendButton = screen.getByRole('button', { name: /send/i });

    await user.type(input, 'Test error');
    await user.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText(/sorry, something went wrong/i)).toBeInTheDocument();
    });
  });

  test('displays conversation history correctly', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    
    // Send first message
    await user.type(input, 'First message');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(screen.getByText('First message')).toBeInTheDocument();
    });

    // Clear input and send second message
    await user.clear(input);
    await user.type(input, 'Second message');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(screen.getByText('Second message')).toBeInTheDocument();
      expect(screen.getByText('First message')).toBeInTheDocument();
    });
  });

  test('shows suggested questions', () => {
    render(<ChatPage />);

    expect(screen.getByText(/suggested questions/i)).toBeInTheDocument();
    expect(screen.getByText(/what stocks should i buy/i)).toBeInTheDocument();
    expect(screen.getByText(/analyze my portfolio/i)).toBeInTheDocument();
  });

  test('clicking suggested question sends it as message', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const suggestion = screen.getByText(/what stocks should i buy/i);
    await user.click(suggestion);

    expect(screen.getByText('What stocks should I buy?')).toBeInTheDocument();
  });

  test('clears conversation when clear button is clicked', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    // Send a message first
    const input = screen.getByPlaceholderText(/ask me anything/i);
    await user.type(input, 'Test message');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(screen.getByText('Test message')).toBeInTheDocument();
    });

    // Clear conversation
    const clearButton = screen.getByRole('button', { name: /clear conversation/i });
    await user.click(clearButton);

    // Confirm clear
    const confirmButton = screen.getByRole('button', { name: /confirm/i });
    await user.click(confirmButton);

    // Message should be gone
    expect(screen.queryByText('Test message')).not.toBeInTheDocument();
  });

  test('auto-scrolls to bottom when new messages arrive', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    // Mock scrollIntoView
    const mockScrollIntoView = jest.fn();
    Element.prototype.scrollIntoView = mockScrollIntoView;

    const input = screen.getByPlaceholderText(/ask me anything/i);
    await user.type(input, 'Test message');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(mockScrollIntoView).toHaveBeenCalled();
    });
  });

  test('shows character count for long messages', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    const longMessage = 'A'.repeat(500);
    
    await user.type(input, longMessage);

    expect(screen.getByText(/500\/1000/)).toBeInTheDocument();
  });

  test('prevents sending messages over character limit', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    const tooLongMessage = 'A'.repeat(1001);
    
    await user.type(input, tooLongMessage);

    const sendButton = screen.getByRole('button', { name: /send/i });
    expect(sendButton).toBeDisabled();
    expect(screen.getByText(/message too long/i)).toBeInTheDocument();
  });

  test('supports voice input when available', async () => {
    // Mock speech recognition
    const mockSpeechRecognition = {
      start: jest.fn(),
      stop: jest.fn(),
      addEventListener: jest.fn(),
    };

    Object.defineProperty(window, 'SpeechRecognition', {
      value: jest.fn(() => mockSpeechRecognition),
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    const voiceButton = screen.getByRole('button', { name: /voice input/i });
    await user.click(voiceButton);

    expect(mockSpeechRecognition.start).toHaveBeenCalled();
  });

  test('formats code blocks in messages', async () => {
    server.use(
      rest.post('/api/chat', (req, res, ctx) => {
        return res(
          ctx.json({
            response: 'Here is some code:\n```python\nprint("Hello World")\n```',
            suggestions: [],
          })
        );
      })
    );

    const user = userEvent.setup();
    render(<ChatPage />);

    const input = screen.getByPlaceholderText(/ask me anything/i);
    await user.type(input, 'Show me some code');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      const codeBlock = screen.getByText('print("Hello World")');
      expect(codeBlock).toHaveClass('language-python');
    });
  });

  test('handles WebSocket connection for real-time updates', () => {
    // Mock WebSocket
    const mockWebSocket = {
      send: jest.fn(),
      close: jest.fn(),
      addEventListener: jest.fn(),
      readyState: WebSocket.OPEN,
    };

    global.WebSocket = jest.fn(() => mockWebSocket) as any;

    render(<ChatPage />);

    expect(WebSocket).toHaveBeenCalledWith(expect.stringContaining('ws://'));
  });

  test('shows connection status indicator', () => {
    render(<ChatPage />);

    const connectionIndicator = screen.getByTestId('connection-status');
    expect(connectionIndicator).toHaveAttribute('aria-label', 'Connected to server');
  });

  test('has proper accessibility attributes', () => {
    render(<ChatPage />);

    const chatContainer = screen.getByRole('log');
    expect(chatContainer).toHaveAttribute('aria-live', 'polite');
    expect(chatContainer).toHaveAttribute('aria-label', 'Chat conversation');

    const input = screen.getByPlaceholderText(/ask me anything/i);
    expect(input).toHaveAttribute('aria-label', 'Message input');
  });

  test('supports keyboard shortcuts', async () => {
    const user = userEvent.setup();
    render(<ChatPage />);

    // Test Ctrl+K to focus input
    await user.keyboard('{Control>}k{/Control}');
    
    const input = screen.getByPlaceholderText(/ask me anything/i);
    expect(input).toHaveFocus();

    // Test Escape to clear input
    await user.type(input, 'Some text');
    await user.keyboard('{Escape}');
    
    expect(input).toHaveValue('');
  });
});