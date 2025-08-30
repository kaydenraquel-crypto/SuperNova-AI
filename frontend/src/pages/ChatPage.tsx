import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  IconButton,
  Avatar,
  Chip,
  Button,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  Divider,
  Paper,
  Menu,
  MenuItem,
  Tooltip,
  CircularProgress,
  Fade,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Send,
  Mic,
  AttachFile,
  MoreVert,
  Add,
  SmartToy,
  Person,
  TrendingUp,
  Assessment,
  ShowChart,
  Settings,
  History,
  Bookmark,
  Share,
  Clear,
  VolumeUp,
  Code,
  Image,
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { useQuery, useMutation, useQueryClient } from 'react-query';

// Hooks
import { useAuth } from '../hooks/useAuth';
import { useChatWebSocket } from '../hooks/useWebSocket';
import { useThemeColors } from '../hooks/useTheme';

// Services
import { apiService } from '../services/api';

// Types
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    confidence?: number;
    suggestions?: string[];
    charts?: any[];
    attachments?: any[];
    voiceMessage?: boolean;
  };
}

interface ChatSession {
  id: string;
  name: string;
  createdAt: string;
  lastActivity: string;
  messageCount: number;
  preview: string;
}

const DRAWER_WIDTH = 300;

// Placeholder components
const MessageBubble: React.FC<{ message: ChatMessage; isOwn: boolean }> = ({ message, isOwn }) => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: isOwn ? 'flex-end' : 'flex-start',
      mb: 2,
    }}
  >
    <Paper
      sx={{
        p: 2,
        maxWidth: '70%',
        bgcolor: isOwn ? 'primary.main' : 'background.paper',
        color: isOwn ? 'primary.contrastText' : 'text.primary',
      }}
    >
      <Typography>{message.content}</Typography>
    </Paper>
  </Box>
);

const TypingIndicator: React.FC<{ message: string }> = ({ message }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
    <CircularProgress size={16} />
    <Typography variant="body2" color="text.secondary">
      {message}
    </Typography>
  </Box>
);

const ChatSuggestions: React.FC<{ suggestions: string[]; onSuggestionClick: (suggestion: string) => void }> = ({ suggestions, onSuggestionClick }) => (
  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
    {suggestions.map((suggestion, index) => (
      <Chip
        key={index}
        label={suggestion}
        variant="outlined"
        clickable
        onClick={() => onSuggestionClick(suggestion)}
      />
    ))}
  </Box>
);

const VoiceRecorder: React.FC<{ onRecordingComplete: (blob: Blob) => void; disabled: boolean }> = ({ disabled }) => (
  <Tooltip title="Voice message">
    <IconButton disabled={disabled}>
      <Mic />
    </IconButton>
  </Tooltip>
);

const ChatHeader: React.FC<{ session: any; connectionStatus: string; onToggleSidebar: () => void; onMenuOpen: (event: React.MouseEvent<HTMLElement>) => void }> = ({ session, connectionStatus, onToggleSidebar, onMenuOpen }) => {
  const theme = useTheme();
  return (
  <Box
    sx={{
      p: 2,
      borderBottom: `1px solid ${theme.palette.divider}`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    }}
  >
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      <Avatar sx={{ bgcolor: 'primary.main' }}>
        <SmartToy />
      </Avatar>
      <Box>
        <Typography variant="h6">AI Financial Advisor</Typography>
        <Typography variant="caption" color="text.secondary">
          {connectionStatus === 'connected' ? 'Online' : 'Connecting...'}
        </Typography>
      </Box>
    </Box>
    <IconButton onClick={onMenuOpen}>
      <MoreVert />
    </IconButton>
  </Box>
  );
};

const ChatPage: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const [isTyping, setIsTyping] = useState(false);
  const [voiceRecording, setVoiceRecording] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [attachmentMenuAnchor, setAttachmentMenuAnchor] = useState<null | HTMLElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  // WebSocket connection for real-time chat
  const {
    connectionStatus,
    lastChatMessage,
    sendChatMessage,
    sendTypingIndicator,
  } = useChatWebSocket(currentSessionId);

  // Fetch chat sessions
  const { data: sessions, isLoading: sessionsLoading } = useQuery<ChatSession[]>(
    'chatSessions',
    apiService.getChatSessions,
    {
      refetchOnWindowFocus: false,
    }
  );

  // Fetch messages for current session
  const { data: messages, isLoading: messagesLoading } = useQuery<ChatMessage[]>(
    ['chatMessages', currentSessionId],
    () => currentSessionId ? apiService.getChatMessages(currentSessionId) : Promise.resolve([]),
    {
      enabled: !!currentSessionId,
      refetchOnWindowFocus: false,
    }
  );

  // Create new session mutation
  const createSessionMutation = useMutation(
    apiService.createChatSession,
    {
      onSuccess: (newSession) => {
        setCurrentSessionId(newSession.id);
        queryClient.invalidateQueries('chatSessions');
      },
    }
  );

  // Send message mutation
  const sendMessageMutation = useMutation(
    ({ sessionId, message }: { sessionId: string; message: string }) =>
      apiService.sendChatMessage(sessionId, message),
    {
      onSuccess: (response) => {
        queryClient.setQueryData(
          ['chatMessages', currentSessionId],
          (old: ChatMessage[] = []) => [...old, response.userMessage, response.assistantMessage]
        );
        queryClient.invalidateQueries('chatSessions');
      },
    }
  );

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, lastChatMessage]);

  // Handle new WebSocket messages
  useEffect(() => {
    if (lastChatMessage && currentSessionId) {
      queryClient.setQueryData(
        ['chatMessages', currentSessionId],
        (old: ChatMessage[] = []) => [...old, lastChatMessage]
      );
    }
  }, [lastChatMessage, currentSessionId, queryClient]);

  // Initialize with first session or create new one
  useEffect(() => {
    if (sessions && sessions.length > 0 && !currentSessionId) {
      setCurrentSessionId(sessions[0].id);
    }
  }, [sessions, currentSessionId]);

  const handleSendMessage = useCallback(async () => {
    if (!message.trim() || !currentSessionId || sendMessageMutation.isLoading) return;

    const messageText = message.trim();
    setMessage('');
    setIsTyping(false);

    // Send via WebSocket for real-time response
    sendChatMessage(messageText);

    // Also send via HTTP for persistence
    try {
      await sendMessageMutation.mutateAsync({
        sessionId: currentSessionId,
        message: messageText,
      });
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  }, [message, currentSessionId, sendChatMessage, sendMessageMutation]);

  const handleKeyPress = useCallback((event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setMessage(event.target.value);
    
    // Send typing indicator
    if (!isTyping) {
      setIsTyping(true);
      sendTypingIndicator(true);
    }
    
    // Clear previous timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }
    
    // Stop typing after 3 seconds of inactivity
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      sendTypingIndicator(false);
    }, 3000);
  }, [isTyping, sendTypingIndicator]);

  const handleNewSession = useCallback(async () => {
    try {
      await createSessionMutation.mutateAsync();
      if (isMobile) {
        setSidebarOpen(false);
      }
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  }, [createSessionMutation, isMobile]);

  const handleSessionSelect = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId);
    if (isMobile) {
      setSidebarOpen(false);
    }
  }, [isMobile]);

  const handleVoiceRecord = useCallback((audioBlob: Blob) => {
    // Handle voice message
    console.log('Voice recording received:', audioBlob);
    // Implement voice-to-text and send as message
  }, []);

  const handleFileUpload = useCallback((files: FileList) => {
    // Handle file upload
    console.log('Files uploaded:', files);
    // Implement file processing and attachment
  }, []);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setMessage(suggestion);
    // Focus the input field after setting the suggestion
    setTimeout(() => {
      messageInputRef.current?.focus();
    }, 0);
  }, []);

  const currentSession = sessions?.find(s => s.id === currentSessionId);
  const allMessages = messages || [];

  const sidebarContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="h6" gutterBottom>
          Chat Sessions
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          fullWidth
          onClick={handleNewSession}
          disabled={createSessionMutation.isLoading}
        >
          New Chat
        </Button>
      </Box>

      {/* Sessions List */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List>
          {sessions?.map((session) => (
            <ListItem key={session.id} disablePadding>
              <ListItemButton
                selected={session.id === currentSessionId}
                onClick={() => handleSessionSelect(session.id)}
              >
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: theme.palette.primary.main }}>
                    <SmartToy />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={session.name || 'New Conversation'}
                  secondary={session.preview}
                  primaryTypographyProps={{
                    noWrap: true,
                    fontWeight: session.id === currentSessionId ? 600 : 400,
                  }}
                  secondaryTypographyProps={{
                    noWrap: true,
                    variant: 'caption',
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Box>
  );

  return (
    <>
      <Helmet>
        <title>AI Advisor Chat - SuperNova AI</title>
        <meta name="description" content="Chat with your AI financial advisor for personalized investment insights and market analysis." />
      </Helmet>

      <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
        {/* Sidebar */}
        <Drawer
          variant={isMobile ? 'temporary' : 'permanent'}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              position: 'relative',
              height: '100%',
            },
          }}
        >
          {sidebarContent}
        </Drawer>

        {/* Main Chat Area */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Chat Header */}
          <ChatHeader
            session={currentSession}
            connectionStatus={connectionStatus}
            onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
            onMenuOpen={(event) => setMenuAnchor(event.currentTarget)}
          />

          {/* Messages Area */}
          <Box
            sx={{
              flex: 1,
              overflow: 'auto',
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
            }}
          >
            {messagesLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : allMessages.length === 0 ? (
              <Box
                sx={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                }}
              >
                <Box>
                  <Avatar
                    sx={{
                      width: 80,
                      height: 80,
                      bgcolor: theme.palette.primary.main,
                      mx: 'auto',
                      mb: 2,
                    }}
                  >
                    <SmartToy sx={{ fontSize: 40 }} />
                  </Avatar>
                  <Typography variant="h5" gutterBottom>
                    Welcome to SuperNova AI
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                    I'm your AI financial advisor. Ask me about investments, market analysis, portfolio optimization, or trading strategies.
                  </Typography>
                  <ChatSuggestions
                    suggestions={[
                      "What's the current market sentiment?",
                      "Analyze my portfolio performance",
                      "Show me trending stocks today",
                      "Explain cryptocurrency investments",
                    ]}
                    onSuggestionClick={handleSuggestionClick}
                  />
                </Box>
              </Box>
            ) : (
              allMessages.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  isOwn={msg.role === 'user'}
                />
              ))
            )}

            {/* Typing Indicator */}
            {sendMessageMutation.isLoading && (
              <TypingIndicator message="AI is thinking..." />
            )}

            <div ref={messagesEndRef} />
          </Box>

          {/* Input Area */}
          <Paper
            elevation={3}
            sx={{
              p: 2,
              borderTop: `1px solid ${theme.palette.divider}`,
              bgcolor: theme.palette.background.paper,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1 }}>
              {/* Attachment Button */}
              <Tooltip title="Attach file">
                <IconButton
                  onClick={(event) => setAttachmentMenuAnchor(event.currentTarget)}
                  disabled={sendMessageMutation.isLoading}
                >
                  <AttachFile />
                </IconButton>
              </Tooltip>

              {/* Message Input */}
              <TextField
                inputRef={messageInputRef}
                fullWidth
                multiline
                maxRows={4}
                value={message}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about finance, markets, or your portfolio..."
                variant="outlined"
                disabled={sendMessageMutation.isLoading}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 3,
                  },
                }}
              />

              {/* Voice Button */}
              <VoiceRecorder
                onRecordingComplete={handleVoiceRecord}
                disabled={sendMessageMutation.isLoading}
              />

              {/* Send Button */}
              <Tooltip title="Send message">
                <IconButton
                  onClick={handleSendMessage}
                  disabled={!message.trim() || sendMessageMutation.isLoading}
                  sx={{
                    bgcolor: theme.palette.primary.main,
                    color: 'white',
                    '&:hover': {
                      bgcolor: theme.palette.primary.dark,
                    },
                    '&:disabled': {
                      bgcolor: theme.palette.action.disabledBackground,
                    },
                  }}
                >
                  {sendMessageMutation.isLoading ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : (
                    <Send />
                  )}
                </IconButton>
              </Tooltip>
            </Box>
          </Paper>
        </Box>

        {/* Chat Options Menu */}
        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <History sx={{ mr: 2 }} />
            Export Chat
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <Bookmark sx={{ mr: 2 }} />
            Save Conversation
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <Share sx={{ mr: 2 }} />
            Share
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <Clear sx={{ mr: 2 }} />
            Clear Chat
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <Settings sx={{ mr: 2 }} />
            Chat Settings
          </MenuItem>
        </Menu>

        {/* Attachment Menu */}
        <Menu
          anchorEl={attachmentMenuAnchor}
          open={Boolean(attachmentMenuAnchor)}
          onClose={() => setAttachmentMenuAnchor(null)}
        >
          <MenuItem onClick={() => setAttachmentMenuAnchor(null)}>
            <Image sx={{ mr: 2 }} />
            Upload Image
          </MenuItem>
          <MenuItem onClick={() => setAttachmentMenuAnchor(null)}>
            <Assessment sx={{ mr: 2 }} />
            Upload Document
          </MenuItem>
          <MenuItem onClick={() => setAttachmentMenuAnchor(null)}>
            <ShowChart sx={{ mr: 2 }} />
            Upload CSV Data
          </MenuItem>
          <MenuItem onClick={() => setAttachmentMenuAnchor(null)}>
            <Code sx={{ mr: 2 }} />
            Code Snippet
          </MenuItem>
        </Menu>
      </Box>
    </>
  );
};

export default ChatPage;