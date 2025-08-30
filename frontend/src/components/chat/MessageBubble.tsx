import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Avatar,
  Chip,
  IconButton,
} from '@mui/material';
import {
  Person,
  SmartToy,
  ContentCopy,
  ThumbUp,
  ThumbDown,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import dayjs from 'dayjs';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: {
    model?: string;
    tokens?: number;
    confidence?: number;
  };
}

interface MessageBubbleProps {
  message: Message;
  onCopy?: (content: string) => void;
  onFeedback?: (messageId: string, feedback: 'up' | 'down') => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onCopy,
  onFeedback,
}) => {
  const theme = useTheme();
  const isUser = message.type === 'user';

  const handleCopy = () => {
    if (onCopy) {
      onCopy(message.content);
    } else {
      navigator.clipboard.writeText(message.content);
    }
  };

  const formatTime = (timestamp: string) => {
    return dayjs(timestamp).format('HH:mm');
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        alignItems: 'flex-start',
        gap: 1,
        mb: 2,
        maxWidth: '100%',
      }}
    >
      {/* Avatar */}
      <Avatar
        sx={{
          width: 32,
          height: 32,
          bgcolor: isUser ? theme.palette.primary.main : theme.palette.secondary.main,
          flexShrink: 0,
        }}
      >
        {isUser ? <Person /> : <SmartToy />}
      </Avatar>

      {/* Message Container */}
      <Box
        sx={{
          flex: 1,
          maxWidth: '70%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: isUser ? 'flex-end' : 'flex-start',
        }}
      >
        {/* Message Bubble */}
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: isUser 
              ? theme.palette.primary.main 
              : theme.palette.background.paper,
            color: isUser 
              ? theme.palette.primary.contrastText 
              : theme.palette.text.primary,
            borderRadius: 2,
            borderTopLeftRadius: isUser ? 2 : 0.5,
            borderTopRightRadius: isUser ? 0.5 : 2,
            maxWidth: '100%',
            wordBreak: 'break-word',
            position: 'relative',
            '&:hover .message-actions': {
              opacity: 1,
            },
          }}
        >
          <Typography 
            variant="body2" 
            sx={{ 
              whiteSpace: 'pre-wrap',
              lineHeight: 1.5,
            }}
          >
            {message.content}
          </Typography>

          {/* Message Actions */}
          <Box
            className="message-actions"
            sx={{
              position: 'absolute',
              top: -12,
              right: isUser ? 'auto' : -12,
              left: isUser ? -12 : 'auto',
              opacity: 0,
              transition: 'opacity 0.2s',
              display: 'flex',
              gap: 0.5,
              bgcolor: theme.palette.background.paper,
              borderRadius: 1,
              boxShadow: 1,
              p: 0.25,
            }}
          >
            <IconButton
              size="small"
              onClick={handleCopy}
              sx={{ 
                width: 24, 
                height: 24,
                color: theme.palette.text.secondary,
                '&:hover': {
                  color: theme.palette.primary.main,
                },
              }}
            >
              <ContentCopy sx={{ fontSize: 14 }} />
            </IconButton>
            
            {!isUser && onFeedback && (
              <>
                <IconButton
                  size="small"
                  onClick={() => onFeedback(message.id, 'up')}
                  sx={{ 
                    width: 24, 
                    height: 24,
                    color: theme.palette.text.secondary,
                    '&:hover': {
                      color: theme.palette.success.main,
                    },
                  }}
                >
                  <ThumbUp sx={{ fontSize: 14 }} />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => onFeedback(message.id, 'down')}
                  sx={{ 
                    width: 24, 
                    height: 24,
                    color: theme.palette.text.secondary,
                    '&:hover': {
                      color: theme.palette.error.main,
                    },
                  }}
                >
                  <ThumbDown sx={{ fontSize: 14 }} />
                </IconButton>
              </>
            )}
          </Box>
        </Paper>

        {/* Message Metadata */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            mt: 0.5,
            flexDirection: isUser ? 'row-reverse' : 'row',
          }}
        >
          <Typography variant="caption" color="text.secondary">
            {formatTime(message.timestamp)}
          </Typography>

          {message.metadata && (
            <>
              {message.metadata.model && (
                <Chip
                  label={message.metadata.model}
                  size="small"
                  variant="outlined"
                  sx={{ 
                    height: 20,
                    fontSize: '0.7rem',
                  }}
                />
              )}
              {message.metadata.confidence && (
                <Chip
                  label={`${Math.round(message.metadata.confidence * 100)}% confident`}
                  size="small"
                  variant="outlined"
                  color={message.metadata.confidence > 0.8 ? 'success' : 'warning'}
                  sx={{ 
                    height: 20,
                    fontSize: '0.7rem',
                  }}
                />
              )}
              {message.metadata.tokens && (
                <Typography variant="caption" color="text.secondary">
                  {message.metadata.tokens} tokens
                </Typography>
              )}
            </>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default MessageBubble;