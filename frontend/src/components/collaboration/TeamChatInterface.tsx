/**
 * SuperNova AI Team Chat Interface
 * Real-time team communication with WebSocket integration
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  IconButton,
  Avatar,
  AvatarGroup,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Button,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Drawer,
  Divider,
  Badge,
  Tooltip,
  Card,
  CardContent,
  InputAdornment,
  CircularProgress,
  Fab,
  Collapse,
  Alert
} from '@mui/material';

import {
  Send as SendIcon,
  AttachFile as AttachFileIcon,
  EmojiEmotions as EmojiIcon,
  MoreVert as MoreVertIcon,
  PersonAdd as PersonAddIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  NotificationsOff as NotificationsOffIcon,
  Reply as ReplyIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Person as PersonIcon,
  Group as GroupIcon,
  Close as CloseIcon,
  KeyboardArrowDown as ExpandIcon,
  KeyboardArrowUp as CollapseIcon,
  Circle as OnlineIcon,
  Schedule as AwayIcon
} from '@mui/icons-material';

import { useWebSocket } from '@/hooks/useWebSocket';
import { formatDistance } from 'date-fns';

// Types
interface ChatMessage {
  id: number;
  author_id: number;
  author_name: string;
  content: string;
  message_type: 'text' | 'file' | 'image' | 'system';
  thread_id?: number;
  mentions?: number[];
  reactions?: Record<string, number[]>;
  created_at: string;
  edited_at?: string;
  is_deleted: boolean;
}

interface ChatChannel {
  id: number;
  name: string;
  description?: string;
  is_private: boolean;
  channel_type: string;
  message_count: number;
  last_message_at?: string;
  unread_count?: number;
}

interface TeamMember {
  user_id: number;
  name: string;
  email: string;
  role: 'owner' | 'admin' | 'moderator' | 'member' | 'viewer';
  is_active: boolean;
  status: 'online' | 'away' | 'busy' | 'offline';
  last_activity?: string;
}

interface Team {
  id: number;
  name: string;
  description?: string;
  your_role: string;
}

// Team Chat Interface Component
interface TeamChatInterfaceProps {
  teamId: number;
  team: Team;
  onClose?: () => void;
}

const TeamChatInterface: React.FC<TeamChatInterfaceProps> = ({
  teamId,
  team,
  onClose
}) => {
  // State
  const [selectedChannel, setSelectedChannel] = useState<ChatChannel | null>(null);
  const [channels, setChannels] = useState<ChatChannel[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isTyping, setIsTyping] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [memberListOpen, setMemberListOpen] = useState(false);
  const [messageMenuAnchor, setMessageMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedMessage, setSelectedMessage] = useState<ChatMessage | null>(null);
  const [replyToMessage, setReplyToMessage] = useState<ChatMessage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket connection
  const { 
    isConnected, 
    sendMessage, 
    lastMessage,
    connectionError
  } = useWebSocket({
    url: '/ws/collaboration',
    enabled: true,
    reconnectAttempts: 5,
    reconnectInterval: 3000
  });

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Load team channels
  useEffect(() => {
    const loadChannels = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/collaboration/teams/${teamId}/channels`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });

        if (!response.ok) throw new Error('Failed to load channels');
        
        const channelsData = await response.json();
        setChannels(channelsData);
        
        // Select first channel by default
        if (channelsData.length > 0) {
          setSelectedChannel(channelsData[0]);
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadChannels();
  }, [teamId]);

  // Load messages when channel changes
  useEffect(() => {
    if (!selectedChannel) return;

    const loadMessages = async () => {
      try {
        const response = await fetch(`/api/collaboration/channels/${selectedChannel.id}/messages?limit=50`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });

        if (!response.ok) throw new Error('Failed to load messages');
        
        const messagesData = await response.json();
        setMessages(messagesData.messages || []);
      } catch (err: any) {
        setError(err.message);
      }
    };

    loadMessages();
  }, [selectedChannel]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage || !isConnected) return;

    try {
      const message = JSON.parse(lastMessage);
      
      switch (message.type) {
        case 'team_message':
          if (message.data.channel_id === selectedChannel?.id) {
            setMessages(prev => [...prev, message.data]);
          }
          break;
        
        case 'typing_indicator':
          if (message.data.channel_id === selectedChannel?.id) {
            setIsTyping(prev => {
              const filtered = prev.filter(name => name !== message.data.user_name);
              if (message.data.typing) {
                return [...filtered, message.data.user_name];
              }
              return filtered;
            });
          }
          break;
        
        case 'presence_update':
          setMembers(prev => prev.map(member => 
            member.user_id === message.data.user_id
              ? { ...member, status: message.data.status }
              : member
          ));
          break;

        case 'user_joined_team':
        case 'user_left_team':
          // Reload members list
          loadTeamMembers();
          break;
      }
    } catch (err) {
      console.error('Error parsing WebSocket message:', err);
    }
  }, [lastMessage, isConnected, selectedChannel]);

  // Join team collaboration on connect
  useEffect(() => {
    if (isConnected) {
      sendMessage({
        type: 'join_team',
        data: { team_id: teamId }
      });
    }
  }, [isConnected, teamId, sendMessage]);

  // Auto-scroll when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load team members
  const loadTeamMembers = useCallback(async () => {
    try {
      const response = await fetch(`/api/collaboration/teams/${teamId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });

      if (!response.ok) throw new Error('Failed to load team members');
      
      const teamData = await response.json();
      setMembers(teamData.members || []);
    } catch (err: any) {
      setError(err.message);
    }
  }, [teamId]);

  useEffect(() => {
    loadTeamMembers();
  }, [loadTeamMembers]);

  // Send message
  const handleSendMessage = async () => {
    if (!newMessage.trim() || !selectedChannel) return;

    try {
      const messageData = {
        type: 'team_message',
        data: {
          team_id: teamId,
          channel_id: selectedChannel.id,
          content: newMessage.trim(),
          message_type: 'text',
          thread_id: replyToMessage?.id || null,
          mentions: extractMentions(newMessage)
        }
      };

      sendMessage(messageData);
      setNewMessage('');
      setReplyToMessage(null);
      
      // Stop typing indicator
      sendTypingIndicator(false);
      
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Handle typing indicator
  const sendTypingIndicator = (typing: boolean) => {
    if (!selectedChannel) return;

    sendMessage({
      type: 'typing_indicator',
      data: {
        team_id: teamId,
        channel_id: selectedChannel.id,
        typing
      }
    });
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setNewMessage(value);

    // Send typing indicator
    if (value.length > 0) {
      sendTypingIndicator(true);
      
      // Clear previous timeout
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      
      // Stop typing after 3 seconds of inactivity
      typingTimeoutRef.current = setTimeout(() => {
        sendTypingIndicator(false);
      }, 3000);
    } else {
      sendTypingIndicator(false);
    }
  };

  // Extract mentions from message
  const extractMentions = (text: string): number[] => {
    const mentionRegex = /@(\w+)/g;
    const mentions: number[] = [];
    let match;

    while ((match = mentionRegex.exec(text)) !== null) {
      const username = match[1];
      const member = members.find(m => m.name.toLowerCase().includes(username.toLowerCase()));
      if (member) {
        mentions.push(member.user_id);
      }
    }

    return mentions;
  };

  // Message menu handlers
  const handleMessageMenuOpen = (event: React.MouseEvent<HTMLButtonElement>, message: ChatMessage) => {
    setMessageMenuAnchor(event.currentTarget);
    setSelectedMessage(message);
  };

  const handleMessageMenuClose = () => {
    setMessageMenuAnchor(null);
    setSelectedMessage(null);
  };

  const handleReplyToMessage = (message: ChatMessage) => {
    setReplyToMessage(message);
    messageInputRef.current?.focus();
    handleMessageMenuClose();
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <OnlineIcon sx={{ color: 'success.main', fontSize: 12 }} />;
      case 'away':
        return <AwayIcon sx={{ color: 'warning.main', fontSize: 12 }} />;
      case 'busy':
        return <OnlineIcon sx={{ color: 'error.main', fontSize: 12 }} />;
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ display: 'flex', height: '100vh', position: 'relative' }}>
      {/* Sidebar - Channels */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={sidebarOpen}
        sx={{
          width: 280,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 280,
            boxSizing: 'border-box',
            position: 'relative',
            height: '100%'
          }
        }}
      >
        <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'white' }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <GroupIcon />
            {team.name}
          </Typography>
          <Typography variant="caption" sx={{ opacity: 0.8 }}>
            {members.length} members
          </Typography>
        </Box>

        <Divider />

        {/* Channels List */}
        <List sx={{ flexGrow: 1, overflow: 'auto' }}>
          {channels.map((channel) => (
            <ListItem
              key={channel.id}
              button
              selected={selectedChannel?.id === channel.id}
              onClick={() => setSelectedChannel(channel)}
            >
              <ListItemText
                primary={`# ${channel.name}`}
                secondary={channel.description}
              />
              {channel.unread_count && channel.unread_count > 0 && (
                <ListItemSecondaryAction>
                  <Badge badgeContent={channel.unread_count} color="primary" />
                </ListItemSecondaryAction>
              )}
            </ListItem>
          ))}
        </List>

        <Divider />

        {/* Team Actions */}
        <Box sx={{ p: 2 }}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<PersonAddIcon />}
            size="small"
            sx={{ mb: 1 }}
          >
            Invite Members
          </Button>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<SettingsIcon />}
            size="small"
          >
            Team Settings
          </Button>
        </Box>
      </Drawer>

      {/* Main Chat Area */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Chat Header */}
        {selectedChannel && (
          <Paper sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'between' }}>
            <Box>
              <Typography variant="h6">
                # {selectedChannel.name}
              </Typography>
              {selectedChannel.description && (
                <Typography variant="body2" color="text.secondary">
                  {selectedChannel.description}
                </Typography>
              )}
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AvatarGroup max={5} sx={{ mr: 2 }}>
                {members.filter(m => m.status === 'online').map((member) => (
                  <Tooltip key={member.user_id} title={`${member.name} - ${member.status}`}>
                    <Avatar sx={{ width: 32, height: 32 }}>
                      {member.name.charAt(0).toUpperCase()}
                      <Badge
                        overlap="circular"
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                        badgeContent={getStatusIcon(member.status)}
                      />
                    </Avatar>
                  </Tooltip>
                ))}
              </AvatarGroup>
              <IconButton onClick={() => setMemberListOpen(true)}>
                <PersonIcon />
              </IconButton>
              {onClose && (
                <IconButton onClick={onClose}>
                  <CloseIcon />
                </IconButton>
              )}
            </Box>
          </Paper>
        )}

        {/* Connection Status */}
        {!isConnected && (
          <Alert severity="warning" sx={{ m: 1 }}>
            Disconnected from chat. Reconnecting...
          </Alert>
        )}

        {/* Messages List */}
        <Box sx={{ flexGrow: 1, overflow: 'auto', p: 1 }}>
          <List>
            {messages.map((message) => (
              <ChatMessageItem
                key={message.id}
                message={message}
                onMenuOpen={handleMessageMenuOpen}
                onReply={handleReplyToMessage}
              />
            ))}
            {/* Typing indicators */}
            {isTyping.length > 0 && (
              <ListItem>
                <ListItemText
                  secondary={
                    <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                      {isTyping.join(', ')} {isTyping.length === 1 ? 'is' : 'are'} typing...
                    </Typography>
                  }
                />
              </ListItem>
            )}
            <div ref={messagesEndRef} />
          </List>
        </Box>

        {/* Reply to message indicator */}
        <Collapse in={Boolean(replyToMessage)}>
          <Paper sx={{ p: 2, m: 1, bgcolor: 'grey.50' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'between' }}>
              <Typography variant="body2" color="text.secondary">
                Replying to {replyToMessage?.author_name}
              </Typography>
              <IconButton size="small" onClick={() => setReplyToMessage(null)}>
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
            <Typography variant="body2" sx={{ mt: 1, opacity: 0.7 }}>
              {replyToMessage?.content}
            </Typography>
          </Paper>
        </Collapse>

        {/* Message Input */}
        <Paper sx={{ p: 2, m: 1 }}>
          <TextField
            ref={messageInputRef}
            fullWidth
            multiline
            maxRows={4}
            placeholder={`Message #${selectedChannel?.name || 'channel'}`}
            value={newMessage}
            onChange={handleInputChange}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            disabled={!isConnected}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton disabled>
                    <AttachFileIcon />
                  </IconButton>
                  <IconButton disabled>
                    <EmojiIcon />
                  </IconButton>
                  <IconButton 
                    onClick={handleSendMessage}
                    disabled={!newMessage.trim() || !isConnected}
                    color="primary"
                  >
                    <SendIcon />
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        </Paper>
      </Box>

      {/* Message Context Menu */}
      <Menu
        anchorEl={messageMenuAnchor}
        open={Boolean(messageMenuAnchor)}
        onClose={handleMessageMenuClose}
      >
        <MenuItem onClick={() => handleReplyToMessage(selectedMessage!)}>
          <ReplyIcon sx={{ mr: 1 }} />
          Reply
        </MenuItem>
        <MenuItem onClick={handleMessageMenuClose}>
          <EditIcon sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={handleMessageMenuClose}>
          <DeleteIcon sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Members List Dialog */}
      <Dialog
        open={memberListOpen}
        onClose={() => setMemberListOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Team Members</DialogTitle>
        <DialogContent>
          <List>
            {members.map((member) => (
              <ListItem key={member.user_id}>
                <ListItemAvatar>
                  <Badge
                    overlap="circular"
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                    badgeContent={getStatusIcon(member.status)}
                  >
                    <Avatar>
                      {member.name.charAt(0).toUpperCase()}
                    </Avatar>
                  </Badge>
                </ListItemAvatar>
                <ListItemText
                  primary={member.name}
                  secondary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip size="small" label={member.role} color="primary" />
                      <Typography variant="caption" color="text.secondary">
                        {member.last_activity && formatDistance(new Date(member.last_activity), new Date(), { addSuffix: true })}
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMemberListOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Chat Message Item Component
interface ChatMessageItemProps {
  message: ChatMessage;
  onMenuOpen: (event: React.MouseEvent<HTMLButtonElement>, message: ChatMessage) => void;
  onReply: (message: ChatMessage) => void;
}

const ChatMessageItem: React.FC<ChatMessageItemProps> = ({
  message,
  onMenuOpen,
  onReply
}) => {
  const [showActions, setShowActions] = useState(false);

  return (
    <ListItem
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      sx={{ 
        alignItems: 'flex-start',
        '&:hover': { bgcolor: 'grey.50' }
      }}
    >
      <ListItemAvatar>
        <Avatar>
          {message.author_name.charAt(0).toUpperCase()}
        </Avatar>
      </ListItemAvatar>
      
      <ListItemText
        primary={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Typography variant="subtitle2" component="span">
              {message.author_name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatDistance(new Date(message.created_at), new Date(), { addSuffix: true })}
            </Typography>
            {message.edited_at && (
              <Chip size="small" label="edited" variant="outlined" />
            )}
          </Box>
        }
        secondary={
          <Typography variant="body2" component="div" sx={{ mt: 0.5 }}>
            {message.content}
            
            {/* Thread indicator */}
            {message.thread_id && (
              <Typography variant="caption" color="primary" sx={{ display: 'block', mt: 1 }}>
                Thread reply
              </Typography>
            )}
            
            {/* Reactions */}
            {message.reactions && Object.keys(message.reactions).length > 0 && (
              <Box sx={{ mt: 1, display: 'flex', gap: 0.5 }}>
                {Object.entries(message.reactions).map(([emoji, users]) => (
                  <Chip
                    key={emoji}
                    size="small"
                    label={`${emoji} ${users.length}`}
                    variant="outlined"
                    clickable
                  />
                ))}
              </Box>
            )}
          </Typography>
        }
      />

      {/* Message Actions */}
      {showActions && (
        <ListItemSecondaryAction>
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <IconButton size="small" onClick={() => onReply(message)}>
              <ReplyIcon fontSize="small" />
            </IconButton>
            <IconButton 
              size="small" 
              onClick={(e) => onMenuOpen(e, message)}
            >
              <MoreVertIcon fontSize="small" />
            </IconButton>
          </Box>
        </ListItemSecondaryAction>
      )}
    </ListItem>
  );
};

export default TeamChatInterface;