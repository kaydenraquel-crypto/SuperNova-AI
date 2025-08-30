import React, { useState } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Avatar,
  Chip,
  Menu,
  MenuItem,
  Divider,
  Badge,
} from '@mui/material';
import {
  SmartToy,
  MoreVert,
  Settings,
  History,
  Delete,
  Download,
  Share,
  Info,
  Circle,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface ChatHeaderProps {
  isOnline: boolean;
  messageCount: number;
  sessionDuration: number;
  onClearChat?: () => void;
  onExportChat?: () => void;
  onShareChat?: () => void;
  onViewHistory?: () => void;
  onSettings?: () => void;
}

const ChatHeader: React.FC<ChatHeaderProps> = ({
  isOnline,
  messageCount,
  sessionDuration,
  onClearChat,
  onExportChat,
  onShareChat,
  onViewHistory,
  onSettings,
}) => {
  const theme = useTheme();
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getStatusColor = () => {
    return isOnline ? theme.palette.success.main : theme.palette.error.main;
  };

  const getStatusText = () => {
    return isOnline ? 'Online' : 'Offline';
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        p: 2,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: theme.palette.background.paper,
      }}
    >
      {/* Left side - AI Avatar and Info */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Badge
          overlap="circular"
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          badgeContent={
            <Circle
              sx={{
                width: 12,
                height: 12,
                color: getStatusColor(),
                border: 2,
                borderColor: 'background.paper',
                borderRadius: '50%',
              }}
            />
          }
        >
          <Avatar
            sx={{
              bgcolor: theme.palette.secondary.main,
              width: 48,
              height: 48,
            }}
          >
            <SmartToy />
          </Avatar>
        </Badge>

        <Box>
          <Typography variant="h6" component="h2">
            SuperNova AI Assistant
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {getStatusText()}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              •
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Ready to help with financial analysis
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Right side - Session Info and Menu */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        {/* Session Stats */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={`${messageCount} messages`}
            size="small"
            variant="outlined"
            icon={<History />}
          />
          <Chip
            label={formatDuration(sessionDuration)}
            size="small"
            variant="outlined"
          />
        </Box>

        {/* Menu */}
        <IconButton
          onClick={(e) => setMenuAnchor(e.currentTarget)}
          sx={{ ml: 1 }}
        >
          <MoreVert />
        </IconButton>

        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
        >
          <MenuItem onClick={() => { setMenuAnchor(null); onViewHistory?.(); }}>
            <History sx={{ mr: 2 }} />
            View Chat History
          </MenuItem>
          
          <Divider />
          
          <MenuItem onClick={() => { setMenuAnchor(null); onExportChat?.(); }}>
            <Download sx={{ mr: 2 }} />
            Export Chat
          </MenuItem>
          
          <MenuItem onClick={() => { setMenuAnchor(null); onShareChat?.(); }}>
            <Share sx={{ mr: 2 }} />
            Share Chat
          </MenuItem>
          
          <Divider />
          
          <MenuItem 
            onClick={() => { setMenuAnchor(null); onClearChat?.(); }}
            sx={{ color: theme.palette.error.main }}
          >
            <Delete sx={{ mr: 2 }} />
            Clear Chat
          </MenuItem>
          
          <Divider />
          
          <MenuItem onClick={() => { setMenuAnchor(null); onSettings?.(); }}>
            <Settings sx={{ mr: 2 }} />
            Chat Settings
          </MenuItem>
          
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <Info sx={{ mr: 2 }} />
            About Assistant
          </MenuItem>
        </Menu>
      </Box>
    </Box>
  );
};

export default ChatHeader;