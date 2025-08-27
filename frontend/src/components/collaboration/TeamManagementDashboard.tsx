/**
 * SuperNova AI Team Management Dashboard
 * Comprehensive team management interface with Material-UI
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  Avatar,
  AvatarGroup,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Badge,
  Skeleton,
  Alert,
  Snackbar,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress
} from '@mui/material';

import {
  Add as AddIcon,
  People as PeopleIcon,
  Settings as SettingsIcon,
  MoreVert as MoreVertIcon,
  Share as ShareIcon,
  Notifications as NotificationsIcon,
  TrendingUp as TrendingUpIcon,
  Chat as ChatIcon,
  PersonAdd as PersonAddIcon,
  ExitToApp as ExitToAppIcon,
  ContentCopy as ContentCopyIcon,
  ExpandMore as ExpandMoreIcon,
  AdminPanelSettings as AdminIcon,
  Visibility as VisibilityIcon,
  Edit as EditIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-hot-toast';

// Types
interface Team {
  id: number;
  name: string;
  description?: string;
  is_private: boolean;
  member_count: number;
  max_members: number;
  created_by: number;
  created_at: string;
  your_role: 'owner' | 'admin' | 'moderator' | 'member' | 'viewer';
  invite_code?: string;
}

interface TeamMember {
  user_id: number;
  name: string;
  email: string;
  role: 'owner' | 'admin' | 'moderator' | 'member' | 'viewer';
  joined_at: string;
  invited_by?: number;
  is_active: boolean;
  last_activity?: string;
}

interface TeamActivity {
  id: number;
  user_id: number;
  user_name: string;
  activity_type: string;
  title: string;
  description?: string;
  created_at: string;
}

interface TeamStatistics {
  member_count: number;
  active_members_7d: number;
  shared_portfolios: number;
  collaboration_score: number;
}

// Team Management Dashboard Component
const TeamManagementDashboard: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [createTeamOpen, setCreateTeamOpen] = useState(false);
  const [inviteMembersOpen, setInviteMembersOpen] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [teamMenuAnchor, setTeamMenuAnchor] = useState<null | HTMLElement>(null);
  const [notification, setNotification] = useState<{ message: string; severity: 'success' | 'error' | 'info' | 'warning' } | null>(null);
  
  const queryClient = useQueryClient();

  // Fetch user's teams
  const { 
    data: teamsData, 
    isLoading: teamsLoading, 
    error: teamsError 
  } = useQuery(['teams'], async () => {
    const response = await fetch('/api/collaboration/teams', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    if (!response.ok) throw new Error('Failed to fetch teams');
    return response.json();
  });

  // Create team mutation
  const createTeamMutation = useMutation(
    async (teamData: { name: string; description?: string; is_private: boolean; max_members: number; generate_invite_code: boolean }) => {
      const response = await fetch('/api/collaboration/teams', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify(teamData)
      });
      if (!response.ok) throw new Error('Failed to create team');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['teams']);
        setCreateTeamOpen(false);
        setNotification({ message: 'Team created successfully!', severity: 'success' });
      },
      onError: (error: any) => {
        setNotification({ message: error.message || 'Failed to create team', severity: 'error' });
      }
    }
  );

  // Join team by code mutation
  const joinTeamMutation = useMutation(
    async (inviteCode: string) => {
      const response = await fetch('/api/collaboration/teams/join', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({ invite_code: inviteCode })
      });
      if (!response.ok) throw new Error('Failed to join team');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['teams']);
        setNotification({ message: 'Successfully joined team!', severity: 'success' });
      },
      onError: (error: any) => {
        setNotification({ message: error.message || 'Failed to join team', severity: 'error' });
      }
    }
  );

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  const handleTeamMenuOpen = (event: React.MouseEvent<HTMLButtonElement>, team: Team) => {
    setTeamMenuAnchor(event.currentTarget);
    setSelectedTeam(team);
  };

  const handleTeamMenuClose = () => {
    setTeamMenuAnchor(null);
    setSelectedTeam(null);
  };

  const handleCopyInviteCode = (inviteCode: string) => {
    navigator.clipboard.writeText(inviteCode);
    setNotification({ message: 'Invite code copied to clipboard!', severity: 'success' });
  };

  const getRoleColor = (role: string): "primary" | "secondary" | "error" | "warning" | "info" | "success" => {
    switch (role) {
      case 'owner': return 'error';
      case 'admin': return 'warning';
      case 'moderator': return 'info';
      case 'member': return 'primary';
      case 'viewer': return 'secondary';
      default: return 'primary';
    }
  };

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'owner': 
      case 'admin': 
        return <AdminIcon fontSize="small" />;
      case 'moderator': 
        return <EditIcon fontSize="small" />;
      case 'viewer': 
        return <VisibilityIcon fontSize="small" />;
      default: 
        return <PeopleIcon fontSize="small" />;
    }
  };

  if (teamsError) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load teams. Please try again later.
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Team Collaboration
        </Typography>
        <Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateTeamOpen(true)}
            sx={{ mr: 2 }}
          >
            Create Team
          </Button>
          <Button
            variant="outlined"
            startIcon={<PersonAddIcon />}
            onClick={() => {
              const code = prompt('Enter team invite code:');
              if (code) joinTeamMutation.mutate(code);
            }}
          >
            Join Team
          </Button>
        </Box>
      </Box>

      {/* Navigation Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="My Teams" icon={<PeopleIcon />} />
          <Tab label="Activity Feed" icon={<TrendingUpIcon />} />
          <Tab label="Notifications" icon={<Badge badgeContent={3} color="error"><NotificationsIcon /></Badge>} />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <TeamsOverview 
          teams={teamsData?.teams || []} 
          loading={teamsLoading}
          onTeamMenuOpen={handleTeamMenuOpen}
          onCopyInviteCode={handleCopyInviteCode}
        />
      )}

      {selectedTab === 1 && (
        <ActivityFeed />
      )}

      {selectedTab === 2 && (
        <NotificationsPanel />
      )}

      {/* Create Team Dialog */}
      <CreateTeamDialog
        open={createTeamOpen}
        onClose={() => setCreateTeamOpen(false)}
        onSubmit={(data) => createTeamMutation.mutate(data)}
        loading={createTeamMutation.isLoading}
      />

      {/* Team Context Menu */}
      <Menu
        anchorEl={teamMenuAnchor}
        open={Boolean(teamMenuAnchor)}
        onClose={handleTeamMenuClose}
      >
        {selectedTeam?.your_role === 'owner' && [
          <MenuItem key="settings" onClick={() => console.log('Team settings')}>
            <SettingsIcon sx={{ mr: 1 }} />
            Team Settings
          </MenuItem>,
          <MenuItem key="invite" onClick={() => setInviteMembersOpen(true)}>
            <PersonAddIcon sx={{ mr: 1 }} />
            Invite Members
          </MenuItem>
        ]}
        {selectedTeam?.invite_code && (
          <MenuItem onClick={() => handleCopyInviteCode(selectedTeam.invite_code!)}>
            <ContentCopyIcon sx={{ mr: 1 }} />
            Copy Invite Code
          </MenuItem>
        )}
        <MenuItem onClick={() => console.log('View team')}>
          <ChatIcon sx={{ mr: 1 }} />
          Open Team Chat
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => console.log('Leave team')} sx={{ color: 'error.main' }}>
          <ExitToAppIcon sx={{ mr: 1 }} />
          Leave Team
        </MenuItem>
      </Menu>

      {/* Notification Snackbar */}
      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={6000}
        onClose={() => setNotification(null)}
      >
        {notification && (
          <Alert severity={notification.severity} onClose={() => setNotification(null)}>
            {notification.message}
          </Alert>
        )}
      </Snackbar>
    </Box>
  );
};

// Teams Overview Component
interface TeamsOverviewProps {
  teams: Team[];
  loading: boolean;
  onTeamMenuOpen: (event: React.MouseEvent<HTMLButtonElement>, team: Team) => void;
  onCopyInviteCode: (code: string) => void;
}

const TeamsOverview: React.FC<TeamsOverviewProps> = ({ 
  teams, 
  loading, 
  onTeamMenuOpen, 
  onCopyInviteCode 
}) => {
  if (loading) {
    return (
      <Grid container spacing={3}>
        {[1, 2, 3, 4].map((i) => (
          <Grid item xs={12} md={6} lg={4} key={i}>
            <Card>
              <CardContent>
                <Skeleton variant="text" width="60%" height={32} />
                <Skeleton variant="text" width="40%" height={20} sx={{ mb: 2 }} />
                <Skeleton variant="rectangular" height={80} sx={{ mb: 2 }} />
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Skeleton variant="circular" width={24} height={24} />
                  <Skeleton variant="circular" width={24} height={24} />
                  <Skeleton variant="circular" width={24} height={24} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  }

  if (teams.length === 0) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <PeopleIcon sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          No teams yet
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Create your first team to start collaborating with others
        </Typography>
        <Button variant="contained" startIcon={<AddIcon />}>
          Create Team
        </Button>
      </Paper>
    );
  }

  return (
    <Grid container spacing={3}>
      {teams.map((team) => (
        <Grid item xs={12} md={6} lg={4} key={team.id}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'flex-start', mb: 2 }}>
                <Box>
                  <Typography variant="h6" component="h3" gutterBottom>
                    {team.name}
                  </Typography>
                  <Chip
                    size="small"
                    label={team.your_role}
                    color={getRoleColor(team.your_role)}
                    icon={getRoleIcon(team.your_role)}
                    sx={{ mb: 1 }}
                  />
                </Box>
                <IconButton
                  size="small"
                  onClick={(e) => onTeamMenuOpen(e, team)}
                >
                  <MoreVertIcon />
                </IconButton>
              </Box>

              {team.description && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {team.description}
                </Typography>
              )}

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  {team.member_count} / {team.max_members} members
                </Typography>
                {!team.is_private && (
                  <Chip size="small" label="Public" color="info" variant="outlined" />
                )}
              </Box>

              <LinearProgress 
                variant="determinate" 
                value={(team.member_count / team.max_members) * 100}
                sx={{ mb: 2 }}
              />

              {/* Member avatars */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AvatarGroup max={4} sx={{ '& .MuiAvatar-root': { width: 24, height: 24, fontSize: '0.75rem' } }}>
                  {/* This would be populated with actual member data */}
                  <Avatar>JD</Avatar>
                  <Avatar>SM</Avatar>
                  <Avatar>AB</Avatar>
                </AvatarGroup>
                <Typography variant="caption" color="text.secondary">
                  Active members
                </Typography>
              </Box>
            </CardContent>

            <CardActions>
              <Button size="small" startIcon={<ChatIcon />}>
                Open Chat
              </Button>
              <Button size="small" startIcon={<ShareIcon />}>
                Share
              </Button>
            </CardActions>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

// Create Team Dialog Component
interface CreateTeamDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: any) => void;
  loading: boolean;
}

const CreateTeamDialog: React.FC<CreateTeamDialogProps> = ({
  open,
  onClose,
  onSubmit,
  loading
}) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    is_private: true,
    max_members: 50,
    generate_invite_code: false
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleInputChange = (field: string) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement> | SelectChangeEvent<number>
  ) => {
    const value = e.target.type === 'checkbox' ? (e.target as HTMLInputElement).checked : e.target.value;
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Create New Team</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Team Name"
            value={formData.name}
            onChange={handleInputChange('name')}
            required
            margin="normal"
            placeholder="Enter team name"
          />
          
          <TextField
            fullWidth
            label="Description"
            value={formData.description}
            onChange={handleInputChange('description')}
            multiline
            rows={3}
            margin="normal"
            placeholder="Describe your team's purpose"
          />

          <FormControl fullWidth margin="normal">
            <InputLabel>Maximum Members</InputLabel>
            <Select
              value={formData.max_members}
              onChange={handleInputChange('max_members')}
              label="Maximum Members"
            >
              <MenuItem value={10}>10 members</MenuItem>
              <MenuItem value={25}>25 members</MenuItem>
              <MenuItem value={50}>50 members</MenuItem>
              <MenuItem value={100}>100 members</MenuItem>
            </Select>
          </FormControl>

          <FormControlLabel
            control={
              <Switch
                checked={formData.is_private}
                onChange={handleInputChange('is_private')}
              />
            }
            label="Private team"
            sx={{ mt: 2, mb: 1 }}
          />
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Private teams require invitations to join
          </Typography>

          <FormControlLabel
            control={
              <Switch
                checked={formData.generate_invite_code}
                onChange={handleInputChange('generate_invite_code')}
              />
            }
            label="Generate invite code"
          />
          
          <Typography variant="body2" color="text.secondary">
            Allow members to join using a shareable code
          </Typography>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button 
            type="submit" 
            variant="contained"
            disabled={loading || !formData.name.trim()}
          >
            {loading ? 'Creating...' : 'Create Team'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

// Activity Feed Component
const ActivityFeed: React.FC = () => {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Recent Activity
      </Typography>
      {/* Activity feed implementation would go here */}
      <Typography variant="body2" color="text.secondary">
        Activity feed coming soon...
      </Typography>
    </Paper>
  );
};

// Notifications Panel Component
const NotificationsPanel: React.FC = () => {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Notifications
      </Typography>
      {/* Notifications implementation would go here */}
      <Typography variant="body2" color="text.secondary">
        Notifications panel coming soon...
      </Typography>
    </Paper>
  );
};

export default TeamManagementDashboard;