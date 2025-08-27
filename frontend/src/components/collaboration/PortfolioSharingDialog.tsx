/**
 * SuperNova AI Portfolio Sharing Dialog
 * Advanced portfolio sharing with team/user/public options
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Autocomplete,
  Chip,
  Switch,
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
  Avatar,
  IconButton,
  Divider,
  Alert,
  CircularProgress,
  Tooltip,
  InputAdornment,
  Grid,
  Card,
  CardContent
} from '@mui/material';

import {
  Share as ShareIcon,
  Public as PublicIcon,
  Group as GroupIcon,
  Person as PersonIcon,
  Lock as LockIcon,
  Link as LinkIcon,
  ContentCopy as CopyIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  AdminPanelSettings as AdminIcon,
  Comment as CommentIcon,
  Schedule as ScheduleIcon,
  Security as SecurityIcon,
  Close as CloseIcon,
  Check as CheckIcon
} from '@mui/icons-material';

import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-hot-toast';

// Types
interface Portfolio {
  id: number;
  name: string;
  description?: string;
  user_id: number;
  created_at: string;
}

interface Team {
  id: number;
  name: string;
  member_count: number;
  your_role: string;
}

interface User {
  id: number;
  name: string;
  email: string;
}

interface ShareTarget {
  type: 'team' | 'user' | 'public';
  id?: number;
  name?: string;
}

interface SharePermission {
  level: 'view' | 'comment' | 'edit' | 'admin';
  label: string;
  description: string;
  icon: React.ReactNode;
}

interface ShareSettings {
  target: ShareTarget;
  permission: string;
  expiresAt?: Date;
  passwordProtected: boolean;
  password?: string;
  title?: string;
  description?: string;
}

// Permission levels
const PERMISSION_LEVELS: SharePermission[] = [
  {
    level: 'view',
    label: 'View Only',
    description: 'Can view portfolio data and performance',
    icon: <ViewIcon />
  },
  {
    level: 'comment',
    label: 'Comment',
    description: 'Can view and add comments',
    icon: <CommentIcon />
  },
  {
    level: 'edit',
    label: 'Edit',
    description: 'Can modify portfolio holdings and settings',
    icon: <EditIcon />
  },
  {
    level: 'admin',
    label: 'Admin',
    description: 'Can manage sharing and permissions',
    icon: <AdminIcon />
  }
];

// Portfolio Sharing Dialog Component
interface PortfolioSharingDialogProps {
  open: boolean;
  onClose: () => void;
  portfolio: Portfolio;
}

const PortfolioSharingDialog: React.FC<PortfolioSharingDialogProps> = ({
  open,
  onClose,
  portfolio
}) => {
  // State
  const [shareSettings, setShareSettings] = useState<ShareSettings>({
    target: { type: 'team' },
    permission: 'view',
    passwordProtected: false
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'share' | 'existing' | 'public'>('share');
  const [publicLinkCreated, setPublicLinkCreated] = useState(false);
  const [publicLink, setPublicLink] = useState<string>('');

  const queryClient = useQueryClient();

  // Fetch teams for sharing options
  const { data: teamsData, isLoading: teamsLoading } = useQuery(
    ['teams'],
    async () => {
      const response = await fetch('/api/collaboration/teams', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });
      if (!response.ok) throw new Error('Failed to fetch teams');
      return response.json();
    },
    { enabled: open }
  );

  // Fetch users for sharing (simplified - in real app would be filtered)
  const { data: usersData, isLoading: usersLoading } = useQuery(
    ['users', searchQuery],
    async () => {
      if (!searchQuery || searchQuery.length < 2) return { users: [] };
      
      const response = await fetch(`/api/users/search?q=${searchQuery}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });
      if (!response.ok) throw new Error('Failed to search users');
      return response.json();
    },
    { enabled: open && shareSettings.target.type === 'user' }
  );

  // Fetch existing shares
  const { data: existingShares, isLoading: sharesLoading } = useQuery(
    ['portfolio-shares', portfolio.id],
    async () => {
      const response = await fetch(`/api/collaboration/portfolios/${portfolio.id}/shares`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });
      if (!response.ok) throw new Error('Failed to fetch existing shares');
      return response.json();
    },
    { enabled: open }
  );

  // Share portfolio mutation
  const sharePortfolioMutation = useMutation(
    async (settings: ShareSettings) => {
      const response = await fetch('/api/collaboration/share/portfolio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          portfolio_id: portfolio.id,
          target_type: settings.target.type,
          target_id: settings.target.id,
          permission_level: settings.permission,
          expires_at: settings.expiresAt?.toISOString(),
          password_protected: settings.passwordProtected,
          password: settings.password,
          title: settings.title,
          description: settings.description
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to share portfolio');
      }
      
      return response.json();
    },
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries(['portfolio-shares', portfolio.id]);
        
        if (shareSettings.target.type === 'public') {
          setPublicLink(data.share_url || '');
          setPublicLinkCreated(true);
          setActiveTab('public');
        } else {
          toast.success('Portfolio shared successfully!');
        }
      },
      onError: (error: any) => {
        toast.error(error.message);
      }
    }
  );

  const handleShare = () => {
    if (shareSettings.target.type !== 'public' && !shareSettings.target.id) {
      toast.error('Please select a target to share with');
      return;
    }

    if (shareSettings.passwordProtected && !shareSettings.password) {
      toast.error('Please enter a password for protected sharing');
      return;
    }

    sharePortfolioMutation.mutate(shareSettings);
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(publicLink);
    toast.success('Link copied to clipboard!');
  };

  const handleTargetChange = (target: ShareTarget) => {
    setShareSettings(prev => ({
      ...prev,
      target
    }));
  };

  const resetForm = () => {
    setShareSettings({
      target: { type: 'team' },
      permission: 'view',
      passwordProtected: false
    });
    setSearchQuery('');
    setPublicLinkCreated(false);
    setPublicLink('');
    setActiveTab('share');
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { minHeight: 600 }
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ShareIcon />
            Share Portfolio: {portfolio.name}
          </Box>
          <IconButton onClick={handleClose}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>

        <DialogContent>
          {/* Tab Navigation */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant={activeTab === 'share' ? 'contained' : 'text'}
                onClick={() => setActiveTab('share')}
                startIcon={<ShareIcon />}
              >
                New Share
              </Button>
              <Button
                variant={activeTab === 'existing' ? 'contained' : 'text'}
                onClick={() => setActiveTab('existing')}
                startIcon={<GroupIcon />}
              >
                Existing Shares
              </Button>
              <Button
                variant={activeTab === 'public' ? 'contained' : 'text'}
                onClick={() => setActiveTab('public')}
                startIcon={<PublicIcon />}
              >
                Public Link
              </Button>
            </Box>
          </Box>

          {/* New Share Tab */}
          {activeTab === 'share' && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Share With
                    </Typography>

                    <FormControl component="fieldset" fullWidth>
                      <RadioGroup
                        value={shareSettings.target.type}
                        onChange={(e) => handleTargetChange({ type: e.target.value as any })}
                      >
                        <FormControlLabel
                          value="team"
                          control={<Radio />}
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <GroupIcon />
                              <Box>
                                <Typography variant="body2">Team</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  Share with entire team
                                </Typography>
                              </Box>
                            </Box>
                          }
                        />
                        
                        <FormControlLabel
                          value="user"
                          control={<Radio />}
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <PersonIcon />
                              <Box>
                                <Typography variant="body2">Individual User</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  Share with specific person
                                </Typography>
                              </Box>
                            </Box>
                          }
                        />
                        
                        <FormControlLabel
                          value="public"
                          control={<Radio />}
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <PublicIcon />
                              <Box>
                                <Typography variant="body2">Public Link</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  Anyone with the link can access
                                </Typography>
                              </Box>
                            </Box>
                          }
                        />
                      </RadioGroup>
                    </FormControl>

                    {/* Target Selection */}
                    {shareSettings.target.type === 'team' && (
                      <Box sx={{ mt: 2 }}>
                        <Autocomplete
                          options={teamsData?.teams || []}
                          getOptionLabel={(option: Team) => option.name}
                          loading={teamsLoading}
                          onChange={(_, value) => {
                            if (value) {
                              handleTargetChange({
                                type: 'team',
                                id: value.id,
                                name: value.name
                              });
                            }
                          }}
                          renderInput={(params) => (
                            <TextField
                              {...params}
                              label="Select Team"
                              placeholder="Choose a team"
                              InputProps={{
                                ...params.InputProps,
                                endAdornment: (
                                  <>
                                    {teamsLoading && <CircularProgress size={20} />}
                                    {params.InputProps.endAdornment}
                                  </>
                                )
                              }}
                            />
                          )}
                          renderOption={(props, option) => (
                            <Box component="li" {...props}>
                              <Avatar sx={{ mr: 2, width: 32, height: 32 }}>
                                <GroupIcon />
                              </Avatar>
                              <Box>
                                <Typography variant="body2">{option.name}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {option.member_count} members · {option.your_role}
                                </Typography>
                              </Box>
                            </Box>
                          )}
                        />
                      </Box>
                    )}

                    {shareSettings.target.type === 'user' && (
                      <Box sx={{ mt: 2 }}>
                        <Autocomplete
                          options={usersData?.users || []}
                          getOptionLabel={(option: User) => `${option.name} (${option.email})`}
                          loading={usersLoading}
                          onInputChange={(_, value) => setSearchQuery(value)}
                          onChange={(_, value) => {
                            if (value) {
                              handleTargetChange({
                                type: 'user',
                                id: value.id,
                                name: value.name
                              });
                            }
                          }}
                          renderInput={(params) => (
                            <TextField
                              {...params}
                              label="Search Users"
                              placeholder="Type name or email"
                              InputProps={{
                                ...params.InputProps,
                                endAdornment: (
                                  <>
                                    {usersLoading && <CircularProgress size={20} />}
                                    {params.InputProps.endAdornment}
                                  </>
                                )
                              }}
                            />
                          )}
                          renderOption={(props, option) => (
                            <Box component="li" {...props}>
                              <Avatar sx={{ mr: 2, width: 32, height: 32 }}>
                                {option.name.charAt(0)}
                              </Avatar>
                              <Box>
                                <Typography variant="body2">{option.name}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {option.email}
                                </Typography>
                              </Box>
                            </Box>
                          )}
                        />
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Permission Level
                    </Typography>

                    <FormControl component="fieldset" fullWidth>
                      <RadioGroup
                        value={shareSettings.permission}
                        onChange={(e) => setShareSettings(prev => ({ ...prev, permission: e.target.value }))}
                      >
                        {PERMISSION_LEVELS.map((perm) => (
                          <FormControlLabel
                            key={perm.level}
                            value={perm.level}
                            control={<Radio />}
                            label={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {perm.icon}
                                <Box>
                                  <Typography variant="body2">{perm.label}</Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {perm.description}
                                  </Typography>
                                </Box>
                              </Box>
                            }
                            sx={{ mb: 1 }}
                          />
                        ))}
                      </RadioGroup>
                    </FormControl>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Advanced Options
                    </Typography>

                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <DateTimePicker
                          label="Expires At (Optional)"
                          value={shareSettings.expiresAt || null}
                          onChange={(date) => setShareSettings(prev => ({ 
                            ...prev, 
                            expiresAt: date || undefined 
                          }))}
                          renderInput={(params) => <TextField {...params} fullWidth />}
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={shareSettings.passwordProtected}
                              onChange={(e) => setShareSettings(prev => ({
                                ...prev,
                                passwordProtected: e.target.checked
                              }))}
                            />
                          }
                          label="Password Protection"
                        />

                        {shareSettings.passwordProtected && (
                          <TextField
                            fullWidth
                            type="password"
                            label="Access Password"
                            value={shareSettings.password || ''}
                            onChange={(e) => setShareSettings(prev => ({
                              ...prev,
                              password: e.target.value
                            }))}
                            margin="normal"
                            size="small"
                            InputProps={{
                              startAdornment: (
                                <InputAdornment position="start">
                                  <LockIcon />
                                </InputAdornment>
                              )
                            }}
                          />
                        )}
                      </Grid>

                      {shareSettings.target.type === 'public' && (
                        <>
                          <Grid item xs={12}>
                            <TextField
                              fullWidth
                              label="Title (Optional)"
                              value={shareSettings.title || ''}
                              onChange={(e) => setShareSettings(prev => ({
                                ...prev,
                                title: e.target.value
                              }))}
                              placeholder="Custom title for shared link"
                            />
                          </Grid>

                          <Grid item xs={12}>
                            <TextField
                              fullWidth
                              multiline
                              rows={3}
                              label="Description (Optional)"
                              value={shareSettings.description || ''}
                              onChange={(e) => setShareSettings(prev => ({
                                ...prev,
                                description: e.target.value
                              }))}
                              placeholder="Describe what you're sharing"
                            />
                          </Grid>
                        </>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Existing Shares Tab */}
          {activeTab === 'existing' && (
            <Box>
              {sharesLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : existingShares?.shares?.length > 0 ? (
                <List>
                  {existingShares.shares.map((share: any, index: number) => (
                    <React.Fragment key={index}>
                      <ListItem>
                        <ListItemAvatar>
                          <Avatar>
                            {share.target_type === 'team' ? <GroupIcon /> : <PersonIcon />}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={share.target_name || `${share.target_type} share`}
                          secondary={
                            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 1 }}>
                              <Chip size="small" label={share.permission_level} />
                              {share.expires_at && (
                                <Chip
                                  size="small"
                                  icon={<ScheduleIcon />}
                                  label={`Expires ${new Date(share.expires_at).toLocaleDateString()}`}
                                  variant="outlined"
                                />
                              )}
                              {share.password_protected && (
                                <Chip
                                  size="small"
                                  icon={<SecurityIcon />}
                                  label="Password protected"
                                  variant="outlined"
                                />
                              )}
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Button size="small" variant="outlined">
                            Edit
                          </Button>
                        </ListItemSecondaryAction>
                      </ListItem>
                      {index < existingShares.shares.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Alert severity="info">
                  No existing shares found. Create a new share to get started.
                </Alert>
              )}
            </Box>
          )}

          {/* Public Link Tab */}
          {activeTab === 'public' && (
            <Box>
              {publicLinkCreated && publicLink ? (
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <CheckIcon color="success" />
                      <Typography variant="h6" color="success.main">
                        Public Link Created!
                      </Typography>
                    </Box>
                    
                    <Paper sx={{ p: 2, bgcolor: 'grey.50', mb: 2 }}>
                      <TextField
                        fullWidth
                        value={publicLink}
                        InputProps={{
                          readOnly: true,
                          startAdornment: (
                            <InputAdornment position="start">
                              <LinkIcon />
                            </InputAdornment>
                          ),
                          endAdornment: (
                            <InputAdornment position="end">
                              <Tooltip title="Copy link">
                                <IconButton onClick={handleCopyLink}>
                                  <CopyIcon />
                                </IconButton>
                              </Tooltip>
                            </InputAdornment>
                          )
                        }}
                      />
                    </Paper>

                    <Alert severity="warning" sx={{ mb: 2 }}>
                      This link allows anyone with access to view your portfolio. 
                      Share it carefully and consider setting an expiration date.
                    </Alert>

                    <Box sx={{ display: 'flex', gap: 2 }}>
                      <Button
                        variant="contained"
                        startIcon={<CopyIcon />}
                        onClick={handleCopyLink}
                      >
                        Copy Link
                      </Button>
                      <Button
                        variant="outlined"
                        onClick={() => {
                          setPublicLinkCreated(false);
                          setActiveTab('share');
                        }}
                      >
                        Create Another
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              ) : (
                <Alert severity="info">
                  Create a public link to share your portfolio with anyone. 
                  Switch to "New Share" tab and select "Public Link" option.
                </Alert>
              )}
            </Box>
          )}
        </DialogContent>

        <DialogActions sx={{ p: 2 }}>
          <Button onClick={handleClose}>
            Cancel
          </Button>
          {activeTab === 'share' && (
            <Button
              variant="contained"
              onClick={handleShare}
              disabled={sharePortfolioMutation.isLoading}
              startIcon={sharePortfolioMutation.isLoading ? <CircularProgress size={16} /> : <ShareIcon />}
            >
              {sharePortfolioMutation.isLoading ? 'Sharing...' : 'Share Portfolio'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </LocalizationProvider>
  );
};

export default PortfolioSharingDialog;