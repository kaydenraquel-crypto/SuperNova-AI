import React from 'react';
import {
  Box,
  Chip,
  Typography,
  Paper,
  Grid,
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  Search,
  Help,
  AccountBalance,
  Psychology,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface Suggestion {
  id: string;
  text: string;
  icon?: React.ReactNode;
  category: string;
}

interface ChatSuggestionsProps {
  show: boolean;
  onSuggestionClick: (suggestion: string) => void;
}

const ChatSuggestions: React.FC<ChatSuggestionsProps> = ({
  show,
  onSuggestionClick,
}) => {
  const theme = useTheme();

  const suggestions: Suggestion[] = [
    {
      id: '1',
      text: "What's my portfolio performance today?",
      icon: <TrendingUp />,
      category: 'Portfolio',
    },
    {
      id: '2',
      text: "Analyze AAPL stock for me",
      icon: <Assessment />,
      category: 'Analysis',
    },
    {
      id: '3',
      text: "Find stocks with strong momentum",
      icon: <Search />,
      category: 'Research',
    },
    {
      id: '4',
      text: "Explain market volatility",
      icon: <Help />,
      category: 'Education',
    },
    {
      id: '5',
      text: "Show me sector allocation",
      icon: <AccountBalance />,
      category: 'Portfolio',
    },
    {
      id: '6',
      text: "What are the market trends?",
      icon: <Psychology />,
      category: 'Market',
    },
  ];

  const groupedSuggestions = suggestions.reduce((acc, suggestion) => {
    if (!acc[suggestion.category]) {
      acc[suggestion.category] = [];
    }
    acc[suggestion.category].push(suggestion);
    return acc;
  }, {} as Record<string, Suggestion[]>);

  if (!show) return null;

  return (
    <Paper
      sx={{
        p: 3,
        mb: 2,
        bgcolor: theme.palette.background.default,
        border: 1,
        borderColor: 'divider',
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
        Try asking me about:
      </Typography>

      {Object.entries(groupedSuggestions).map(([category, categorySuggestions]) => (
        <Box key={category} sx={{ mb: 3 }}>
          <Typography 
            variant="subtitle2" 
            color="text.secondary" 
            sx={{ mb: 1, fontWeight: 600 }}
          >
            {category}
          </Typography>
          
          <Grid container spacing={1}>
            {categorySuggestions.map((suggestion) => (
              <Grid item key={suggestion.id}>
                <Chip
                  label={suggestion.text}
                  onClick={() => onSuggestionClick(suggestion.text)}
                  icon={suggestion.icon}
                  clickable
                  variant="outlined"
                  sx={{
                    height: 'auto',
                    py: 1,
                    px: 1.5,
                    '& .MuiChip-label': {
                      whiteSpace: 'normal',
                      lineHeight: 1.2,
                    },
                    '&:hover': {
                      bgcolor: theme.palette.primary.main,
                      color: theme.palette.primary.contrastText,
                      borderColor: theme.palette.primary.main,
                      '& .MuiChip-icon': {
                        color: theme.palette.primary.contrastText,
                      },
                    },
                    transition: 'all 0.2s ease-in-out',
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      ))}

      <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary">
          💡 Tip: You can ask me anything about your investments, market analysis, or trading strategies!
        </Typography>
      </Box>
    </Paper>
  );
};

export default ChatSuggestions;