import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  Avatar,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  Link,
} from '@mui/material';
import {
  MoreVert,
  Article,
  TrendingUp,
  Public,
  Schedule,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

dayjs.extend(relativeTime);

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  publishedAt: string;
  url: string;
  category: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  symbols: string[];
}

interface NewsCardProps {
  loading?: boolean;
}

const NewsCard: React.FC<NewsCardProps> = ({
  loading = false,
}) => {
  const theme = useTheme();
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [newsItems] = useState<NewsItem[]>([
    {
      id: '1',
      title: 'Tesla Reports Strong Q4 Earnings Despite Market Concerns',
      summary: 'Tesla exceeded analyst expectations with record-breaking vehicle deliveries...',
      source: 'Financial Times',
      publishedAt: '2024-01-15T08:30:00Z',
      url: '#',
      category: 'Earnings',
      sentiment: 'positive',
      symbols: ['TSLA'],
    },
    {
      id: '2',
      title: 'Fed Signals Potential Rate Cuts in 2024',
      summary: 'Federal Reserve officials hint at policy changes amid cooling inflation...',
      source: 'Reuters',
      publishedAt: '2024-01-15T07:15:00Z',
      url: '#',
      category: 'Economic Policy',
      sentiment: 'positive',
      symbols: ['SPY', 'QQQ'],
    },
    {
      id: '3',
      title: 'Tech Sector Faces Regulatory Scrutiny',
      summary: 'New antitrust investigations launched against major technology companies...',
      source: 'Wall Street Journal',
      publishedAt: '2024-01-14T18:45:00Z',
      url: '#',
      category: 'Regulation',
      sentiment: 'negative',
      symbols: ['AAPL', 'GOOGL', 'MSFT'],
    },
    {
      id: '4',
      title: 'Oil Prices Surge on Geopolitical Tensions',
      summary: 'Crude oil futures jump 3% amid supply concerns from Middle East conflicts...',
      source: 'Bloomberg',
      publishedAt: '2024-01-14T16:20:00Z',
      url: '#',
      category: 'Commodities',
      sentiment: 'neutral',
      symbols: ['XOM', 'CVX'],
    },
  ]);

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return theme.palette.success.main;
      case 'negative':
        return theme.palette.error.main;
      case 'neutral':
        return theme.palette.text.secondary;
      default:
        return theme.palette.text.secondary;
    }
  };

  const getSentimentLabel = (sentiment: string) => {
    return sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      'Earnings': theme.palette.info.main,
      'Economic Policy': theme.palette.warning.main,
      'Regulation': theme.palette.error.main,
      'Commodities': theme.palette.success.main,
    };
    return colors[category as keyof typeof colors] || theme.palette.primary.main;
  };

  const formatTimeAgo = (timestamp: string) => {
    return dayjs(timestamp).fromNow();
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="h3">
            Market News
          </Typography>
          <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
            <MoreVert />
          </IconButton>
        </Box>

        <List disablePadding>
          {newsItems.map((item) => (
            <ListItem
              key={item.id}
              sx={{
                px: 0,
                py: 1.5,
                borderBottom: 1,
                borderColor: 'divider',
                '&:last-child': {
                  borderBottom: 0,
                },
              }}
            >
              <Avatar
                sx={{ 
                  bgcolor: getCategoryColor(item.category),
                  width: 36,
                  height: 36,
                  mr: 2,
                }}
              >
                <Article />
              </Avatar>

              <ListItemText
                primary={
                  <Link
                    href={item.url}
                    color="inherit"
                    underline="hover"
                    sx={{ 
                      display: 'block',
                      mb: 0.5,
                      fontWeight: 600,
                      fontSize: '0.9rem',
                      lineHeight: 1.3,
                    }}
                  >
                    {item.title}
                  </Link>
                }
                secondary={
                  <Box>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ 
                        mb: 1,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}
                    >
                      {item.summary}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center' }}>
                      <Chip
                        label={item.category}
                        size="small"
                        variant="outlined"
                        sx={{ 
                          fontSize: '0.7rem',
                          height: 20,
                          borderColor: getCategoryColor(item.category),
                          color: getCategoryColor(item.category),
                        }}
                      />
                      
                      <Chip
                        label={getSentimentLabel(item.sentiment)}
                        size="small"
                        variant="filled"
                        sx={{ 
                          fontSize: '0.7rem',
                          height: 20,
                          bgcolor: getSentimentColor(item.sentiment),
                          color: 'white',
                        }}
                      />

                      {item.symbols.map((symbol) => (
                        <Chip
                          key={symbol}
                          label={symbol}
                          size="small"
                          variant="outlined"
                          sx={{ 
                            fontSize: '0.7rem',
                            height: 20,
                          }}
                        />
                      ))}
                    </Box>

                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, gap: 1 }}>
                      <Public sx={{ fontSize: 14, color: 'text.secondary' }} />
                      <Typography variant="caption" color="text.secondary">
                        {item.source}
                      </Typography>
                      <Schedule sx={{ fontSize: 14, color: 'text.secondary' }} />
                      <Typography variant="caption" color="text.secondary">
                        {formatTimeAgo(item.publishedAt)}
                      </Typography>
                    </Box>
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>

        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={() => setMenuAnchor(null)}>
            View All News
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Customize Sources
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            News Preferences
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default NewsCard;