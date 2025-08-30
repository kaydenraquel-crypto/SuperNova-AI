import React from 'react';
import { Box, Typography, keyframes } from '@mui/material';
import { useTheme } from '@mui/material/styles';

const bounce = keyframes`
  0%, 60%, 100% {
    animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
    transform: translate3d(0,0,0);
  }
  25% {
    animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
    transform: translate3d(0,-6px,0);
  }
  50% {
    animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
    transform: translate3d(0,-3px,0);
  }
`;

interface TypingIndicatorProps {
  show: boolean;
  text?: string;
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({ 
  show,
  text = "AI is typing..."
}) => {
  const theme = useTheme();

  if (!show) return null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        py: 2,
        px: 3,
        bgcolor: theme.palette.background.paper,
        borderRadius: 2,
        maxWidth: 200,
        mb: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          gap: 0.5,
          alignItems: 'center',
        }}
      >
        {[0, 1, 2].map((index) => (
          <Box
            key={index}
            sx={{
              width: 8,
              height: 8,
              bgcolor: theme.palette.primary.main,
              borderRadius: '50%',
              animation: `${bounce} 1.4s infinite ease-in-out both`,
              animationDelay: `${index * 0.16}s`,
            }}
          />
        ))}
      </Box>
      
      <Typography 
        variant="caption" 
        color="text.secondary"
        sx={{ fontStyle: 'italic' }}
      >
        {text}
      </Typography>
    </Box>
  );
};

export default TypingIndicator;