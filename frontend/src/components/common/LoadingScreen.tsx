import React from 'react';
import { Box, CircularProgress, Typography, Fade } from '@mui/material';
import { TrendingUp } from '@mui/icons-material';

interface LoadingScreenProps {
  message?: string;
  size?: 'small' | 'medium' | 'large';
  variant?: 'page' | 'overlay' | 'inline';
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Loading...',
  size = 'medium',
  variant = 'page',
}) => {
  const sizeMap = {
    small: 32,
    medium: 48,
    large: 64,
  };

  const progressSize = sizeMap[size];

  const content = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        p: 3,
      }}
    >
      <Box sx={{ position: 'relative' }}>
        <CircularProgress
          size={progressSize}
          thickness={4}
          sx={{
            color: 'primary.main',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <TrendingUp
            sx={{
              fontSize: progressSize * 0.4,
              color: 'primary.main',
            }}
          />
        </Box>
      </Box>
      
      <Typography
        variant={size === 'small' ? 'body2' : 'body1'}
        color="text.secondary"
        textAlign="center"
      >
        {message}
      </Typography>
    </Box>
  );

  if (variant === 'page') {
    return (
      <Fade in timeout={300}>
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.default',
            zIndex: 9999,
          }}
        >
          {content}
        </Box>
      </Fade>
    );
  }

  if (variant === 'overlay') {
    return (
      <Fade in timeout={300}>
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(4px)',
            zIndex: 1000,
          }}
        >
          {content}
        </Box>
      </Fade>
    );
  }

  // inline variant
  return (
    <Fade in timeout={300}>
      <Box>{content}</Box>
    </Fade>
  );
};

export default LoadingScreen;