import React, { useState, useEffect } from 'react';
import { LinearProgress, Box } from '@mui/material';
import { useLocation } from 'react-router-dom';

const ProgressBar: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const location = useLocation();

  useEffect(() => {
    // Show loading briefly when location changes
    setLoading(true);
    const timer = setTimeout(() => setLoading(false), 300);
    
    return () => clearTimeout(timer);
  }, [location]);

  if (!loading) return null;

  return (
    <Box sx={{ 
      position: 'fixed', 
      top: 0, 
      left: 0, 
      right: 0, 
      zIndex: 9999,
      height: 4
    }}>
      <LinearProgress 
        sx={{ 
          height: 4,
          '& .MuiLinearProgress-bar': {
            transition: 'transform 0.2s ease-in-out'
          }
        }} 
      />
    </Box>
  );
};

export default ProgressBar;