import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { PaletteMode } from '@mui/material';

interface ThemeContextType {
  mode: PaletteMode;
  toggleTheme: () => void;
  setTheme: (mode: PaletteMode) => void;
  isDark: boolean;
  isLight: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeContextProviderProps {
  children: React.ReactNode;
  defaultMode?: PaletteMode;
}

export const ThemeContextProvider: React.FC<ThemeContextProviderProps> = ({ 
  children, 
  defaultMode = 'light' 
}) => {
  const [mode, setMode] = useState<PaletteMode>(() => {
    // Check localStorage first
    const savedMode = localStorage.getItem('superNovaTheme') as PaletteMode | null;
    if (savedMode && ['light', 'dark'].includes(savedMode)) {
      return savedMode;
    }
    
    // Fall back to system preference or default
    if (typeof window !== 'undefined' && window.matchMedia) {
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      return systemPrefersDark ? 'dark' : 'light';
    }
    
    return defaultMode;
  });

  // Save theme preference
  useEffect(() => {
    localStorage.setItem('superNovaTheme', mode);
    
    // Update HTML attribute for CSS-based theming
    document.documentElement.setAttribute('data-theme', mode);
    
    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', mode === 'dark' ? '#121212' : '#1976d2');
    }
  }, [mode]);

  // Listen for system theme changes
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) {
      return;
    }
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      // Only change if user hasn't manually set a preference
      if (!localStorage.getItem('superNovaTheme')) {
        setMode(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const toggleTheme = useCallback(() => {
    setMode(prevMode => prevMode === 'light' ? 'dark' : 'light');
  }, []);

  const setTheme = useCallback((newMode: PaletteMode) => {
    setMode(newMode);
  }, []);

  const contextValue: ThemeContextType = {
    mode,
    toggleTheme,
    setTheme,
    isDark: mode === 'dark',
    isLight: mode === 'light',
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeContextProvider');
  }
  return context;
};

// Hook for responsive design based on theme breakpoints
export const useResponsiveTheme = () => {
  const [screenSize, setScreenSize] = useState(() => {
    if (typeof window === 'undefined') return 'md';
    
    const width = window.innerWidth;
    if (width < 600) return 'xs';
    if (width < 900) return 'sm';
    if (width < 1200) return 'md';
    if (width < 1536) return 'lg';
    return 'xl';
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleResize = () => {
      const width = window.innerWidth;
      let newSize = 'md';
      
      if (width < 600) newSize = 'xs';
      else if (width < 900) newSize = 'sm';
      else if (width < 1200) newSize = 'md';
      else if (width < 1536) newSize = 'lg';
      else newSize = 'xl';
      
      setScreenSize(newSize);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return {
    screenSize,
    isMobile: screenSize === 'xs',
    isTablet: screenSize === 'sm',
    isDesktop: ['md', 'lg', 'xl'].includes(screenSize),
    isLargeScreen: ['lg', 'xl'].includes(screenSize),
    breakpoints: {
      xs: screenSize === 'xs',
      sm: screenSize === 'sm',
      md: screenSize === 'md',
      lg: screenSize === 'lg',
      xl: screenSize === 'xl',
    },
  };
};

// Hook for theme-aware color utilities
export const useThemeColors = () => {
  const { mode } = useTheme();
  
  const getFinancialColor = useCallback((value: number) => {
    const colors = {
      positive: mode === 'dark' ? '#4caf50' : '#2e7d32',
      negative: mode === 'dark' ? '#f44336' : '#d32f2f',
      neutral: mode === 'dark' ? '#9e9e9e' : '#757575',
    };
    
    if (value > 0) return colors.positive;
    if (value < 0) return colors.negative;
    return colors.neutral;
  }, [mode]);

  const getChartColors = useCallback(() => {
    return mode === 'dark' 
      ? {
          grid: '#333333',
          axis: '#999999',
          line1: '#2196f3',
          line2: '#4caf50',
          line3: '#ff9800',
          line4: '#9c27b0',
          line5: '#f44336',
          background: 'rgba(255, 255, 255, 0.05)',
        }
      : {
          grid: '#e0e0e0',
          axis: '#757575',
          line1: '#1976d2',
          line2: '#2e7d32',
          line3: '#f57c00',
          line4: '#7b1fa2',
          line5: '#c62828',
          background: 'rgba(0, 0, 0, 0.02)',
        };
  }, [mode]);

  const getSurfaceColor = useCallback((elevation: number = 1) => {
    const baseColor = mode === 'dark' ? '#1e1e1e' : '#ffffff';
    const overlay = mode === 'dark' 
      ? `rgba(255, 255, 255, ${0.05 + elevation * 0.02})` 
      : `rgba(0, 0, 0, ${0.02 + elevation * 0.01})`;
    
    return { backgroundColor: baseColor, background: overlay };
  }, [mode]);

  return {
    mode,
    getFinancialColor,
    getChartColors,
    getSurfaceColor,
    isDark: mode === 'dark',
    isLight: mode === 'light',
  };
};

export default useTheme;