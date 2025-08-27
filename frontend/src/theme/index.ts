import { createTheme, Theme, ThemeOptions } from '@mui/material/styles';
import { PaletteMode } from '@mui/material';

// SuperNova brand colors
export const superNovaColors = {
  primary: {
    50: '#e3f2fd',
    100: '#bbdefb',
    200: '#90caf9',
    300: '#64b5f6',
    400: '#42a5f5',
    500: '#1976d2', // Main brand color
    600: '#1565c0',
    700: '#1976d2',
    800: '#1565c0',
    900: '#0d47a1',
  },
  secondary: {
    50: '#e8f5e8',
    100: '#c8e6c9',
    200: '#a5d6a7',
    300: '#81c784',
    400: '#66bb6a',
    500: '#4caf50', // Success green
    600: '#43a047',
    700: '#388e3c',
    800: '#2e7d32',
    900: '#1b5e20',
  },
  accent: {
    50: '#fff3e0',
    100: '#ffe0b2',
    200: '#ffcc82',
    300: '#ffb74d',
    400: '#ffa726',
    500: '#ff9800', // Warning/accent orange
    600: '#fb8c00',
    700: '#f57c00',
    800: '#ef6c00',
    900: '#e65100',
  },
  financial: {
    bull: '#2e7d32', // Green for positive
    bear: '#d32f2f', // Red for negative
    neutral: '#757575', // Gray for neutral
    volume: '#9c27b0', // Purple for volume
  },
  chart: {
    grid: '#e0e0e0',
    axis: '#757575',
    line1: '#1976d2',
    line2: '#4caf50',
    line3: '#ff9800',
    line4: '#9c27b0',
    line5: '#f44336',
    candlestick: {
      up: '#2e7d32',
      down: '#d32f2f',
      wick: '#757575',
    },
  },
};

// Typography configuration
const typography = {
  fontFamily: [
    'Inter',
    'Roboto',
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    '"Helvetica Neue"',
    'Arial',
    'sans-serif',
  ].join(','),
  h1: {
    fontWeight: 700,
    fontSize: '2.5rem',
    lineHeight: 1.2,
    letterSpacing: '-0.02em',
  },
  h2: {
    fontWeight: 600,
    fontSize: '2rem',
    lineHeight: 1.3,
    letterSpacing: '-0.01em',
  },
  h3: {
    fontWeight: 600,
    fontSize: '1.5rem',
    lineHeight: 1.4,
    letterSpacing: '-0.005em',
  },
  h4: {
    fontWeight: 500,
    fontSize: '1.25rem',
    lineHeight: 1.4,
  },
  h5: {
    fontWeight: 500,
    fontSize: '1.125rem',
    lineHeight: 1.5,
  },
  h6: {
    fontWeight: 500,
    fontSize: '1rem',
    lineHeight: 1.5,
  },
  subtitle1: {
    fontWeight: 400,
    fontSize: '1rem',
    lineHeight: 1.75,
  },
  subtitle2: {
    fontWeight: 500,
    fontSize: '0.875rem',
    lineHeight: 1.57,
  },
  body1: {
    fontWeight: 400,
    fontSize: '1rem',
    lineHeight: 1.5,
  },
  body2: {
    fontWeight: 400,
    fontSize: '0.875rem',
    lineHeight: 1.43,
  },
  button: {
    fontWeight: 500,
    fontSize: '0.875rem',
    textTransform: 'none' as const,
  },
  caption: {
    fontWeight: 400,
    fontSize: '0.75rem',
    lineHeight: 1.66,
  },
  overline: {
    fontWeight: 700,
    fontSize: '0.625rem',
    lineHeight: 2.5,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
  },
};

// Component customizations
const getComponents = (mode: PaletteMode) => ({
  MuiCssBaseline: {
    styleOverrides: {
      html: {
        fontSize: '16px',
        WebkitFontSmoothing: 'antialiased',
        MozOsxFontSmoothing: 'grayscale',
      },
      body: {
        scrollbarGutter: 'stable',
      },
      '*::-webkit-scrollbar': {
        width: '8px',
        height: '8px',
      },
      '*::-webkit-scrollbar-track': {
        backgroundColor: mode === 'dark' ? '#2e2e2e' : '#f1f1f1',
      },
      '*::-webkit-scrollbar-thumb': {
        backgroundColor: mode === 'dark' ? '#555' : '#c1c1c1',
        borderRadius: '4px',
      },
      '*::-webkit-scrollbar-thumb:hover': {
        backgroundColor: mode === 'dark' ? '#777' : '#a8a8a8',
      },
    },
  },
  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        boxShadow: 'none',
        borderBottom: `1px solid ${mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)'}`,
      },
    },
  },
  MuiDrawer: {
    styleOverrides: {
      paper: {
        backgroundImage: 'none',
        borderRight: `1px solid ${mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)'}`,
      },
    },
  },
  MuiCard: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        boxShadow: mode === 'dark' 
          ? '0 1px 3px rgba(0, 0, 0, 0.3), 0 4px 8px rgba(0, 0, 0, 0.15)'
          : '0 1px 3px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.05)',
        transition: 'box-shadow 0.15s ease-in-out, transform 0.15s ease-in-out',
        '&:hover': {
          boxShadow: mode === 'dark'
            ? '0 4px 8px rgba(0, 0, 0, 0.3), 0 8px 16px rgba(0, 0, 0, 0.15)'
            : '0 4px 8px rgba(0, 0, 0, 0.12), 0 8px 16px rgba(0, 0, 0, 0.05)',
        },
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
      },
      elevation1: {
        boxShadow: mode === 'dark'
          ? '0 1px 3px rgba(0, 0, 0, 0.3)'
          : '0 1px 3px rgba(0, 0, 0, 0.12)',
      },
    },
  },
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        textTransform: 'none',
        fontWeight: 500,
        fontSize: '0.875rem',
        padding: '8px 16px',
        minHeight: 36,
      },
      contained: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        },
      },
    },
  },
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 16,
        fontSize: '0.75rem',
        height: 24,
      },
    },
  },
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 8,
        },
      },
    },
  },
  MuiDataGrid: {
    styleOverrides: {
      root: {
        border: 'none',
        '& .MuiDataGrid-cell': {
          borderColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
        },
        '& .MuiDataGrid-columnHeaders': {
          backgroundColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
          borderBottom: `2px solid ${mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)'}`,
        },
        '& .MuiDataGrid-row:hover': {
          backgroundColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)',
        },
      },
    },
  },
});

// Light theme configuration
const lightTheme: ThemeOptions = {
  palette: {
    mode: 'light',
    primary: {
      main: superNovaColors.primary[500],
      light: superNovaColors.primary[300],
      dark: superNovaColors.primary[700],
      contrastText: '#ffffff',
    },
    secondary: {
      main: superNovaColors.secondary[500],
      light: superNovaColors.secondary[300],
      dark: superNovaColors.secondary[700],
      contrastText: '#ffffff',
    },
    error: {
      main: '#d32f2f',
      light: '#ef5350',
      dark: '#c62828',
      contrastText: '#ffffff',
    },
    warning: {
      main: superNovaColors.accent[500],
      light: superNovaColors.accent[300],
      dark: superNovaColors.accent[700],
      contrastText: '#000000',
    },
    success: {
      main: superNovaColors.secondary[500],
      light: superNovaColors.secondary[300],
      dark: superNovaColors.secondary[700],
      contrastText: '#ffffff',
    },
    background: {
      default: '#fafafa',
      paper: '#ffffff',
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.6)',
      disabled: 'rgba(0, 0, 0, 0.38)',
    },
    divider: 'rgba(0, 0, 0, 0.12)',
    action: {
      active: 'rgba(0, 0, 0, 0.54)',
      hover: 'rgba(0, 0, 0, 0.04)',
      selected: 'rgba(0, 0, 0, 0.08)',
      disabled: 'rgba(0, 0, 0, 0.26)',
      disabledBackground: 'rgba(0, 0, 0, 0.12)',
    },
  },
  typography,
  shape: {
    borderRadius: 8,
  },
  spacing: 8,
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 900,
      lg: 1200,
      xl: 1536,
    },
  },
};

// Dark theme configuration
const darkTheme: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: {
      main: superNovaColors.primary[400],
      light: superNovaColors.primary[200],
      dark: superNovaColors.primary[600],
      contrastText: '#ffffff',
    },
    secondary: {
      main: superNovaColors.secondary[400],
      light: superNovaColors.secondary[200],
      dark: superNovaColors.secondary[600],
      contrastText: '#ffffff',
    },
    error: {
      main: '#f44336',
      light: '#e57373',
      dark: '#d32f2f',
      contrastText: '#ffffff',
    },
    warning: {
      main: superNovaColors.accent[400],
      light: superNovaColors.accent[200],
      dark: superNovaColors.accent[600],
      contrastText: '#000000',
    },
    success: {
      main: superNovaColors.secondary[400],
      light: superNovaColors.secondary[200],
      dark: superNovaColors.secondary[600],
      contrastText: '#ffffff',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
      disabled: 'rgba(255, 255, 255, 0.5)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
    action: {
      active: '#fff',
      hover: 'rgba(255, 255, 255, 0.08)',
      selected: 'rgba(255, 255, 255, 0.12)',
      disabled: 'rgba(255, 255, 255, 0.3)',
      disabledBackground: 'rgba(255, 255, 255, 0.12)',
    },
  },
  typography,
  shape: {
    borderRadius: 8,
  },
  spacing: 8,
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 900,
      lg: 1200,
      xl: 1536,
    },
  },
};

// Create theme instances
export const createSuperNovaTheme = (mode: PaletteMode): Theme => {
  const baseTheme = mode === 'dark' ? darkTheme : lightTheme;
  const theme = createTheme(baseTheme);
  
  return createTheme(theme, {
    components: getComponents(mode),
  });
};

// Default themes
export const lightSuperNovaTheme = createSuperNovaTheme('light');
export const darkSuperNovaTheme = createSuperNovaTheme('dark');

// Theme utilities
export const getFinancialColor = (value: number, mode: PaletteMode = 'light') => {
  if (value > 0) return superNovaColors.financial.bull;
  if (value < 0) return superNovaColors.financial.bear;
  return superNovaColors.financial.neutral;
};

export const formatFinancialNumber = (
  value: number,
  options: {
    currency?: boolean;
    percentage?: boolean;
    decimals?: number;
    compact?: boolean;
  } = {}
) => {
  const { currency = false, percentage = false, decimals = 2, compact = false } = options;
  
  let formattedValue: string;
  
  if (compact && Math.abs(value) >= 1000) {
    const units = ['', 'K', 'M', 'B', 'T'];
    let unitIndex = 0;
    let compactValue = Math.abs(value);
    
    while (compactValue >= 1000 && unitIndex < units.length - 1) {
      compactValue /= 1000;
      unitIndex++;
    }
    
    formattedValue = (value < 0 ? '-' : '') + compactValue.toFixed(decimals) + units[unitIndex];
  } else {
    formattedValue = value.toFixed(decimals);
  }
  
  if (percentage) {
    formattedValue += '%';
  } else if (currency) {
    formattedValue = '$' + formattedValue;
  }
  
  return formattedValue;
};

export const getChartColors = (mode: PaletteMode = 'light') => {
  return mode === 'dark' 
    ? {
        ...superNovaColors.chart,
        grid: '#333333',
        axis: '#999999',
      }
    : superNovaColors.chart;
};

// Responsive breakpoint utilities
export const useResponsive = () => {
  return {
    isMobile: window.innerWidth < 600,
    isTablet: window.innerWidth >= 600 && window.innerWidth < 900,
    isDesktop: window.innerWidth >= 900,
    isLarge: window.innerWidth >= 1200,
  };
};

export default {
  light: lightSuperNovaTheme,
  dark: darkSuperNovaTheme,
  create: createSuperNovaTheme,
  colors: superNovaColors,
  getFinancialColor,
  formatFinancialNumber,
  getChartColors,
};