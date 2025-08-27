# SuperNova AI - Frontend (Material-UI)

A professional React/TypeScript frontend application with Material-UI components for the SuperNova AI Financial Platform.

## Features

### 🎨 Professional UI/UX
- **Material-UI v5** with custom SuperNova theme
- **Dark/Light Mode** with system preference detection  
- **Responsive Design** optimized for desktop, tablet, and mobile
- **Professional Financial Dashboard** with real-time data visualization
- **Accessibility** WCAG 2.1 AA compliant components

### 💰 Financial Components
- **Portfolio Overview** with performance metrics
- **Real-time Market Data** integration
- **Interactive Charts** with Recharts and TradingView integration
- **Advanced Data Grid** for financial data tables
- **Financial Number Formatting** with currency and percentage support

### 🤖 AI Chat Interface  
- **Enhanced Chat UI** with Material-UI components
- **Real-time WebSocket** communication
- **Voice Message** support with recording capabilities
- **File Upload** with drag-and-drop functionality
- **Message Threading** and conversation management
- **Typing Indicators** and presence detection

### 📊 Advanced Features
- **Progressive Web App** (PWA) support
- **Service Worker** for offline functionality
- **Code Splitting** and lazy loading
- **Performance Monitoring** and error boundaries
- **TypeScript** for type safety
- **React Query** for server state management

## Quick Start

### Prerequisites
- Node.js 16+ 
- npm 8+
- SuperNova Backend running on port 8081

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at `http://localhost:3000`

### Build for Production

```bash
# Build optimized production bundle
npm run build

# Analyze bundle size
npm run analyze
```

## Project Structure

```
frontend/
├── public/              # Static assets
│   ├── index.html      # HTML template
│   ├── manifest.json   # PWA manifest  
│   └── favicon.ico     # Favicon
├── src/
│   ├── components/     # Reusable UI components
│   │   ├── common/     # Common components (LoadingScreen, ErrorBoundary)
│   │   ├── dashboard/  # Dashboard-specific components
│   │   ├── charts/     # Chart components (FinancialChart)
│   │   ├── chat/       # Chat interface components
│   │   └── layout/     # Layout components (MainLayout, AuthLayout)
│   ├── pages/          # Page components
│   │   ├── DashboardPage.tsx
│   │   ├── ChatPage.tsx
│   │   ├── LoginPage.tsx
│   │   └── ...
│   ├── hooks/          # Custom React hooks
│   │   ├── useAuth.tsx
│   │   ├── useTheme.tsx  
│   │   ├── useWebSocket.tsx
│   │   └── ...
│   ├── services/       # API and external services
│   │   ├── api.ts      # Main API service
│   │   └── ...
│   ├── types/          # TypeScript type definitions
│   ├── theme/          # Material-UI theme configuration
│   │   └── index.ts    # SuperNova theme
│   ├── utils/          # Utility functions
│   ├── App.tsx         # Main App component
│   └── index.tsx       # Application entry point
├── package.json        # Dependencies and scripts
├── tsconfig.json       # TypeScript configuration
├── webpack.config.js   # Webpack configuration
└── README.md          # This file
```

## Key Components

### Theme System (`src/theme/`)
- **SuperNova Brand Colors** with financial-specific color schemes
- **Dark/Light Mode** support with automatic switching
- **Responsive Breakpoints** for mobile-first design
- **Financial Utilities** for number formatting and color coding

### Authentication (`src/hooks/useAuth.tsx`)
- **JWT Token Management** with automatic refresh
- **React Query Integration** for cached user data
- **Route Protection** with automatic redirects
- **Social Login** support (Google, GitHub)

### Real-time Data (`src/hooks/useWebSocket.tsx`)
- **WebSocket Connection** management with auto-reconnection
- **Market Data Streaming** with subscription management
- **Chat WebSocket** for real-time messaging
- **Connection Status** monitoring and display

### API Service (`src/services/api.ts`)
- **Axios-based HTTP Client** with request/response interceptors
- **Automatic Token Refresh** on 401 responses
- **Request Retry Logic** with exponential backoff
- **Type-safe API Methods** for all backend endpoints

## Available Scripts

```bash
# Development
npm start                 # Start development server
npm run build:dev         # Build for development

# Production  
npm run build            # Build for production
npm run analyze          # Analyze bundle size

# Code Quality
npm run lint             # Run ESLint
npm run lint:fix         # Fix ESLint errors
npm run type-check       # Run TypeScript compiler

# Testing
npm test                 # Run tests
npm run test:watch       # Run tests in watch mode  
npm run test:coverage    # Run tests with coverage

# Storybook (Component Development)
npm run storybook        # Start Storybook dev server
npm run build-storybook  # Build Storybook
```

## Configuration

### Environment Variables
Create a `.env` file in the frontend directory:

```env
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8081/api
REACT_APP_WS_URL=ws://localhost:8081

# Feature Flags  
REACT_APP_ENABLE_PWA=true
REACT_APP_ENABLE_ANALYTICS=false

# External Services
REACT_APP_GOOGLE_CLIENT_ID=your_google_client_id
REACT_APP_GITHUB_CLIENT_ID=your_github_client_id
```

### Webpack Configuration
The webpack configuration includes:
- **TypeScript** compilation with ts-loader
- **Code Splitting** with dynamic imports
- **PWA Support** with Workbox
- **Development Proxy** to backend API
- **Bundle Optimization** for production

## Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t supernova-frontend .

# Run container
docker run -p 3000:80 supernova-frontend
```

### Static Deployment
The built files in `/dist` can be deployed to any static hosting service:
- **Netlify, Vercel** - Automatic deployments from Git
- **AWS S3 + CloudFront** - Scalable static hosting
- **Nginx** - Traditional web server setup

## Browser Support

- **Chrome** 90+
- **Firefox** 88+  
- **Safari** 14+
- **Edge** 90+

## Performance

### Optimization Features
- **Code Splitting** reduces initial bundle size
- **Lazy Loading** for route-based components  
- **Image Optimization** with webpack loaders
- **Service Worker** caching for offline support
- **Bundle Analysis** tools for monitoring

### Performance Metrics
- **First Contentful Paint** < 1.5s
- **Largest Contentful Paint** < 2.5s
- **Time to Interactive** < 3.5s
- **Cumulative Layout Shift** < 0.1

## Contributing

### Code Style
- **Prettier** for code formatting
- **ESLint** for code linting  
- **TypeScript** strict mode enabled
- **Material-UI** component patterns

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with tests
3. Run `npm run lint && npm run type-check`
4. Submit pull request with description

## Troubleshooting

### Common Issues

**Port 3000 already in use:**
```bash
# Kill process on port 3000
npx kill-port 3000

# Or use different port
PORT=3001 npm start
```

**WebSocket connection errors:**
- Ensure backend is running on port 8081
- Check firewall settings for WebSocket connections
- Verify CORS configuration in backend

**Build errors:**
```bash  
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear TypeScript cache
npx tsc --build --clean
```

## License

Copyright © 2024 SuperNova AI. All rights reserved.