/**
 * Mock Service Worker (MSW) server configuration for API mocking
 */
import { setupServer } from 'msw/node';
import { rest } from 'msw';
import { mockApiResponses, mockOHLCVData, mockPortfolioData, mockChatMessages } from '../index';

export const handlers = [
  // Auth endpoints
  rest.post('/api/auth/login', (req, res, ctx) => {
    return res(
      ctx.json({
        user: {
          id: 1,
          name: 'Test User',
          email: 'test@example.com',
        },
        token: 'mock-jwt-token',
      })
    );
  }),

  rest.post('/api/auth/logout', (req, res, ctx) => {
    return res(ctx.json({ success: true }));
  }),

  // User intake endpoint
  rest.post('/api/intake', (req, res, ctx) => {
    return res(
      ctx.json({
        profileId: 1,
        riskScore: 60,
      })
    );
  }),

  // Advice endpoint
  rest.post('/api/advice', (req, res, ctx) => {
    return res(ctx.json(mockApiResponses.getAdvice));
  }),

  // Backtest endpoint
  rest.post('/api/backtest', (req, res, ctx) => {
    return res(ctx.json(mockApiResponses.getBacktest));
  }),

  // Watchlist endpoints
  rest.post('/api/watchlist/add', (req, res, ctx) => {
    return res(
      ctx.json({
        addedIds: [1, 2, 3],
      })
    );
  }),

  rest.get('/api/watchlist/:profileId', (req, res, ctx) => {
    return res(
      ctx.json([
        { id: 1, symbol: 'AAPL', assetClass: 'stock' },
        { id: 2, symbol: 'GOOGL', assetClass: 'stock' },
        { id: 3, symbol: 'MSFT', assetClass: 'stock' },
      ])
    );
  }),

  // Portfolio endpoints
  rest.get('/api/portfolio/:profileId', (req, res, ctx) => {
    return res(ctx.json(mockPortfolioData));
  }),

  rest.get('/api/portfolio/:profileId/performance', (req, res, ctx) => {
    return res(
      ctx.json({
        daily: Array.from({ length: 30 }, (_, i) => ({
          date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
          value: 100000 + Math.random() * 25000,
        })),
        monthly: Array.from({ length: 12 }, (_, i) => ({
          date: new Date(Date.now() - (11 - i) * 30 * 24 * 60 * 60 * 1000).toISOString(),
          value: 100000 + i * 2000 + Math.random() * 10000,
        })),
      })
    );
  }),

  // Market data endpoints
  rest.get('/api/market/:symbol', (req, res, ctx) => {
    const { symbol } = req.params;
    return res(
      ctx.json({
        symbol,
        price: 150.00 + Math.random() * 50,
        change: (Math.random() - 0.5) * 10,
        changePercent: (Math.random() - 0.5) * 5,
        volume: Math.floor(Math.random() * 100000000),
        bars: mockOHLCVData,
      })
    );
  }),

  rest.get('/api/market/:symbol/bars', (req, res, ctx) => {
    return res(ctx.json(mockOHLCVData));
  }),

  // Chat endpoints
  rest.post('/api/chat', (req, res, ctx) => {
    return res(ctx.json(mockApiResponses.getChatResponse));
  }),

  rest.get('/api/chat/history/:sessionId', (req, res, ctx) => {
    return res(ctx.json(mockChatMessages));
  }),

  rest.post('/api/chat/session', (req, res, ctx) => {
    return res(
      ctx.json({
        sessionId: 'test-session-123',
      })
    );
  }),

  // Alerts endpoint
  rest.post('/api/alerts/evaluate', (req, res, ctx) => {
    return res(
      ctx.json([
        {
          id: 1,
          symbol: 'AAPL',
          message: 'RSI above 70 - potential overbought condition',
          triggeredAt: new Date().toISOString(),
          severity: 'warning',
        },
      ])
    );
  }),

  // Sentiment endpoints
  rest.get('/api/sentiment/:symbol', (req, res, ctx) => {
    const { symbol } = req.params;
    return res(
      ctx.json({
        symbol,
        currentSentiment: Math.random() * 2 - 1, // -1 to 1
        sentimentTrend: 'positive',
        confidence: Math.random(),
        sources: ['twitter', 'news', 'reddit'],
        history: Array.from({ length: 30 }, (_, i) => ({
          date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
          sentiment: Math.random() * 2 - 1,
          confidence: Math.random(),
        })),
      })
    );
  }),

  // Health check endpoint
  rest.get('/api/health', (req, res, ctx) => {
    return res(
      ctx.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: {
          database: 'connected',
          timescale: 'connected',
          redis: 'connected',
        },
      })
    );
  }),

  // Error simulation endpoints
  rest.get('/api/error/500', (req, res, ctx) => {
    return res(ctx.status(500), ctx.json({ error: 'Internal Server Error' }));
  }),

  rest.get('/api/error/404', (req, res, ctx) => {
    return res(ctx.status(404), ctx.json({ error: 'Not Found' }));
  }),

  rest.get('/api/error/timeout', (req, res, ctx) => {
    // Simulate timeout
    return res(ctx.delay(10000), ctx.json({ data: 'delayed' }));
  }),
];

export const server = setupServer(...handlers);