import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { createContext, useContext, ReactNode } from 'react';

// Types
interface ApiConfig {
  baseURL: string;
  timeout: number;
  retryAttempts: number;
}

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: string;
}

interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  preferences?: Record<string, any>;
}

interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

interface RegisterData {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
}

interface AuthResponse {
  user: User;
  tokens: AuthTokens;
}

// API Configuration
const defaultConfig: ApiConfig = {
  baseURL: process.env.NODE_ENV === 'development' 
    ? 'http://localhost:8081/api'
    : '/api',
  timeout: 30000,
  retryAttempts: 3,
};

class ApiService {
  private client: AxiosInstance;
  private config: ApiConfig;

  constructor(config: Partial<ApiConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
    this.client = this.createAxiosInstance();
  }

  private createAxiosInstance(): AxiosInstance {
    const client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    client.interceptors.request.use(
      (config) => {
        const tokens = this.getStoredTokens();
        if (tokens?.accessToken) {
          config.headers.Authorization = `Bearer ${tokens.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and token refresh
    client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const tokens = this.getStoredTokens();
            if (tokens?.refreshToken) {
              const newTokens = await this.refreshToken(tokens.refreshToken);
              this.storeTokens(newTokens.tokens);
              
              // Retry original request with new token
              originalRequest.headers.Authorization = `Bearer ${newTokens.tokens.accessToken}`;
              return client(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed, clear tokens and redirect to login
            this.clearTokens();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );

    return client;
  }

  private getStoredTokens(): AuthTokens | null {
    try {
      const stored = localStorage.getItem('superNova_tokens');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  }

  private storeTokens(tokens: AuthTokens): void {
    localStorage.setItem('superNova_tokens', JSON.stringify(tokens));
  }

  private clearTokens(): void {
    localStorage.removeItem('superNova_tokens');
  }

  // Authentication Methods
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/login', credentials);
    this.storeTokens(response.data.tokens);
    return response.data;
  }

  async register(data: RegisterData): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/register', data);
    this.storeTokens(response.data.tokens);
    return response.data;
  }

  async logout(allDevices: boolean = false): Promise<void> {
    try {
      await this.client.post('/auth/logout', { all_devices: allDevices });
    } finally {
      this.clearTokens();
    }
  }

  async refreshToken(refreshToken: string): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/refresh', {
      refresh_token: refreshToken,
    });
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/profile');
    return response.data;
  }

  async updateProfile(updates: Partial<User>): Promise<User> {
    const response = await this.client.patch<User>('/auth/profile', updates);
    return response.data;
  }

  // MFA Methods
  async setupMFA(password: string): Promise<any> {
    const response = await this.client.post('/auth/mfa/setup', { password });
    return response.data;
  }

  async verifyMFA(token: string, backupCode?: string): Promise<void> {
    await this.client.post('/auth/mfa/verify', { 
      token, 
      backup_code: backupCode 
    });
  }

  async disableMFA(password: string): Promise<void> {
    await this.client.post('/auth/mfa/disable', { password });
  }

  // Password Management
  async changePassword(currentPassword: string, newPassword: string, confirmPassword: string): Promise<void> {
    await this.client.post('/auth/password/change', {
      current_password: currentPassword,
      new_password: newPassword,
      confirm_password: confirmPassword,
    });
  }

  async requestPasswordReset(email: string): Promise<void> {
    await this.client.post('/auth/password/reset', { email });
  }

  async confirmPasswordReset(token: string, newPassword: string, confirmPassword: string): Promise<void> {
    await this.client.post('/auth/password/reset/confirm', {
      token,
      new_password: newPassword,
      confirm_password: confirmPassword,
    });
  }

  // Dashboard Methods
  async getDashboardData(): Promise<any> {
    const response = await this.client.get('/dashboard');
    return response.data;
  }

  // Portfolio Methods
  async getPortfolioOverview(): Promise<any> {
    const response = await this.client.get('/portfolio/overview');
    return response.data;
  }

  async getPortfolioHoldings(): Promise<any> {
    const response = await this.client.get('/portfolio/holdings');
    return response.data;
  }

  async getPortfolioPerformance(timeframe: string = '1Y'): Promise<any> {
    const response = await this.client.get(`/portfolio/performance?timeframe=${timeframe}`);
    return response.data;
  }

  // Market Data Methods
  async getMarketOverview(): Promise<any> {
    const response = await this.client.get('/market/overview');
    return response.data;
  }

  async getWatchlist(): Promise<any> {
    const response = await this.client.get('/market/watchlist');
    return response.data;
  }

  async addToWatchlist(symbols: string[]): Promise<any> {
    const response = await this.client.post('/market/watchlist', { symbols });
    return response.data;
  }

  async removeFromWatchlist(symbols: string[]): Promise<any> {
    const response = await this.client.delete('/market/watchlist', { data: { symbols } });
    return response.data;
  }

  async getStockQuote(symbol: string): Promise<any> {
    const response = await this.client.get(`/market/quote/${symbol}`);
    return response.data;
  }

  async getHistoricalData(symbol: string, timeframe: string = '1Y'): Promise<any> {
    const response = await this.client.get(`/market/history/${symbol}?timeframe=${timeframe}`);
    return response.data;
  }

  // Chat Methods
  async getChatSessions(): Promise<any> {
    const response = await this.client.get('/chat/sessions');
    return response.data.sessions;
  }

  async createChatSession(): Promise<any> {
    const response = await this.client.post('/chat/session');
    return response.data;
  }

  async getChatMessages(sessionId: string): Promise<any> {
    const response = await this.client.get(`/chat/session/${sessionId}`);
    return response.data.messages;
  }

  async sendChatMessage(sessionId: string, message: string): Promise<any> {
    const response = await this.client.post('/chat', {
      message,
      session_id: sessionId,
    });
    return response.data;
  }

  async deleteChatSession(sessionId: string): Promise<void> {
    await this.client.delete(`/chat/session/${sessionId}`);
  }

  async submitChatFeedback(messageId: string, rating: number, feedback?: string): Promise<void> {
    await this.client.post('/chat/feedback', {
      message_id: messageId,
      rating,
      feedback,
    });
  }

  // Backtesting Methods
  async runBacktest(params: any): Promise<any> {
    const response = await this.client.post('/backtest', params);
    return response.data;
  }

  async getBacktestResults(backtestId: string): Promise<any> {
    const response = await this.client.get(`/backtest/${backtestId}`);
    return response.data;
  }

  async getBacktestHistory(): Promise<any> {
    const response = await this.client.get('/backtest/history');
    return response.data;
  }

  // Optimization Methods
  async startOptimization(params: any): Promise<any> {
    const response = await this.client.post('/optimize/strategy', params);
    return response.data;
  }

  async getOptimizationProgress(studyId: string): Promise<any> {
    const response = await this.client.get(`/optimize/progress/${studyId}`);
    return response.data;
  }

  async getOptimizationResults(studyId: string): Promise<any> {
    const response = await this.client.get(`/optimize/study/${studyId}`);
    return response.data;
  }

  // Sentiment Analysis Methods
  async getSentimentData(symbol: string, timeframe: string = '1D'): Promise<any> {
    const response = await this.client.get(`/sentiment/historical/${symbol}?timeframe=${timeframe}`);
    return response.data;
  }

  // News Methods
  async getMarketNews(limit: number = 20): Promise<any> {
    const response = await this.client.get(`/news/market?limit=${limit}`);
    return response.data;
  }

  async getSymbolNews(symbol: string, limit: number = 10): Promise<any> {
    const response = await this.client.get(`/news/symbol/${symbol}?limit=${limit}`);
    return response.data;
  }

  // Alerts Methods
  async getAlerts(): Promise<any> {
    const response = await this.client.get('/alerts');
    return response.data;
  }

  async createAlert(alertData: any): Promise<any> {
    const response = await this.client.post('/alerts', alertData);
    return response.data;
  }

  async deleteAlert(alertId: string): Promise<void> {
    await this.client.delete(`/alerts/${alertId}`);
  }

  // Settings Methods
  async getUserSettings(): Promise<any> {
    const response = await this.client.get('/settings');
    return response.data;
  }

  async updateUserSettings(settings: any): Promise<any> {
    const response = await this.client.patch('/settings', settings);
    return response.data;
  }

  // Utility Methods
  async healthCheck(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // File upload method
  async uploadFile(file: File, type: 'document' | 'image' | 'data' = 'document'): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    const response = await this.client.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }
}

// Create default API service instance
export const apiService = new ApiService();

// API Context for dependency injection
const ApiContext = createContext<ApiService>(apiService);

export const ApiProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <ApiContext.Provider value={apiService}>
      {children}
    </ApiContext.Provider>
  );
};

export const useApi = (): ApiService => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

// Export types
export type {
  ApiConfig,
  AuthTokens,
  User,
  LoginCredentials,
  RegisterData,
  AuthResponse,
};

export default apiService;