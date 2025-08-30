export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'user' | 'admin' | 'premium';
  avatar?: string;
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    language: string;
    timezone: string;
    notifications: {
      email: boolean;
      push: boolean;
      sms: boolean;
    };
  };
  subscription: {
    plan: 'free' | 'basic' | 'premium' | 'enterprise';
    status: 'active' | 'inactive' | 'trial' | 'expired';
    expiresAt?: string;
  };
  createdAt: string;
  updatedAt: string;
  lastLoginAt?: string;
  isEmailVerified: boolean;
  isTwoFactorEnabled: boolean;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  tokenType: 'Bearer';
  expiresIn: number;
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
  twoFactorCode?: string;
}

export interface RegisterData {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  acceptTerms: boolean;
  marketingConsent?: boolean;
}

export interface ResetPasswordData {
  email: string;
}

export interface ChangePasswordData {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

export interface TwoFactorSetupData {
  secret: string;
  qrCode: string;
  backupCodes: string[];
}

export interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  resetPassword: (data: ResetPasswordData) => Promise<void>;
  changePassword: (data: ChangePasswordData) => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
  verifyEmail: (token: string) => Promise<void>;
  setupTwoFactor: () => Promise<TwoFactorSetupData>;
  verifyTwoFactor: (code: string) => Promise<void>;
  disableTwoFactor: (code: string) => Promise<void>;
}

export interface AuthError {
  code: string;
  message: string;
  field?: string;
}

export interface LoginResponse {
  user: User;
  tokens: AuthTokens;
}

export interface RefreshTokenResponse {
  tokens: AuthTokens;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: AuthError;
  message?: string;
}