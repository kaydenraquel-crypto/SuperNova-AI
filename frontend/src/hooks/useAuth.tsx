import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { apiService } from '@/services/api';
import type { User, AuthTokens, LoginCredentials, RegisterData } from '@/types/auth';

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  refreshToken: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(() => {
    try {
      const stored = localStorage.getItem('superNova_tokens');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });
  const [isInitialized, setIsInitialized] = useState(false);

  const queryClient = useQueryClient();

  // Store tokens in localStorage
  const storeTokens = useCallback((newTokens: AuthTokens | null) => {
    setTokens(newTokens);
    if (newTokens) {
      localStorage.setItem('superNova_tokens', JSON.stringify(newTokens));
      // Set up automatic token refresh
      scheduleTokenRefresh(newTokens);
    } else {
      localStorage.removeItem('superNova_tokens');
    }
  }, []);

  // Schedule automatic token refresh
  const scheduleTokenRefresh = useCallback((authTokens: AuthTokens) => {
    if (!authTokens.refreshToken || !authTokens.expiresAt) return;

    const now = Date.now();
    const expiresAt = new Date(authTokens.expiresAt).getTime();
    const refreshTime = expiresAt - (15 * 60 * 1000); // Refresh 15 minutes before expiry

    if (refreshTime <= now) {
      // Token already expired or will expire very soon
      refreshTokenMutation.mutate();
      return;
    }

    const timeUntilRefresh = refreshTime - now;
    const timeoutId = setTimeout(() => {
      refreshTokenMutation.mutate();
    }, timeUntilRefresh);

    // Store timeout ID to clear it later
    return () => clearTimeout(timeoutId);
  }, []);

  // Fetch current user profile
  const { data: userData, isLoading: isUserLoading } = useQuery(
    'currentUser',
    () => apiService.getCurrentUser(),
    {
      enabled: !!tokens?.accessToken,
      retry: false,
      onSuccess: (data) => {
        setUser(data);
      },
      onError: (error: any) => {
        if (error?.response?.status === 401) {
          // Token is invalid, try to refresh
          refreshTokenMutation.mutate();
        }
      },
    }
  );

  // Login mutation
  const loginMutation = useMutation(
    (credentials: LoginCredentials) => apiService.login(credentials),
    {
      onSuccess: (data) => {
        storeTokens(data.tokens);
        setUser(data.user);
        queryClient.setQueryData('currentUser', data.user);
      },
      onError: (error) => {
        console.error('Login failed:', error);
        throw error;
      },
    }
  );

  // Register mutation
  const registerMutation = useMutation(
    (data: RegisterData) => apiService.register(data),
    {
      onSuccess: (data) => {
        storeTokens(data.tokens);
        setUser(data.user);
        queryClient.setQueryData('currentUser', data.user);
      },
      onError: (error) => {
        console.error('Registration failed:', error);
        throw error;
      },
    }
  );

  // Refresh token mutation
  const refreshTokenMutation = useMutation(
    () => {
      if (!tokens?.refreshToken) {
        throw new Error('No refresh token available');
      }
      return apiService.refreshToken(tokens.refreshToken);
    },
    {
      onSuccess: (data) => {
        storeTokens(data.tokens);
        setUser(data.user);
        queryClient.setQueryData('currentUser', data.user);
      },
      onError: (error) => {
        console.error('Token refresh failed:', error);
        // If refresh fails, logout user
        logout();
      },
    }
  );

  // Update profile mutation
  const updateProfileMutation = useMutation(
    (updates: Partial<User>) => apiService.updateProfile(updates),
    {
      onSuccess: (updatedUser) => {
        setUser(updatedUser);
        queryClient.setQueryData('currentUser', updatedUser);
      },
      onError: (error) => {
        console.error('Profile update failed:', error);
        throw error;
      },
    }
  );

  // Login function
  const login = useCallback(async (credentials: LoginCredentials) => {
    await loginMutation.mutateAsync(credentials);
  }, [loginMutation]);

  // Register function
  const register = useCallback(async (data: RegisterData) => {
    await registerMutation.mutateAsync(data);
  }, [registerMutation]);

  // Logout function
  const logout = useCallback(async () => {
    try {
      if (tokens?.refreshToken) {
        await apiService.logout(tokens.refreshToken);
      }
    } catch (error) {
      console.error('Logout request failed:', error);
    } finally {
      // Clear local state regardless of API call success
      storeTokens(null);
      setUser(null);
      queryClient.clear();
      
      // Clear any scheduled token refresh
      // (handled by storeTokens setting tokens to null)
    }
  }, [tokens, storeTokens, queryClient]);

  // Refresh token function
  const refreshToken = useCallback(async () => {
    await refreshTokenMutation.mutateAsync();
  }, [refreshTokenMutation]);

  // Update profile function
  const updateProfile = useCallback(async (updates: Partial<User>) => {
    await updateProfileMutation.mutateAsync(updates);
  }, [updateProfileMutation]);

  // Check for existing session on mount
  useEffect(() => {
    const initializeAuth = async () => {
      if (tokens?.accessToken) {
        try {
          // Validate existing token by fetching user data
          const userData = await apiService.getCurrentUser();
          setUser(userData);
          scheduleTokenRefresh(tokens);
        } catch (error) {
          console.error('Token validation failed:', error);
          // Try to refresh token if it fails
          if (tokens.refreshToken) {
            try {
              await refreshTokenMutation.mutateAsync();
            } catch (refreshError) {
              console.error('Token refresh on init failed:', refreshError);
              // Clear invalid tokens
              storeTokens(null);
            }
          } else {
            storeTokens(null);
          }
        }
      }
      setIsInitialized(true);
    };

    initializeAuth();
  }, []); // Empty dependency array - only run on mount

  const contextValue: AuthContextType = {
    user,
    tokens,
    isAuthenticated: !!user && !!tokens?.accessToken,
    isLoading: !isInitialized || isUserLoading || loginMutation.isLoading || registerMutation.isLoading,
    login,
    logout,
    register,
    refreshToken,
    updateProfile,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Hook for protected API calls that automatically handle auth
export const useAuthenticatedQuery = <T,>(
  key: string | readonly unknown[],
  queryFn: () => Promise<T>,
  options?: {
    enabled?: boolean;
    retry?: number;
    staleTime?: number;
    cacheTime?: number;
  }
) => {
  const { isAuthenticated, logout } = useAuth();

  return useQuery(key, queryFn, {
    enabled: isAuthenticated && (options?.enabled ?? true),
    retry: options?.retry ?? 3,
    staleTime: options?.staleTime ?? 5 * 60 * 1000,
    cacheTime: options?.cacheTime ?? 10 * 60 * 1000,
    onError: (error: any) => {
      if (error?.response?.status === 401) {
        // Unauthorized - token might be expired
        logout();
      }
    },
    ...options,
  });
};

// Hook for authenticated mutations
export const useAuthenticatedMutation = <TData, TVariables>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options?: {
    onSuccess?: (data: TData, variables: TVariables) => void;
    onError?: (error: any, variables: TVariables) => void;
  }
) => {
  const { logout } = useAuth();
  const queryClient = useQueryClient();

  return useMutation(mutationFn, {
    onSuccess: options?.onSuccess,
    onError: (error: any, variables: TVariables) => {
      if (error?.response?.status === 401) {
        logout();
      }
      options?.onError?.(error, variables);
    },
    onSettled: () => {
      // Invalidate and refetch relevant queries after mutation
      queryClient.invalidateQueries();
    },
  });
};

export default useAuth;