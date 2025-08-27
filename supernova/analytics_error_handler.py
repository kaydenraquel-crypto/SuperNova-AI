"""
Advanced Error Handling and Logging for Analytics System
Comprehensive error handling with detailed logging and user-friendly messages
"""

import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import json
import asyncio
from contextlib import asynccontextmanager
from functools import wraps

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd
import numpy as np

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

class AnalyticsErrorCode(str, Enum):
    """Analytics-specific error codes"""
    # Data errors
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_DATA_FORMAT = "INVALID_DATA_FORMAT"
    DATA_QUALITY_LOW = "DATA_QUALITY_LOW"
    MISSING_BENCHMARK = "MISSING_BENCHMARK"
    
    # Calculation errors
    CALCULATION_FAILED = "CALCULATION_FAILED"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    CORRELATION_ERROR = "CORRELATION_ERROR"
    OPTIMIZATION_ERROR = "OPTIMIZATION_ERROR"
    
    # Portfolio errors
    PORTFOLIO_NOT_FOUND = "PORTFOLIO_NOT_FOUND"
    PORTFOLIO_ACCESS_DENIED = "PORTFOLIO_ACCESS_DENIED"
    NO_POSITIONS = "NO_POSITIONS"
    INVALID_POSITION_DATA = "INVALID_POSITION_DATA"
    
    # Risk calculation errors
    VAR_CALCULATION_ERROR = "VAR_CALCULATION_ERROR"
    RISK_MODEL_ERROR = "RISK_MODEL_ERROR"
    STRESS_TEST_ERROR = "STRESS_TEST_ERROR"
    
    # Report generation errors
    REPORT_GENERATION_FAILED = "REPORT_GENERATION_FAILED"
    REPORT_NOT_FOUND = "REPORT_NOT_FOUND"
    REPORT_EXPIRED = "REPORT_EXPIRED"
    
    # External service errors
    MARKET_DATA_UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"
    SENTIMENT_SERVICE_ERROR = "SENTIMENT_SERVICE_ERROR"
    BENCHMARK_DATA_ERROR = "BENCHMARK_DATA_ERROR"
    
    # System errors
    MEMORY_ERROR = "MEMORY_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

class AnalyticsException(Exception):
    """Base exception for analytics operations"""
    
    def __init__(
        self,
        message: str,
        error_code: AnalyticsErrorCode,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.user_message = user_message or self._get_user_friendly_message(error_code)
        self.status_code = status_code
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def _get_user_friendly_message(self, error_code: AnalyticsErrorCode) -> str:
        """Get user-friendly error message"""
        messages = {
            AnalyticsErrorCode.INSUFFICIENT_DATA: "Not enough data available for analysis. Please try a longer time period.",
            AnalyticsErrorCode.INVALID_DATA_FORMAT: "The data format is invalid. Please check your inputs and try again.",
            AnalyticsErrorCode.DATA_QUALITY_LOW: "The data quality is too low for reliable analysis. Results may be inaccurate.",
            AnalyticsErrorCode.MISSING_BENCHMARK: "Benchmark data is not available for comparison.",
            AnalyticsErrorCode.CALCULATION_FAILED: "The calculation could not be completed. Please try again.",
            AnalyticsErrorCode.NUMERICAL_ERROR: "A numerical error occurred during calculation. Please check your data.",
            AnalyticsErrorCode.PORTFOLIO_NOT_FOUND: "The specified portfolio could not be found.",
            AnalyticsErrorCode.PORTFOLIO_ACCESS_DENIED: "You do not have permission to access this portfolio.",
            AnalyticsErrorCode.NO_POSITIONS: "No positions found in the portfolio.",
            AnalyticsErrorCode.VAR_CALCULATION_ERROR: "Value at Risk calculation failed. Please check the data and try again.",
            AnalyticsErrorCode.REPORT_GENERATION_FAILED: "Report generation failed. Please try again later.",
            AnalyticsErrorCode.REPORT_NOT_FOUND: "The requested report could not be found.",
            AnalyticsErrorCode.REPORT_EXPIRED: "The report has expired and is no longer available.",
            AnalyticsErrorCode.MARKET_DATA_UNAVAILABLE: "Market data is temporarily unavailable. Please try again later.",
            AnalyticsErrorCode.TIMEOUT_ERROR: "The operation timed out. Please try again with a smaller dataset.",
            AnalyticsErrorCode.MEMORY_ERROR: "Insufficient memory to complete the operation. Please try a smaller dataset."
        }
        return messages.get(error_code, "An error occurred while processing your request.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": "AnalyticsException",
            "error_code": self.error_code.value,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "status_code": self.status_code
        }

class DataValidationException(AnalyticsException):
    """Exception for data validation errors"""
    
    def __init__(self, message: str, field: str, value: Any, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=AnalyticsErrorCode.INVALID_DATA_FORMAT,
            details={
                "field": field,
                "invalid_value": str(value),
                **(details or {})
            },
            status_code=400
        )
        self.field = field
        self.value = value

class InsufficientDataException(AnalyticsException):
    """Exception for insufficient data scenarios"""
    
    def __init__(self, required_points: int, actual_points: int, data_type: str = "data"):
        super().__init__(
            message=f"Insufficient {data_type}: need at least {required_points} points, got {actual_points}",
            error_code=AnalyticsErrorCode.INSUFFICIENT_DATA,
            details={
                "required_points": required_points,
                "actual_points": actual_points,
                "data_type": data_type
            },
            status_code=400
        )

class CalculationException(AnalyticsException):
    """Exception for calculation errors"""
    
    def __init__(self, calculation_type: str, original_error: Exception, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Failed to calculate {calculation_type}: {str(original_error)}",
            error_code=AnalyticsErrorCode.CALCULATION_FAILED,
            details={
                "calculation_type": calculation_type,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__,
                **(details or {})
            },
            status_code=500
        )
        self.calculation_type = calculation_type
        self.original_error = original_error

class AnalyticsErrorHandler:
    """Centralized error handling for analytics operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        user_id: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        """Log error with context"""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "user_id": user_id,
            "request_id": request_id,
            "traceback": traceback.format_exc()
        }
        
        if isinstance(error, AnalyticsException):
            error_data.update({
                "error_code": error.error_code.value,
                "details": error.details,
                "status_code": error.status_code
            })
            self.logger.error(f"Analytics error: {error.error_code.value}", extra=error_data)
        else:
            self.logger.error(f"Unexpected error: {str(error)}", extra=error_data)
        
        # Track error frequency
        error_key = f"{type(error).__name__}:{str(error)[:100]}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def handle_pandas_error(self, error: Exception, operation: str, data_info: Dict[str, Any]) -> AnalyticsException:
        """Handle pandas-related errors"""
        if isinstance(error, (ValueError, KeyError)):
            return DataValidationException(
                message=f"Pandas operation failed: {str(error)}",
                field=operation,
                value=data_info,
                details={"pandas_error": True}
            )
        elif isinstance(error, MemoryError):
            return AnalyticsException(
                message="Insufficient memory for pandas operation",
                error_code=AnalyticsErrorCode.MEMORY_ERROR,
                details={"operation": operation, "data_info": data_info}
            )
        else:
            return CalculationException(operation, error, {"pandas_error": True})
    
    def handle_numpy_error(self, error: Exception, operation: str, array_info: Dict[str, Any]) -> AnalyticsException:
        """Handle numpy-related errors"""
        if isinstance(error, np.linalg.LinAlgError):
            return AnalyticsException(
                message=f"Linear algebra error in {operation}: {str(error)}",
                error_code=AnalyticsErrorCode.NUMERICAL_ERROR,
                details={"operation": operation, "array_info": array_info}
            )
        elif isinstance(error, (ValueError, TypeError)):
            return DataValidationException(
                message=f"NumPy operation failed: {str(error)}",
                field=operation,
                value=array_info,
                details={"numpy_error": True}
            )
        else:
            return CalculationException(operation, error, {"numpy_error": True})
    
    def validate_dataframe(self, df: pd.DataFrame, min_rows: int = 1, required_columns: Optional[List[str]] = None) -> None:
        """Validate DataFrame meets requirements"""
        if df is None or df.empty:
            raise InsufficientDataException(
                required_points=min_rows,
                actual_points=0,
                data_type="DataFrame rows"
            )
        
        if len(df) < min_rows:
            raise InsufficientDataException(
                required_points=min_rows,
                actual_points=len(df),
                data_type="DataFrame rows"
            )
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise DataValidationException(
                    message=f"Missing required columns: {missing_columns}",
                    field="columns",
                    value=list(df.columns),
                    details={"missing_columns": list(missing_columns)}
                )
        
        # Check for excessive NaN values
        nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if nan_ratio > 0.5:
            raise AnalyticsException(
                message=f"Too many missing values: {nan_ratio:.1%}",
                error_code=AnalyticsErrorCode.DATA_QUALITY_LOW,
                details={"nan_ratio": nan_ratio}
            )
    
    def validate_series(self, series: pd.Series, min_length: int = 1, allow_nan: bool = False) -> None:
        """Validate Series meets requirements"""
        if series is None or series.empty:
            raise InsufficientDataException(
                required_points=min_length,
                actual_points=0,
                data_type="Series values"
            )
        
        if len(series) < min_length:
            raise InsufficientDataException(
                required_points=min_length,
                actual_points=len(series),
                data_type="Series values"
            )
        
        if not allow_nan and series.isnull().any():
            nan_count = series.isnull().sum()
            raise DataValidationException(
                message=f"Series contains {nan_count} NaN values",
                field="series_values",
                value=nan_count,
                details={"nan_positions": series[series.isnull()].index.tolist()}
            )

# Decorators for error handling
def handle_analytics_errors(operation_name: str = "analytics operation"):
    """Decorator to handle analytics errors"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = AnalyticsErrorHandler()
            try:
                return await func(*args, **kwargs)
            except AnalyticsException:
                raise  # Re-raise analytics exceptions as-is
            except pd.errors.EmptyDataError:
                raise InsufficientDataException(1, 0, "pandas data")
            except pd.errors.ParserError as e:
                raise DataValidationException(
                    message=f"Data parsing error: {str(e)}",
                    field="data_format",
                    value="unparseable"
                )
            except np.linalg.LinAlgError as e:
                raise AnalyticsException(
                    message=f"Linear algebra error: {str(e)}",
                    error_code=AnalyticsErrorCode.NUMERICAL_ERROR,
                    details={"operation": operation_name}
                )
            except MemoryError:
                raise AnalyticsException(
                    message="Insufficient memory for operation",
                    error_code=AnalyticsErrorCode.MEMORY_ERROR,
                    details={"operation": operation_name}
                )
            except asyncio.TimeoutError:
                raise AnalyticsException(
                    message="Operation timed out",
                    error_code=AnalyticsErrorCode.TIMEOUT_ERROR,
                    details={"operation": operation_name}
                )
            except Exception as e:
                error_handler.log_error(e, {"operation": operation_name})
                raise CalculationException(operation_name, e)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = AnalyticsErrorHandler()
            try:
                return func(*args, **kwargs)
            except AnalyticsException:
                raise  # Re-raise analytics exceptions as-is
            except pd.errors.EmptyDataError:
                raise InsufficientDataException(1, 0, "pandas data")
            except pd.errors.ParserError as e:
                raise DataValidationException(
                    message=f"Data parsing error: {str(e)}",
                    field="data_format",
                    value="unparseable"
                )
            except np.linalg.LinAlgError as e:
                raise AnalyticsException(
                    message=f"Linear algebra error: {str(e)}",
                    error_code=AnalyticsErrorCode.NUMERICAL_ERROR,
                    details={"operation": operation_name}
                )
            except MemoryError:
                raise AnalyticsException(
                    message="Insufficient memory for operation",
                    error_code=AnalyticsErrorCode.MEMORY_ERROR,
                    details={"operation": operation_name}
                )
            except Exception as e:
                error_handler.log_error(e, {"operation": operation_name})
                raise CalculationException(operation_name, e)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def safe_calculation(default_value: Any = None):
    """Decorator for safe numerical calculations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Check for invalid numerical results
                if isinstance(result, (int, float)):
                    if np.isnan(result) or np.isinf(result):
                        logger.warning(f"Invalid numerical result in {func.__name__}: {result}")
                        return default_value
                elif isinstance(result, np.ndarray):
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        logger.warning(f"Invalid values in array result from {func.__name__}")
                        if default_value is not None:
                            return default_value
                        # Replace invalid values with zeros
                        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                
                return result
                
            except (ZeroDivisionError, FloatingPointError) as e:
                logger.warning(f"Numerical error in {func.__name__}: {str(e)}")
                return default_value
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator

class AnalyticsErrorMiddleware(BaseHTTPMiddleware):
    """Middleware to handle analytics errors globally"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except AnalyticsException as e:
            # Log the error
            logger.error(f"Analytics error: {e.error_code.value}", extra={
                "path": request.url.path,
                "method": request.method,
                "error_details": e.to_dict()
            })
            
            # Return user-friendly error response
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": True,
                    "error_code": e.error_code.value,
                    "message": e.user_message,
                    "details": e.details,
                    "timestamp": e.timestamp.isoformat()
                }
            )
        except HTTPException:
            raise  # Let FastAPI handle HTTP exceptions
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in analytics middleware: {str(e)}", extra={
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc()
            })
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_code": "INTERNAL_ERROR",
                    "message": "An internal error occurred. Please try again later.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

# Context managers for error handling
@asynccontextmanager
async def analytics_operation_context(operation_name: str, user_id: Optional[int] = None):
    """Context manager for analytics operations with error handling"""
    start_time = datetime.utcnow()
    error_handler = AnalyticsErrorHandler()
    
    try:
        logger.info(f"Starting analytics operation: {operation_name}", extra={
            "operation": operation_name,
            "user_id": user_id,
            "start_time": start_time.isoformat()
        })
        yield error_handler
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Completed analytics operation: {operation_name}", extra={
            "operation": operation_name,
            "user_id": user_id,
            "duration_seconds": duration
        })
        
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        error_handler.log_error(e, {
            "operation": operation_name,
            "user_id": user_id,
            "duration_seconds": duration
        })
        raise

# Utility functions
def create_error_response(error: AnalyticsException) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": True,
        "error_code": error.error_code.value,
        "message": error.user_message,
        "details": error.details,
        "timestamp": error.timestamp.isoformat()
    }

def log_analytics_performance(operation: str, duration: float, data_size: Optional[int] = None):
    """Log performance metrics for analytics operations"""
    performance_data = {
        "operation": operation,
        "duration_seconds": duration,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data_size is not None:
        performance_data["data_size"] = data_size
        performance_data["throughput"] = data_size / duration if duration > 0 else 0
    
    logger.info(f"Analytics performance: {operation}", extra=performance_data)