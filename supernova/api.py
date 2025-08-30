import json
import time
import secrets
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import socketio
from .db import init_db, SessionLocal, User, Profile, Asset, WatchlistItem, is_timescale_available, get_timescale_session
from .schemas import (
    # Authentication schemas
    LoginRequest, RegisterRequest, TokenResponse, RefreshTokenRequest,
    MFASetupRequest, MFASetupResponse, MFAVerifyRequest,
    PasswordChangeRequest, PasswordResetRequest, PasswordResetConfirmRequest,
    LogoutRequest, UserProfile, SessionInfo, UserSessionsResponse,
    APIKeyRequest, APIKeyResponse, APIKeyInfo, SecurityLog,
    # Existing schemas
    IntakeRequest, ProfileOut, AdviceRequest, AdviceOut, WatchlistRequest, AlertOut, BacktestRequest, BacktestOut,
    SentimentDataPoint, SentimentHistoryRequest, SentimentHistoryResponse, SentimentSummaryStats,
    SentimentAggregateRequest, SentimentAggregateResponse, TimescaleHealthResponse,
    BulkSentimentInsertRequest, BulkSentimentInsertResponse,
    ChatMessage, ChatSession, ChatRequest, ChatResponse, ChatHistory, ChatFeedback, ChatContext,
    ChatSuggestion, ChatSessionList, ChatAnalytics, WebSocketMessage, PresenceUpdate, VoiceMessage,
    FileShare, MarketDataUpdate, ChartData, ConversationSummary, ChatNotification
)
from .advisor import score_risk, advise
from .backtester import run_backtest, run_vbt_backtest, VBT_AVAILABLE
from .config import settings
from .alerting import evaluate_alerts, send_custom_alert
from .journal import log_event, append_session
from .auth import (
    auth_manager, get_current_user, require_permission, require_admin,
    AuthenticationError, AuthorizationError, MFARequiredError,
    UserRole, Permission, TokenPayload
)
from .api_security import SecurityMiddleware
from sqlalchemy import select, text, func, and_, or_
import logging

# Import analytics API routes
from .analytics_api import router as analytics_router

# Configure logging
logger = logging.getLogger(__name__)

# TimescaleDB imports (optional)
try:
    from .sentiment_models import SentimentData, SentimentAggregates, to_dict
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False
    SentimentData = None
    SentimentAggregates = None
    to_dict = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    log_event("startup", {"service":"api"})
    yield
    # Shutdown
    log_event("shutdown", {"service":"api"})

app = FastAPI(title="SuperNova Advisor API", version="0.1.0", lifespan=lifespan)

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"],
    logger=True,
    engineio_logger=True
)

# Mount Socket.IO to the FastAPI app
socket_asgi_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_asgi_app)

# Add CORS middleware for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Add security middleware - temporarily disabled for error hunting
# app.add_middleware(SecurityMiddleware)

# Add API management middleware - temporarily disabled due to rate limiter interface mismatch
# from .api_management_middleware import APIManagementMiddleware
# app.add_middleware(APIManagementMiddleware)

# Include analytics router
app.include_router(analytics_router)

# Include collaboration router
# Temporarily disabled for testing basic functionality
# from .collaboration_api import router as collaboration_router
# app.include_router(collaboration_router)

# Include WebSocket router
from .websocket_api import router as websocket_router
app.include_router(websocket_router)

# Include API Management router
from .api_management_api import router as api_management_router
app.include_router(api_management_router)

# Include Indicators router
from .indicators_api import router as indicators_router
app.include_router(indicators_router)

# Include History router
from .history_api import router as history_router
app.include_router(history_router)

# Add global OPTIONS handler for CORS preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}

# Add health endpoint for integration testing
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "supernova-api",
        "version": "0.1.0"
    }

# ====================================
# AUTHENTICATION ENDPOINTS
# ====================================

@app.post("/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest, req: Request):
    """Register a new user account"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Validate password strength
        password_validation = auth_manager.validate_password_strength(request.password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
            )
        
        # Create new user
        hashed_password = auth_manager.hash_password(request.password)
        user = User(
            name=request.name,
            email=request.email,
            hashed_password=hashed_password,
            role=request.role.value,
            is_active=True,
            email_verified=False  # In production, require email verification
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create session
        session_id = auth_manager.create_session(user.id, req)
        
        # Create tokens
        token_payload = TokenPayload(
            user_id=user.id,
            email=user.email,
            role=UserRole(user.role),
            permissions=[p.value for p in ROLE_PERMISSIONS[UserRole(user.role)]],
            session_id=session_id
        )
        
        access_token = auth_manager.create_access_token(token_payload)
        refresh_token = auth_manager.create_refresh_token(token_payload)
        
        # Log event
        log_event("user_registration", {
            "user_id": user.id,
            "email": user.email,
            "role": user.role
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.jwt_config["access_token_expire"].total_seconds(),
            user={
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role,
                "mfa_enabled": bool(user.mfa_secret)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )
    finally:
        db.close()

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, req: Request):
    """Login user and return JWT tokens"""
    db = SessionLocal()
    try:
        # Check if account is locked
        if auth_manager.is_account_locked(request.email, req):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to too many failed attempts"
            )
        
        # Find user
        user = db.query(User).filter(User.email == request.email).first()
        if not user or not auth_manager.verify_password(request.password, user.hashed_password):
            # Record failed attempt
            auth_manager.record_failed_attempt(request.email, req)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        # Check MFA if enabled
        if user.mfa_secret and not request.mfa_token:
            raise HTTPException(
                status_code=status.HTTP_200_OK,  # Special status for MFA required
                detail="MFA_REQUIRED",
                headers={"X-MFA-Required": "true"}
            )
        
        if user.mfa_secret and request.mfa_token:
            if not auth_manager.verify_mfa_token(user.mfa_secret, request.mfa_token):
                auth_manager.record_failed_attempt(request.email, req)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid MFA token"
                )
        
        # Clear failed attempts on successful login
        auth_manager.clear_failed_attempts(request.email)
        
        # Create session
        session_id = auth_manager.create_session(user.id, req)
        
        # Create tokens
        token_payload = TokenPayload(
            user_id=user.id,
            email=user.email,
            role=UserRole(user.role),
            permissions=[p.value for p in ROLE_PERMISSIONS[UserRole(user.role)]],
            session_id=session_id
        )
        
        access_token = auth_manager.create_access_token(token_payload)
        refresh_token = auth_manager.create_refresh_token(token_payload)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Log successful login
        log_event("user_login", {
            "user_id": user.id,
            "email": user.email,
            "session_id": session_id
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.jwt_config["access_token_expire"].total_seconds(),
            user={
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role,
                "mfa_enabled": bool(user.mfa_secret)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
    finally:
        db.close()

@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh JWT access token"""
    try:
        # Verify refresh token and create new access token
        new_access_token = auth_manager.refresh_access_token(request.refresh_token)
        
        # Get user info from new token
        payload = auth_manager.verify_token(new_access_token)
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(payload["sub"])).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            return TokenResponse(
                access_token=new_access_token,
                refresh_token=request.refresh_token,  # Keep same refresh token
                expires_in=auth_manager.jwt_config["access_token_expire"].total_seconds(),
                user={
                    "id": user.id,
                    "name": user.name,
                    "email": user.email,
                    "role": user.role,
                    "mfa_enabled": bool(user.mfa_secret)
                }
            )
        finally:
            db.close()
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@app.post("/auth/logout")
async def logout(
    request: LogoutRequest,
    current_user: dict = Depends(get_current_user)
):
    """Logout user and invalidate tokens"""
    try:
        # Get current token from request (this would need to be extracted from middleware)
        # For now, we'll revoke all sessions if requested
        if request.all_devices:
            auth_manager.revoke_all_user_sessions(current_user["sub"])
        else:
            # Revoke current session
            session_id = current_user.get("session_id")
            if session_id:
                auth_manager.revoke_session(session_id)
        
        log_event("user_logout", {
            "user_id": current_user["sub"],
            "all_devices": request.all_devices
        })
        
        return {"message": "Successfully logged out"}
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@app.get("/auth/profile", response_model=UserProfile)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(current_user["sub"])).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserProfile(
            id=user.id,
            name=user.name,
            email=user.email,
            role=UserRole(user.role),
            is_active=user.is_active,
            mfa_enabled=bool(user.mfa_secret),
            created_at=user.created_at,
            last_login=user.last_login
        )
    finally:
        db.close()

@app.post("/auth/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    request: MFASetupRequest,
    current_user: dict = Depends(get_current_user)
):
    """Setup Multi-Factor Authentication"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(current_user["sub"])).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not auth_manager.verify_password(request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )
        
        # Generate MFA secret and QR code
        secret = auth_manager.generate_mfa_secret(user.email)
        qr_code = auth_manager.generate_mfa_qr_code(user.email, secret)
        backup_codes = auth_manager.generate_backup_codes()
        
        # Store secret temporarily (user must verify before it's saved)
        # In a real implementation, you'd store this temporarily and require verification
        user.mfa_secret = secret
        user.mfa_backup_codes = ",".join(backup_codes)  # Store securely in production
        db.commit()
        
        log_event("mfa_setup", {"user_id": user.id})
        
        return MFASetupResponse(
            secret=secret,
            qr_code=qr_code,
            backup_codes=backup_codes
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup failed"
        )
    finally:
        db.close()

@app.post("/auth/password/change")
async def change_password(
    request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Change user password"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(current_user["sub"])).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not auth_manager.verify_password(request.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        password_validation = auth_manager.validate_password_strength(request.new_password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
            )
        
        # Update password
        user.hashed_password = auth_manager.hash_password(request.new_password)
        user.password_changed_at = datetime.utcnow()
        db.commit()
        
        # Revoke all sessions to force re-login
        auth_manager.revoke_all_user_sessions(user.id)
        
        log_event("password_change", {"user_id": user.id})
        
        return {"message": "Password changed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )
    finally:
        db.close()

@app.post("/auth/password/reset")
async def request_password_reset(request: PasswordResetRequest, req: Request):
    """Request password reset token"""
    db = SessionLocal()
    try:
        # Find user
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            # Don't reveal whether email exists - always return success
            return {"message": "If the email exists, a reset link has been sent"}
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        token_hash = auth_manager.hash_password(reset_token)
        
        # Store reset token
        from .db import PasswordResetToken
        reset_record = PasswordResetToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            ip_address=req.client.host,
            user_agent=req.headers.get("user-agent", "")
        )
        db.add(reset_record)
        db.commit()
        
        # In production, send email with reset link
        # For now, just log the token (remove in production)
        logger.info(f"Password reset token for {request.email}: {reset_token}")
        
        log_event("password_reset_requested", {
            "user_id": user.id,
            "email": request.email
        })
        
        return {"message": "If the email exists, a reset link has been sent"}
    
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )
    finally:
        db.close()

@app.post("/auth/password/reset/confirm")
async def confirm_password_reset(request: PasswordResetConfirmRequest):
    """Confirm password reset with token"""
    db = SessionLocal()
    try:
        # Find valid reset token
        from .db import PasswordResetToken
        reset_records = db.query(PasswordResetToken).filter(
            PasswordResetToken.is_used == False,
            PasswordResetToken.expires_at > datetime.utcnow()
        ).all()
        
        # Check if token matches any record
        token_record = None
        for record in reset_records:
            if auth_manager.verify_password(request.token, record.token_hash):
                token_record = record
                break
        
        if not token_record:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Validate new password
        password_validation = auth_manager.validate_password_strength(request.new_password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
            )
        
        # Update user password
        user = db.query(User).filter(User.id == token_record.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user.hashed_password = auth_manager.hash_password(request.new_password)
        user.password_changed_at = datetime.utcnow()
        
        # Mark token as used
        token_record.is_used = True
        token_record.used_at = datetime.utcnow()
        
        db.commit()
        
        # Revoke all sessions
        auth_manager.revoke_all_user_sessions(user.id)
        
        log_event("password_reset_completed", {
            "user_id": user.id,
            "email": user.email
        })
        
        return {"message": "Password reset successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset confirmation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )
    finally:
        db.close()

# ====================================
# EXISTING ENDPOINTS (with authentication)
# ====================================

@app.post("/intake", response_model=ProfileOut)
def intake(req: IntakeRequest, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        u = User(name=req.name, email=req.email)
        db.add(u); db.flush()
        risk = score_risk(req.risk_questions)
        p = Profile(user_id=u.id, risk_score=risk, time_horizon_yrs=req.time_horizon_yrs,
                    objectives=req.objectives, constraints=req.constraints)
        db.add(p); db.commit()
        log_event("intake", {"user_id": u.id, "profile_id": p.id, "risk": risk})
        append_session(
            supervisor="AutoLogger",
            subagents=["IntakeAgent"],
            actions=["Created user and profile", "Computed risk score"],
            results=[f"profile_id={p.id}", f"risk_score={risk}"],
            suggestions=["Review constraints & objectives for suitability"],
            meta={"endpoint":"/intake", "user_id": u.id, "profile_id": p.id}
        )
        return ProfileOut(profile_id=p.id, risk_score=risk)
    finally:
        db.close()

@app.post("/advice", response_model=AdviceOut)
def get_advice(req: AdviceRequest, current_user: dict = Depends(get_current_user)):
    action, conf, details, rationale, risk_notes = advise(
        bars=[b.model_dump() for b in req.bars],
        risk_score=_get_risk(req.profile_id),
        sentiment_hint=req.sentiment_hint,
        template=req.strategy_template,
        params=req.params,
        symbol=req.symbol,
        asset_class=req.asset_class,
        timeframe=req.timeframe,
    )
    log_event("advice", {"profile_id": req.profile_id, "symbol": req.symbol, "action": action, "conf": conf})
    append_session(
        supervisor="AutoLogger",
        subagents=["StrategyAgent","RiskBlendAgent"],
        actions=[f"Evaluated advice for {req.symbol} ({req.timeframe})"],
        results=[f"action={action}", f"confidence={conf:.2f}"] ,
        suggestions=["Backtest this configuration overnight"],
        meta={"endpoint":"/advice", "profile_id": req.profile_id, "symbol": req.symbol, "timeframe": req.timeframe}
    )
    return AdviceOut(symbol=req.symbol, timeframe=req.timeframe, action=action, confidence=conf,
                     rationale=rationale, key_indicators=details, risk_notes=risk_notes)

def _get_risk(profile_id: int) -> int:
    db = SessionLocal()
    try:
        p = db.get(Profile, profile_id)
        if not p: raise HTTPException(404, "Profile not found")
        return p.risk_score
    finally:
        db.close()

@app.post("/watchlist/add")
def add_watchlist(req: WatchlistRequest, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        ids = []
        for sym in req.symbols:
            asset = db.scalar(select(Asset).where(Asset.symbol==sym))
            if not asset:
                asset = Asset(symbol=sym, asset_class=req.asset_class)
                db.add(asset); db.flush()
            wl = WatchlistItem(profile_id=req.profile_id, asset_id=asset.id, notes=None, active=True)
            db.add(wl); db.flush()
            ids.append(wl.id)
        db.commit()
        log_event("watchlist_add", {"profile_id": req.profile_id, "symbols": req.symbols})
        append_session(
            supervisor="AutoLogger",
            subagents=["WatchlistAgent"],
            actions=["Added symbols to watchlist"],
            results=[f"added_ids={ids}"],
            suggestions=["Schedule alert evaluation hourly"],
            meta={"endpoint":"/watchlist/add", "profile_id": req.profile_id, "symbols": req.symbols}
        )
        return {"added_ids": ids}
    finally:
        db.close()

@app.post("/alerts/evaluate", response_model=list[AlertOut])
async def evaluate(req: dict):
    triggered = await evaluate_alerts(req.get("watch", []), req.get("bars", {}))
    append_session(
        supervisor="AutoLogger",
        subagents=["AlertAgent"],
        actions=["Evaluated alerts for watchlist"],
        results=[f"triggered={len(triggered)}"],
        suggestions=["Consider adding MACD-based alerts"],
        meta={"endpoint":"/alerts/evaluate"}
    )
    return [AlertOut(id=i+1, symbol=a["symbol"], message=a["message"], triggered_at="now") for i,a in enumerate(triggered)]

@app.post("/backtest", response_model=BacktestOut)
def backtest(req: BacktestRequest, current_user: dict = Depends(get_current_user)):
    # Validate input data
    if len(req.bars) > settings.BACKTEST_MAX_BARS:
        raise HTTPException(status_code=400, detail=f"Too many bars. Maximum allowed: {settings.BACKTEST_MAX_BARS}")
    
    if len(req.bars) < settings.BACKTEST_MIN_BARS:
        raise HTTPException(status_code=400, detail=f"Insufficient bars. Minimum required: {settings.BACKTEST_MIN_BARS}")
    
    # Determine which backtesting engine to use based on configuration and request
    use_vbt = (
        VBT_AVAILABLE and 
        settings.VECTORBT_ENABLED and 
        req.use_vectorbt and
        (settings.DEFAULT_STRATEGY_ENGINE == "vectorbt" or settings.VECTORBT_DEFAULT_ENGINE)
    )
    engine = "VectorBT" if use_vbt else "Legacy"
    
    # Use configuration defaults if not specified in request
    start_cash = req.start_cash if req.start_cash != 10000.0 else 10000.0
    fees = req.fees if req.fees != 0.001 else settings.VECTORBT_DEFAULT_FEES
    slippage = req.slippage if req.slippage != 0.001 else settings.VECTORBT_DEFAULT_SLIPPAGE
    
    if use_vbt:
        # Use high-performance VectorBT backtester
        metrics = run_vbt_backtest(
            bars=[b.model_dump() for b in req.bars], 
            strategy_template=req.strategy_template, 
            params=req.params,
            start_cash=start_cash,
            fees=fees,
            slippage=slippage
        )
    else:
        # Fallback to legacy backtester
        metrics = run_backtest(
            bars=[b.model_dump() for b in req.bars], 
            template=req.strategy_template, 
            params=req.params,
            start_cash=start_cash,
            fee_rate=fees  # Legacy backtester uses fee_rate instead of fees
        )
    
    # Handle errors
    if isinstance(metrics, dict) and "error" in metrics:
        raise HTTPException(status_code=400, detail=metrics["error"])
    
    # Add engine info to metrics if not present
    if isinstance(metrics, dict):
        metrics["engine"] = engine
        metrics["config_used"] = {
            "vectorbt_enabled": settings.VECTORBT_ENABLED,
            "default_engine": settings.DEFAULT_STRATEGY_ENGINE,
            "fees": fees,
            "slippage": slippage
        }
    
    log_event("backtest", {"template": req.strategy_template, "engine": engine, "metrics": metrics})
    append_session(
        supervisor="AutoLogger",
        subagents=["BacktestAgent"],
        actions=[f"Backtested {req.strategy_template} on {req.symbol} ({req.timeframe}) using {engine}"],
        results=[json.dumps(metrics) if isinstance(metrics, dict) else str(metrics)],
        suggestions=["Store metrics and schedule parameter sweep", "Consider parameter optimization"],
        meta={"endpoint":"/backtest", "symbol": req.symbol, "timeframe": req.timeframe, 
              "template": req.strategy_template, "engine": engine}
    )
    return BacktestOut(symbol=req.symbol, timeframe=req.timeframe, metrics=metrics, engine=engine)

@app.post("/backtest/vectorbt")
def vectorbt_backtest(req: BacktestRequest):
    """
    VectorBT-specific backtesting endpoint with comprehensive strategy templates.
    Supports: sma_crossover, rsi_strategy, macd_strategy, bb_strategy, sentiment_strategy
    """
    if not VBT_AVAILABLE:
        raise HTTPException(status_code=503, detail="VectorBT not available. Install vectorbt, ta-lib, and numba.")
    
    metrics = run_vbt_backtest(
        bars=[b.model_dump() for b in req.bars], 
        strategy_template=req.strategy_template, 
        params=req.params,
        start_cash=req.start_cash,
        fees=req.fees,
        slippage=req.slippage
    )
    
    if isinstance(metrics, dict) and "error" in metrics:
        raise HTTPException(status_code=400, detail=metrics["error"])
    
    log_event("vectorbt_backtest", {
        "template": req.strategy_template, 
        "params": req.params,
        "metrics": metrics
    })
    
    append_session(
        supervisor="AutoLogger",
        subagents=["VectorBTAgent"],
        actions=[f"VectorBT backtested {req.strategy_template} on {req.symbol} ({req.timeframe})"],
        results=[f"Sharpe: {metrics.get('Sharpe', 0):.2f}, CAGR: {metrics.get('CAGR', 0):.1f}%"],
        suggestions=["Optimize parameters", "Try walk-forward analysis", "Compare with benchmark"],
        meta={"endpoint":"/backtest/vectorbt", "symbol": req.symbol, "template": req.strategy_template}
    )
    
    return BacktestOut(symbol=req.symbol, timeframe=req.timeframe, metrics=metrics, engine="VectorBT")

# NovaSignal Integration Endpoints
@app.post("/novasignal/alerts/send")
async def send_novasignal_alert(alert_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Send custom alert to NovaSignal platform"""
    required_fields = ["symbol", "message"]
    for field in required_fields:
        if field not in alert_data:
            raise HTTPException(400, f"Missing required field: {field}")
    
    background_tasks.add_task(send_custom_alert, **alert_data)
    
    log_event("novasignal_alert", {"symbol": alert_data["symbol"]})
    append_session(
        supervisor="AutoLogger",
        subagents=["NovaSignalAgent"],
        actions=["Sent custom alert to NovaSignal"],
        results=[f"symbol={alert_data['symbol']}"],
        suggestions=["Monitor alert delivery status"],
        meta={"endpoint": "/novasignal/alerts/send", "symbol": alert_data["symbol"]}
    )
    
    return {"status": "alert_queued", "symbol": alert_data["symbol"]}


@app.get("/novasignal/status")
async def get_novasignal_status():
    """Get NovaSignal integration connection status"""
    try:
        from ..connectors.novasignal import get_connector
        connector = await get_connector()
        status = await connector.get_connection_status()
        
        return {
            "status": "operational",
            "connection_details": status,
            "timestamp": json.dumps(None)  # Will be filled by actual timestamp
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": json.dumps(None)
        }


@app.get("/novasignal/watchlist/{profile_id}")
async def sync_novasignal_watchlist(profile_id: int):
    """Sync watchlist from NovaSignal for a specific profile"""
    try:
        from ..connectors.novasignal import get_connector
        connector = await get_connector()
        watchlist = await connector.get_watchlist(profile_id)
        
        log_event("novasignal_watchlist_sync", {"profile_id": profile_id, "items": len(watchlist)})
        
        return {
            "profile_id": profile_id,
            "watchlist": watchlist,
            "synced_at": json.dumps(None)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to sync watchlist: {str(e)}")


@app.post("/novasignal/profile/{profile_id}/sync")
async def sync_profile_to_novasignal(profile_id: int, profile_data: Dict[str, Any]):
    """Sync profile data to NovaSignal"""
    try:
        from ..connectors.novasignal import get_connector
        connector = await get_connector()
        success = await connector.sync_profile(profile_id, profile_data)
        
        if success:
            log_event("novasignal_profile_sync", {"profile_id": profile_id})
            return {"status": "synced", "profile_id": profile_id}
        else:
            raise HTTPException(500, "Profile sync failed")
            
    except Exception as e:
        raise HTTPException(500, f"Failed to sync profile: {str(e)}")


@app.get("/novasignal/historical/{symbol}")
async def get_novasignal_historical_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    asset_class: str = "stock"
):
    """Fetch historical data from NovaSignal"""
    try:
        from ..connectors.novasignal import get_connector
        connector = await get_connector()
        bars = await connector.get_historical_data(symbol, timeframe, limit, asset_class)
        
        log_event("novasignal_historical_data", {
            "symbol": symbol, 
            "timeframe": timeframe, 
            "bars_count": len(bars)
        })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": [bar.model_dump() for bar in bars],
            "count": len(bars)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch historical data: {str(e)}")


@app.post("/novasignal/advice-explanation/{profile_id}")
async def send_advice_explanation_to_novasignal(profile_id: int, advice_data: Dict[str, Any]):
    """Send advice explanation to NovaSignal for UI rendering"""
    try:
        from ..connectors.novasignal import get_connector
        connector = await get_connector()
        success = await connector.send_advice_explanation(profile_id, advice_data)
        
        if success:
            log_event("novasignal_advice_explanation", {"profile_id": profile_id})
            return {"status": "sent", "profile_id": profile_id}
        else:
            raise HTTPException(500, "Failed to send advice explanation")
            
    except Exception as e:
        raise HTTPException(500, f"Failed to send advice explanation: {str(e)}")


@app.get("/novasignal/health")
async def novasignal_health_check():
    """Comprehensive health check for NovaSignal integration"""
    try:
        from ..connectors.novasignal import health_check
        health_status = await health_check()
        return health_status
    except Exception as e:
        return {
            "status": "critical_error",
            "error": str(e),
            "timestamp": json.dumps(None)
        }

# ================================
# TIMESCALEDB SENTIMENT ENDPOINTS
# ================================

def _check_timescale_availability():
    """Helper to check TimescaleDB availability and raise appropriate errors."""
    if not TIMESCALE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="TimescaleDB sentiment features not available. Missing dependencies or models."
        )
    
    if not is_timescale_available():
        raise HTTPException(
            status_code=503,
            detail="TimescaleDB not configured or connection failed. Check configuration and database status."
        )

@app.get("/sentiment/historical/{symbol}", response_model=SentimentHistoryResponse)
async def get_sentiment_historical_single(
    symbol: str,
    start_date: datetime = Query(..., description="Start date for historical data"),
    end_date: datetime = Query(..., description="End date for historical data"),
    interval: str = Query("raw", description="Data aggregation interval", pattern="^(raw|1h|6h|1d|1w)$"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Records to skip for pagination")
):
    """
    Get historical sentiment data for a single symbol.
    
    Supports both raw data and time-bucketed aggregations for efficient querying.
    """
    _check_timescale_availability()
    
    start_time = time.time()
    
    try:
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        # Validate date range isn't too large
        max_days = 365  # 1 year maximum
        if (end_date - start_date).days > max_days:
            raise HTTPException(
                status_code=400,
                detail=f"Date range too large. Maximum {max_days} days allowed."
            )
        
        TimescaleSession = get_timescale_session()
        
        with TimescaleSession() as session:
            if interval == "raw":
                # Query raw sentiment data
                query = select(SentimentData).where(
                    and_(
                        SentimentData.symbol == symbol,
                        SentimentData.timestamp >= start_date,
                        SentimentData.timestamp <= end_date
                    )
                )
                
                if min_confidence is not None:
                    query = query.where(SentimentData.confidence >= min_confidence)
                
                query = query.order_by(SentimentData.timestamp.desc()).offset(offset).limit(limit)
                
                results = session.execute(query).scalars().all()
                
            else:
                # Query aggregated data
                query = select(SentimentAggregates).where(
                    and_(
                        SentimentAggregates.symbol == symbol,
                        SentimentAggregates.interval_type == interval,
                        SentimentAggregates.time_bucket >= start_date,
                        SentimentAggregates.time_bucket <= end_date
                    )
                )
                
                query = query.order_by(SentimentAggregates.time_bucket.desc()).offset(offset).limit(limit)
                
                agg_results = session.execute(query).scalars().all()
                
                # Convert aggregated data to SentimentDataPoint format
                results = []
                for agg in agg_results:
                    results.append(SentimentData(
                        symbol=agg.symbol,
                        timestamp=agg.time_bucket,
                        overall_score=agg.avg_score,
                        confidence=agg.avg_confidence,
                        twitter_sentiment=agg.twitter_avg,
                        reddit_sentiment=agg.reddit_avg,
                        news_sentiment=agg.news_avg,
                        total_data_points=agg.data_points
                    ))
            
            # Convert to response format
            data_points = []
            for result in results:
                data_point = SentimentDataPoint(
                    symbol=result.symbol,
                    timestamp=result.timestamp,
                    overall_score=result.overall_score,
                    confidence=result.confidence,
                    social_momentum=result.social_momentum,
                    news_sentiment=result.news_sentiment,
                    twitter_sentiment=result.twitter_sentiment,
                    reddit_sentiment=result.reddit_sentiment,
                    source_counts=result.source_counts,
                    market_regime=result.market_regime,
                    figure_influence=result.figure_influence,
                    contrarian_indicator=result.contrarian_indicator,
                    regime_adjusted_score=result.regime_adjusted_score,
                    total_data_points=result.total_data_points
                )
                data_points.append(data_point)
            
            # Calculate metadata
            has_more = len(results) == limit
            next_offset = offset + len(results) if has_more else None
            
            # Calculate data quality metrics
            avg_confidence = None
            if data_points:
                confidences = [dp.confidence for dp in data_points if dp.confidence is not None]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            
            query_duration_ms = (time.time() - start_time) * 1000
            
            # Determine actual date range from results
            actual_start = data_points[-1].timestamp if data_points else start_date
            actual_end = data_points[0].timestamp if data_points else end_date
            
            return SentimentHistoryResponse(
                symbols=[symbol],
                start_date=actual_start,
                end_date=actual_end,
                interval=interval,
                total_records=len(data_points),
                data_points=data_points,
                has_more=has_more,
                next_offset=next_offset,
                query_duration_ms=query_duration_ms,
                avg_confidence=avg_confidence
            )
            
    except Exception as e:
        logger.error(f"Error fetching historical sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment data: {str(e)}")

@app.post("/sentiment/historical", response_model=SentimentHistoryResponse)
async def get_sentiment_historical_multi(request: SentimentHistoryRequest):
    """
    Get historical sentiment data for multiple symbols with advanced filtering.
    
    Supports batch queries with pagination, confidence filtering, and regime filtering.
    """
    _check_timescale_availability()
    
    start_time = time.time()
    
    try:
        # Validate request
        if request.start_date >= request.end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        # Validate symbols list
        if not request.symbols:
            raise HTTPException(status_code=400, detail="At least one symbol must be provided")
        
        TimescaleSession = get_timescale_session()
        
        with TimescaleSession() as session:
            if request.interval == "raw":
                # Query raw sentiment data for multiple symbols
                query = select(SentimentData).where(
                    and_(
                        SentimentData.symbol.in_(request.symbols),
                        SentimentData.timestamp >= request.start_date,
                        SentimentData.timestamp <= request.end_date
                    )
                )
                
                if request.min_confidence is not None:
                    query = query.where(SentimentData.confidence >= request.min_confidence)
                
                if request.market_regime is not None:
                    query = query.where(SentimentData.market_regime == request.market_regime)
                
                query = query.order_by(
                    SentimentData.symbol,
                    SentimentData.timestamp.desc()
                ).offset(request.offset).limit(request.limit)
                
                results = session.execute(query).scalars().all()
                
            else:
                # Query aggregated data for multiple symbols
                query = select(SentimentAggregates).where(
                    and_(
                        SentimentAggregates.symbol.in_(request.symbols),
                        SentimentAggregates.interval_type == request.interval,
                        SentimentAggregates.time_bucket >= request.start_date,
                        SentimentAggregates.time_bucket <= request.end_date
                    )
                )
                
                query = query.order_by(
                    SentimentAggregates.symbol,
                    SentimentAggregates.time_bucket.desc()
                ).offset(request.offset).limit(request.limit)
                
                agg_results = session.execute(query).scalars().all()
                
                # Convert aggregated data to SentimentData format
                results = []
                for agg in agg_results:
                    results.append(SentimentData(
                        symbol=agg.symbol,
                        timestamp=agg.time_bucket,
                        overall_score=agg.avg_score,
                        confidence=agg.avg_confidence,
                        twitter_sentiment=agg.twitter_avg,
                        reddit_sentiment=agg.reddit_avg,
                        news_sentiment=agg.news_avg,
                        total_data_points=agg.data_points
                    ))
            
            # Convert to response format
            data_points = []
            symbols_in_response = set()
            
            for result in results:
                data_point = SentimentDataPoint(
                    symbol=result.symbol,
                    timestamp=result.timestamp,
                    overall_score=result.overall_score,
                    confidence=result.confidence,
                    social_momentum=result.social_momentum,
                    news_sentiment=result.news_sentiment,
                    twitter_sentiment=result.twitter_sentiment,
                    reddit_sentiment=result.reddit_sentiment,
                    source_counts=result.source_counts,
                    market_regime=result.market_regime,
                    figure_influence=result.figure_influence,
                    contrarian_indicator=result.contrarian_indicator,
                    regime_adjusted_score=result.regime_adjusted_score,
                    total_data_points=result.total_data_points
                )
                data_points.append(data_point)
                symbols_in_response.add(result.symbol)
            
            # Calculate metadata
            has_more = len(results) == request.limit
            next_offset = request.offset + len(results) if has_more else None
            
            # Calculate data quality metrics
            avg_confidence = None
            source_distribution = {}
            
            if data_points:
                confidences = [dp.confidence for dp in data_points if dp.confidence is not None]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                
                # Calculate source distribution
                for dp in data_points:
                    if dp.source_counts:
                        for source, count in dp.source_counts.items():
                            source_distribution[source] = source_distribution.get(source, 0) + count
            
            query_duration_ms = (time.time() - start_time) * 1000
            
            # Determine actual date range from results
            timestamps = [dp.timestamp for dp in data_points]
            actual_start = min(timestamps) if timestamps else request.start_date
            actual_end = max(timestamps) if timestamps else request.end_date
            
            return SentimentHistoryResponse(
                symbols=list(symbols_in_response),
                start_date=actual_start,
                end_date=actual_end,
                interval=request.interval,
                total_records=len(data_points),
                data_points=data_points,
                has_more=has_more,
                next_offset=next_offset,
                query_duration_ms=query_duration_ms,
                avg_confidence=avg_confidence,
                source_distribution=source_distribution
            )
            
    except Exception as e:
        logger.error(f"Error fetching multi-symbol historical sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment data: {str(e)}")

@app.get("/sentiment/aggregated", response_model=SentimentAggregateResponse)
async def get_sentiment_aggregated(
    symbols: List[str] = Query(..., description="Stock symbols to aggregate"),
    start_date: datetime = Query(..., description="Start date for aggregation"),
    end_date: datetime = Query(..., description="End date for aggregation"),
    aggregation: str = Query("1d", description="Aggregation interval", pattern="^(1h|6h|1d|1w)$"),
    include_stats: bool = Query(True, description="Include summary statistics")
):
    """
    Get aggregated sentiment data with optional summary statistics.
    
    Uses pre-computed aggregates for fast queries on large date ranges.
    """
    _check_timescale_availability()
    
    start_time = time.time()
    
    try:
        # Validate input
        if len(symbols) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed for aggregation")
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        TimescaleSession = get_timescale_session()
        
        with TimescaleSession() as session:
            # Query aggregated data
            query = select(SentimentAggregates).where(
                and_(
                    SentimentAggregates.symbol.in_(symbols),
                    SentimentAggregates.interval_type == aggregation,
                    SentimentAggregates.time_bucket >= start_date,
                    SentimentAggregates.time_bucket <= end_date
                )
            ).order_by(SentimentAggregates.symbol, SentimentAggregates.time_bucket)
            
            agg_results = session.execute(query).scalars().all()
            
            # Convert to response format
            aggregated_data = []
            raw_points_count = 0
            
            for agg in agg_results:
                data_point = {
                    "symbol": agg.symbol,
                    "timestamp": agg.time_bucket.isoformat(),
                    "avg_score": agg.avg_score,
                    "min_score": agg.min_score,
                    "max_score": agg.max_score,
                    "avg_confidence": agg.avg_confidence,
                    "data_points": agg.data_points,
                    "twitter_avg": agg.twitter_avg,
                    "reddit_avg": agg.reddit_avg,
                    "news_avg": agg.news_avg,
                    "score_stddev": agg.score_stddev,
                    "momentum_change": agg.momentum_change
                }
                aggregated_data.append(data_point)
                raw_points_count += agg.data_points
            
            # Calculate summary statistics if requested
            summary_stats = []
            if include_stats:
                for symbol in symbols:
                    symbol_data = [agg for agg in agg_results if agg.symbol == symbol]
                    
                    if symbol_data:
                        # Calculate stats for this symbol
                        scores = [agg.avg_score for agg in symbol_data]
                        confidences = [agg.avg_confidence for agg in symbol_data]
                        
                        stats = SentimentSummaryStats(
                            symbol=symbol,
                            period_start=start_date,
                            period_end=end_date,
                            total_data_points=sum(agg.data_points for agg in symbol_data),
                            avg_score=sum(scores) / len(scores),
                            min_score=min(agg.min_score for agg in symbol_data),
                            max_score=max(agg.max_score for agg in symbol_data),
                            avg_confidence=sum(confidences) / len(confidences),
                            score_volatility=None  # Could calculate from score_stddev if needed
                        )
                        summary_stats.append(stats)
            
            query_duration_ms = (time.time() - start_time) * 1000
            
            return SentimentAggregateResponse(
                symbols=symbols,
                aggregation_interval=aggregation,
                period_start=start_date,
                period_end=end_date,
                aggregated_data=aggregated_data,
                summary_stats=summary_stats if include_stats else None,
                total_raw_points=raw_points_count,
                query_duration_ms=query_duration_ms
            )
            
    except Exception as e:
        logger.error(f"Error fetching aggregated sentiment data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching aggregated data: {str(e)}")

@app.get("/sentiment/health", response_model=TimescaleHealthResponse)
async def get_timescale_health():
    """
    Comprehensive health check for TimescaleDB sentiment data system.
    
    Returns connection status, data metrics, and performance indicators.
    """
    start_time = time.time()
    check_timestamp = datetime.utcnow()
    
    if not TIMESCALE_AVAILABLE:
        return TimescaleHealthResponse(
            status="unavailable",
            connection_status="dependencies_missing",
            check_timestamp=check_timestamp,
            response_time_ms=(time.time() - start_time) * 1000,
            errors=["TimescaleDB dependencies not installed"]
        )
    
    if not is_timescale_available():
        return TimescaleHealthResponse(
            status="unavailable",
            connection_status="not_configured",
            check_timestamp=check_timestamp,
            response_time_ms=(time.time() - start_time) * 1000,
            warnings=["TimescaleDB not configured or connection failed"]
        )
    
    try:
        TimescaleSession = get_timescale_session()
        
        with TimescaleSession() as session:
            # Test basic connection
            version_result = session.execute(text("SELECT version()")).scalar()
            
            # Get TimescaleDB version
            try:
                timescale_version = session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
            except:
                timescale_version = None
            
            # Get connection count
            try:
                active_connections = session.execute(text("SELECT count(*) FROM pg_stat_activity")).scalar()
            except:
                active_connections = None
            
            # Get sentiment data metrics
            try:
                total_records = session.execute(select(func.count(SentimentData.symbol))).scalar()
                
                latest_data = session.execute(
                    select(func.max(SentimentData.timestamp))
                ).scalar()
                
                oldest_data = session.execute(
                    select(func.min(SentimentData.timestamp))
                ).scalar()
                
            except Exception as e:
                logger.warning(f"Could not get sentiment data metrics: {e}")
                total_records = None
                latest_data = None
                oldest_data = None
            
            # Get database size
            try:
                db_size_result = session.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))")).scalar()
            except:
                db_size_result = None
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine overall health status
            status = "healthy"
            warnings = []
            errors = []
            
            if response_time_ms > 5000:  # 5 seconds
                warnings.append("Slow response time detected")
                status = "degraded"
            
            if total_records == 0:
                warnings.append("No sentiment data found in database")
            
            if latest_data and (datetime.utcnow() - latest_data).days > 1:
                warnings.append("Latest sentiment data is more than 1 day old")
                status = "degraded"
            
            return TimescaleHealthResponse(
                status=status,
                connection_status="connected",
                database_version=version_result,
                timescale_version=timescale_version,
                active_connections=active_connections,
                total_sentiment_records=total_records,
                latest_data_timestamp=latest_data,
                oldest_data_timestamp=oldest_data,
                database_size=db_size_result,
                check_timestamp=check_timestamp,
                response_time_ms=response_time_ms,
                warnings=warnings if warnings else None,
                errors=errors if errors else None
            )
            
    except Exception as e:
        logger.error(f"TimescaleDB health check failed: {e}")
        return TimescaleHealthResponse(
            status="unhealthy",
            connection_status="connection_failed",
            check_timestamp=check_timestamp,
            response_time_ms=(time.time() - start_time) * 1000,
            errors=[f"Health check failed: {str(e)}"]
        )

# ================================
# CHAT API ENDPOINTS
# ================================

from datetime import timedelta
import uuid
from fastapi import WebSocket, WebSocketDisconnect, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
import json
import asyncio

# Add CORS middleware for WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple authentication scheme
security = HTTPBearer()

# In-memory chat sessions (use Redis in production)
chat_sessions = {}
active_websockets = {}

class ChatSessionManager:
    def __init__(self):
        self.sessions = {}
        self.connections = {}
    
    def create_session(self, user_id: str, profile_id: Optional[int] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "profile_id": profile_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "messages": [],
            "context": {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, message: Dict):
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append(message)
            self.sessions[session_id]["last_activity"] = datetime.now()
    
    def update_context(self, session_id: str, context: Dict):
        if session_id in self.sessions:
            self.sessions[session_id]["context"].update(context)
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        return [session for session in self.sessions.values() if session["user_id"] == user_id]

chat_manager = ChatSessionManager()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Simple user extraction - implement proper JWT validation in production
    return {"id": "demo_user", "email": "demo@example.com"}

@app.post("/chat")
async def chat_endpoint(
    message: str,
    session_id: Optional[str] = None,
    profile_id: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Main chat endpoint for user queries.
    Integrates with the conversational agent system for financial advice.
    """
    try:
        # Create or get session
        if not session_id:
            session_id = chat_manager.create_session(current_user["id"], profile_id)
        
        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Update context if provided
        if context:
            chat_manager.update_context(session_id, context)
        
        # Process the message
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"session_id": session_id}
        }
        
        chat_manager.add_message(session_id, user_message)
        
        # Generate AI response (integrate with your conversational agent)
        ai_response_content = await process_chat_message(
            message=message,
            session=session,
            profile_id=profile_id
        )
        
        ai_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant", 
            "content": ai_response_content["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "session_id": session_id,
                "confidence": ai_response_content.get("confidence", 0.8),
                "suggestions": ai_response_content.get("suggestions", []),
                "charts": ai_response_content.get("charts", [])
            }
        }
        
        chat_manager.add_message(session_id, ai_message)
        
        # Log the interaction
        log_event("chat_interaction", {
            "session_id": session_id,
            "user_id": current_user["id"], 
            "message_length": len(message),
            "response_length": len(ai_response_content["content"])
        })
        
        # Send to WebSocket clients if connected
        if session_id in active_websockets:
            for websocket in active_websockets[session_id]:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "message",
                        "data": ai_message
                    }))
                except:
                    pass  # Connection closed
        
        return {
            "session_id": session_id,
            "message": ai_message,
            "suggestions": ai_response_content.get("suggestions", []),
            "charts": ai_response_content.get("charts", [])
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

async def process_chat_message(message: str, session: Dict, profile_id: Optional[int] = None) -> Dict:
    """
    Process chat message and generate AI response.
    This integrates with your conversational agent system.
    """
    try:
        # Simple response logic - replace with your actual conversational agent
        context = session.get("context", {})
        message_lower = message.lower()
        
        # Financial advice queries
        if any(word in message_lower for word in ["advice", "recommend", "should i", "buy", "sell"]):
            if profile_id:
                # Get user profile for personalized advice
                try:
                    db = SessionLocal()
                    profile = db.get(Profile, profile_id)
                    risk_score = profile.risk_score if profile else 5
                    db.close()
                    
                    risk_level = "conservative" if risk_score <= 3 else "moderate" if risk_score <= 7 else "aggressive"
                    
                    return {
                        "content": f"Based on your {risk_level} risk profile, I'd recommend focusing on diversified investments. However, I need more specific information about what you're considering. Are you looking at a particular stock, sector, or investment strategy?",
                        "confidence": 0.85,
                        "suggestions": [
                            "Tell me about a specific stock symbol",
                            "Ask about portfolio diversification", 
                            "Request market analysis"
                        ]
                    }
                except:
                    pass
            
            return {
                "content": "I can help you with investment advice! To provide personalized recommendations, I'll need to know more about your situation. What specific investment or financial decision are you considering?",
                "confidence": 0.8,
                "suggestions": [
                    "Ask about a specific stock",
                    "Request portfolio review",
                    "Get market insights"
                ]
            }
        
        # Market data queries
        elif any(word in message_lower for word in ["price", "chart", "market", "stock", "crypto"]):
            symbols = extract_symbols_from_message(message)
            
            response_content = "I can help you analyze market data and stocks. "
            suggestions = ["Get technical analysis", "View price charts", "Check market sentiment"]
            charts = []
            
            if symbols:
                symbol = symbols[0].upper()
                response_content += f"Let me check the latest information for {symbol}."
                charts = [{"symbol": symbol, "type": "candlestick", "timeframe": "1d"}]
                suggestions = [
                    f"Get technical indicators for {symbol}",
                    f"View {symbol} price history",
                    f"Compare {symbol} with similar stocks"
                ]
            else:
                response_content += "Which stock symbol would you like me to analyze?"
            
            return {
                "content": response_content,
                "confidence": 0.9,
                "suggestions": suggestions,
                "charts": charts
            }
        
        # Backtesting queries
        elif any(word in message_lower for word in ["backtest", "strategy", "test", "performance"]):
            return {
                "content": "I can help you backtest trading strategies and analyze their historical performance. What strategy would you like to test, and on which timeframe?",
                "confidence": 0.85,
                "suggestions": [
                    "Test SMA crossover strategy",
                    "Analyze RSI strategy",
                    "Compare multiple strategies"
                ]
            }
        
        # General financial questions
        elif any(word in message_lower for word in ["portfolio", "diversification", "risk", "allocation"]):
            return {
                "content": "Portfolio management is crucial for long-term success. I can help you analyze your current allocation, assess risk levels, and suggest improvements. What aspect of your portfolio would you like to discuss?",
                "confidence": 0.8,
                "suggestions": [
                    "Analyze current portfolio",
                    "Assess risk tolerance", 
                    "Suggest rebalancing"
                ]
            }
        
        # Default response
        else:
            return {
                "content": "Hello! I'm your AI financial advisor. I can help you with investment advice, market analysis, backtesting strategies, and portfolio management. What would you like to explore today?",
                "confidence": 0.75,
                "suggestions": [
                    "Ask for investment advice",
                    "Analyze a stock symbol",
                    "Backtest a trading strategy",
                    "Review portfolio allocation"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return {
            "content": "I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
            "confidence": 0.5,
            "suggestions": ["Try rephrasing your question", "Ask about a specific topic"]
        }

def extract_symbols_from_message(message: str) -> List[str]:
    """Extract potential stock symbols from message"""
    import re
    # Simple regex to find potential stock symbols (2-5 uppercase letters)
    symbols = re.findall(r'\b[A-Z]{2,5}\b', message)
    # Filter out common false positives
    common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "ITS", "NEW", "NOW", "OLD", "SEE", "TWO", "WHO", "BOY", "DID", "MAX", "MIN", "SUM", "AVG"}
    return [s for s in symbols if s not in common_words]

@app.post("/chat/session")
async def create_chat_session(
    profile_id: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new chat session"""
    session_id = chat_manager.create_session(current_user["id"], profile_id)
    
    if context:
        chat_manager.update_context(session_id, context)
    
    log_event("chat_session_created", {"session_id": session_id, "user_id": current_user["id"]})
    
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "profile_id": profile_id
    }

@app.get("/chat/sessions")
async def get_chat_sessions(current_user: Dict = Depends(get_current_user)):
    """Get all chat sessions for the current user"""
    sessions = chat_manager.get_user_sessions(current_user["id"])
    
    # Return session metadata without full message history
    session_list = []
    for session in sessions:
        session_summary = {
            "session_id": session["id"],
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "message_count": len(session["messages"]),
            "profile_id": session.get("profile_id"),
            "preview": session["messages"][-1]["content"][:100] + "..." if session["messages"] else "New conversation"
        }
        session_list.append(session_summary)
    
    return {
        "sessions": session_list,
        "total_count": len(session_list)
    }

@app.get("/chat/session/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: Dict = Depends(get_current_user)
):
    """Get chat history for a specific session"""
    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify user owns this session
    if session["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    messages = session["messages"]
    total_messages = len(messages)
    
    # Apply pagination
    start_idx = max(0, total_messages - offset - limit)
    end_idx = total_messages - offset
    paginated_messages = messages[start_idx:end_idx] if start_idx < end_idx else []
    
    return {
        "session_id": session_id,
        "messages": paginated_messages,
        "total_messages": total_messages,
        "has_more": offset + len(paginated_messages) < total_messages,
        "context": session.get("context", {})
    }

@app.delete("/chat/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a chat session"""
    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify user owns this session
    if session["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Remove from sessions
    if session_id in chat_manager.sessions:
        del chat_manager.sessions[session_id]
    
    # Close any active websockets
    if session_id in active_websockets:
        for websocket in active_websockets[session_id]:
            try:
                await websocket.close()
            except:
                pass
        del active_websockets[session_id]
    
    log_event("chat_session_deleted", {"session_id": session_id, "user_id": current_user["id"]})
    
    return {"message": "Session deleted successfully"}

@app.post("/chat/feedback")
async def submit_chat_feedback(
    message_id: str,
    rating: int = Query(..., ge=1, le=5),
    feedback: Optional[str] = None,
    helpful: bool = True,
    current_user: Dict = Depends(get_current_user)
):
    """Submit feedback on a chat response"""
    feedback_data = {
        "message_id": message_id,
        "user_id": current_user["id"],
        "rating": rating,
        "feedback": feedback,
        "helpful": helpful,
        "timestamp": datetime.now().isoformat()
    }
    
    log_event("chat_feedback", feedback_data)
    
    return {"message": "Feedback submitted successfully", "feedback_id": str(uuid.uuid4())}

@app.get("/chat/suggestions")
async def get_chat_suggestions(
    profile_id: Optional[int] = None,
    category: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Get smart query suggestions based on user profile and recent activity"""
    suggestions = []
    
    # Base suggestions
    base_suggestions = [
        "What's the market outlook for this week?",
        "Analyze AAPL stock performance",
        "Show me top performing sectors",
        "Help me rebalance my portfolio",
        "What are the best dividend stocks?",
        "Explain market volatility"
    ]
    
    # Profile-based suggestions
    if profile_id:
        try:
            db = SessionLocal()
            profile = db.get(Profile, profile_id)
            if profile:
                risk_level = "conservative" if profile.risk_score <= 3 else "moderate" if profile.risk_score <= 7 else "aggressive"
                
                risk_suggestions = {
                    "conservative": [
                        "Find stable dividend stocks",
                        "Show me bond market analysis",
                        "What are safe haven assets?"
                    ],
                    "moderate": [
                        "Analyze S&P 500 trends", 
                        "Find balanced growth stocks",
                        "Compare ETF options"
                    ],
                    "aggressive": [
                        "Find high growth stocks",
                        "Analyze crypto market trends",
                        "Show me volatile trading opportunities"
                    ]
                }
                
                suggestions.extend(risk_suggestions.get(risk_level, []))
            db.close()
        except Exception as e:
            logger.error(f"Error getting profile-based suggestions: {e}")
    
    # Category-based suggestions
    category_suggestions = {
        "stocks": ["Analyze stock fundamentals", "Compare stock valuations", "Find undervalued stocks"],
        "crypto": ["Bitcoin price analysis", "Ethereum market trends", "DeFi opportunities"],
        "portfolio": ["Portfolio risk assessment", "Asset allocation review", "Rebalancing recommendations"],
        "strategy": ["Backtest SMA strategy", "Test momentum strategies", "Compare strategy performance"]
    }
    
    if category and category in category_suggestions:
        suggestions.extend(category_suggestions[category])
    else:
        suggestions.extend(base_suggestions)
    
    return {
        "suggestions": suggestions[:10],  # Limit to 10 suggestions
        "category": category,
        "personalized": profile_id is not None
    }

@app.post("/chat/context")
async def update_chat_context(
    session_id: str,
    context: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Update user context for a chat session"""
    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify user owns this session
    if session["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    chat_manager.update_context(session_id, context)
    
    log_event("chat_context_updated", {
        "session_id": session_id,
        "user_id": current_user["id"],
        "context_keys": list(context.keys())
    })
    
    return {"message": "Context updated successfully"}

# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat communication"""
    await websocket.accept()
    
    # Add to active connections
    if session_id not in active_websockets:
        active_websockets[session_id] = []
    active_websockets[session_id].append(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "chat":
                # Process chat message and broadcast response
                response = await process_chat_message(
                    message=message_data["content"],
                    session=chat_manager.get_session(session_id) or {},
                    profile_id=message_data.get("profile_id")
                )
                
                # Broadcast to all connections in this session
                for ws in active_websockets[session_id]:
                    try:
                        await ws.send_text(json.dumps({
                            "type": "message",
                            "data": {
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "content": response["content"],
                                "timestamp": datetime.now().isoformat(),
                                "metadata": {
                                    "confidence": response.get("confidence", 0.8),
                                    "suggestions": response.get("suggestions", [])
                                }
                            }
                        }))
                    except:
                        pass
                        
            elif message_data.get("type") == "typing":
                # Broadcast typing indicator
                for ws in active_websockets[session_id]:
                    if ws != websocket:  # Don't send back to sender
                        try:
                            await ws.send_text(json.dumps({
                                "type": "typing",
                                "user_id": message_data.get("user_id", "unknown")
                            }))
                        except:
                            pass
                            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove from active connections
        if session_id in active_websockets:
            try:
                active_websockets[session_id].remove(websocket)
                if not active_websockets[session_id]:
                    del active_websockets[session_id]
            except:
                pass

# Chat UI endpoint
@app.get("/chat/ui", response_class=HTMLResponse)
async def chat_ui_endpoint(
    session_id: Optional[str] = Query(None),
    profile_id: Optional[int] = Query(None),
    theme: str = Query("dark")
):
    """Serve the chat UI interface"""
    from .chat_ui import get_chat_interface_html
    
    html_content = get_chat_interface_html(
        session_id=session_id,
        profile_id=profile_id,
        theme=theme,
        api_base_url="/api",  # Adjust based on your API path
        websocket_url="/ws/chat"
    )
    
    return HTMLResponse(content=html_content)

# ================================
# OPTIMIZATION API ENDPOINTS
# ================================

# Import optimization components with fallback
try:
    from .optimizer import OptunaOptimizer, OptimizationConfig, OPTUNA_AVAILABLE
    from .optimization_models import (
        OptimizationStudyModel, OptimizationTrialModel, WatchlistOptimizationModel,
        get_study_statistics
    )
    from .workflows import (
        optimize_strategy_parameters_flow, optimize_watchlist_strategies_flow,
        scheduled_optimization_flow, OPTIMIZATION_AVAILABLE
    )
    from .schemas import (
        OptimizationRequest, OptimizationStudy, OptimizationTrial, OptimizationResults,
        OptimizationProgress, WatchlistOptimizationRequest, WatchlistOptimizationResponse,
        OptimizationDashboardData, OptimizationComparison
    )
    OPTIMIZATION_ENDPOINTS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_ENDPOINTS_AVAILABLE = False

def _check_optimization_availability():
    """Helper to check optimization availability and raise appropriate errors."""
    if not OPTIMIZATION_ENDPOINTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Optimization features not available. Missing dependencies or models."
        )
    
    if not OPTUNA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Optuna not available. Install with: pip install optuna>=3.4.0"
        )

@app.post("/optimize/strategy", response_model=OptimizationResults)
async def optimize_strategy(req: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start strategy parameter optimization for a single symbol.
    
    Creates an optimization study and runs parameter search using Optuna.
    For long-running optimizations, results are processed in background.
    """
    _check_optimization_availability()
    
    try:
        # Generate unique study name if not provided
        study_name = req.study_name or f"{req.symbol}_{req.strategy_template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate request
        if len(req.bars) < 100:
            raise HTTPException(status_code=400, detail="Insufficient historical data. Minimum 100 bars required.")
        
        # For quick optimizations (< 50 trials), run synchronously
        if req.n_trials <= 50:
            logger.info(f"Running synchronous optimization for {req.symbol} with {req.n_trials} trials")
            
            try:
                # Run optimization flow
                result = await optimize_strategy_parameters_flow(
                    symbol=req.symbol,
                    strategy_template=req.strategy_template,
                    n_trials=req.n_trials,
                    lookback_days=365,  # Use all provided data
                    timeframe="1h",
                    walk_forward=req.walk_forward,
                    save_results=True
                )
                
                if not result.get("success", False):
                    raise HTTPException(status_code=400, detail=f"Optimization failed: {result.get('error', 'Unknown error')}")
                
                opt_result = result["optimization"]
                
                # Create response
                optimization_results = OptimizationResults(
                    study_id=study_name,
                    study_name=study_name,
                    symbol=req.symbol,
                    strategy_template=req.strategy_template,
                    best_params=opt_result["best_params"],
                    best_value=opt_result["best_value"],
                    best_trial_number=opt_result["best_trial"],
                    best_metrics=opt_result["metrics"],
                    validation_metrics=opt_result.get("validation_metrics"),
                    total_trials=opt_result["n_trials"],
                    completed_trials=opt_result.get("study_stats", {}).get("n_complete_trials", opt_result["n_trials"]),
                    pruned_trials=opt_result.get("study_stats", {}).get("n_pruned_trials", 0),
                    failed_trials=opt_result.get("study_stats", {}).get("n_failed_trials", 0),
                    optimization_duration=opt_result["optimization_duration"],
                    trials_per_minute=opt_result["n_trials"] / (opt_result["optimization_duration"] / 60) if opt_result["optimization_duration"] > 0 else 0,
                    completed_at=datetime.fromisoformat(opt_result["timestamp"]),
                    configuration={
                        "n_trials": req.n_trials,
                        "primary_objective": req.primary_objective,
                        "walk_forward": req.walk_forward,
                        "include_transaction_costs": req.include_transaction_costs
                    }
                )
                
                log_event("optimization_completed", {
                    "symbol": req.symbol,
                    "strategy": req.strategy_template,
                    "best_value": opt_result["best_value"],
                    "n_trials": req.n_trials
                })
                
                append_session(
                    supervisor="AutoLogger",
                    subagents=["OptimizationAgent"],
                    actions=[f"Optimized {req.strategy_template} for {req.symbol}"],
                    results=[f"Best Sharpe: {opt_result['best_value']:.3f}", f"Trials: {req.n_trials}"],
                    suggestions=["Consider walk-forward validation", "Test on out-of-sample data"],
                    meta={"endpoint": "/optimize/strategy", "symbol": req.symbol, "study_name": study_name}
                )
                
                return optimization_results
                
            except Exception as e:
                logger.error(f"Synchronous optimization failed: {e}")
                raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
        
        else:
            # For long optimizations (>= 50 trials), run in background
            logger.info(f"Running background optimization for {req.symbol} with {req.n_trials} trials")
            
            # Add to background tasks
            background_tasks.add_task(
                _run_background_optimization,
                req, study_name
            )
            
            # Return immediate response with study info
            return OptimizationResults(
                study_id=study_name,
                study_name=study_name,
                symbol=req.symbol,
                strategy_template=req.strategy_template,
                best_params={},
                best_value=0.0,
                best_trial_number=0,
                best_metrics={},
                total_trials=req.n_trials,
                completed_trials=0,
                pruned_trials=0,
                failed_trials=0,
                optimization_duration=0.0,
                trials_per_minute=0.0,
                completed_at=datetime.now(),
                configuration={
                    "n_trials": req.n_trials,
                    "status": "running_in_background",
                    "estimated_duration_minutes": req.n_trials * 0.5  # Rough estimate
                }
            )
    
    except Exception as e:
        logger.error(f"Optimization request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

async def _run_background_optimization(req: OptimizationRequest, study_name: str):
    """Run optimization in background task"""
    try:
        result = await optimize_strategy_parameters_flow(
            symbol=req.symbol,
            strategy_template=req.strategy_template,
            n_trials=req.n_trials,
            lookback_days=365,
            timeframe="1h",
            walk_forward=req.walk_forward,
            save_results=True
        )
        
        logger.info(f"Background optimization completed for {req.symbol}: {result.get('success', False)}")
        
    except Exception as e:
        logger.error(f"Background optimization failed for {req.symbol}: {e}")

@app.get("/optimize/studies", response_model=List[OptimizationStudy])
def get_optimization_studies(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_template: Optional[str] = Query(None, description="Filter by strategy template"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of studies to return"),
    offset: int = Query(0, ge=0, description="Number of studies to skip")
):
    """
    Get list of optimization studies with optional filtering.
    
    Returns paginated list of studies with metadata and status information.
    """
    _check_optimization_availability()
    
    db = SessionLocal()
    try:
        # Build query
        query = db.query(OptimizationStudyModel)
        
        if symbol:
            query = query.filter(OptimizationStudyModel.symbol == symbol)
        if strategy_template:
            query = query.filter(OptimizationStudyModel.strategy_template == strategy_template)
        if status:
            query = query.filter(OptimizationStudyModel.status == status)
        
        # Apply pagination and ordering
        query = query.order_by(OptimizationStudyModel.created_at.desc())
        query = query.offset(offset).limit(limit)
        
        studies = query.all()
        
        # Convert to response model
        study_list = []
        for study in studies:
            study_data = OptimizationStudy(
                study_id=study.study_id,
                study_name=study.study_name,
                symbol=study.symbol,
                strategy_template=study.strategy_template,
                n_trials=study.n_trials,
                primary_objective=study.primary_objective,
                secondary_objectives=study.secondary_objectives or [],
                status=study.status.value,
                progress=study.progress,
                n_complete_trials=study.n_complete_trials,
                n_pruned_trials=study.n_pruned_trials,
                n_failed_trials=study.n_failed_trials,
                best_value=study.best_value,
                best_params=study.best_params,
                best_trial_number=study.best_trial_number,
                created_at=study.created_at,
                started_at=study.started_at,
                completed_at=study.completed_at,
                estimated_completion=study.estimated_completion,
                user_attributes=study.user_attributes or {},
                system_attributes=study.system_attributes or {}
            )
            study_list.append(study_data)
        
        log_event("studies_queried", {"count": len(study_list), "filters": {"symbol": symbol, "strategy": strategy_template}})
        
        return study_list
        
    except Exception as e:
        logger.error(f"Failed to get optimization studies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve studies: {str(e)}")
    finally:
        db.close()

@app.get("/optimize/study/{study_id}", response_model=OptimizationResults)
def get_optimization_study(study_id: str):
    """
    Get detailed information about a specific optimization study.
    
    Returns comprehensive study results including best parameters,
    performance metrics, and trial statistics.
    """
    _check_optimization_availability()
    
    db = SessionLocal()
    try:
        # Get study details
        study = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.study_id == study_id
        ).first()
        
        if not study:
            raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
        
        # Get comprehensive statistics
        stats = get_study_statistics(study_id, db)
        if not stats:
            raise HTTPException(status_code=500, detail="Failed to retrieve study statistics")
        
        # Create detailed response
        optimization_results = OptimizationResults(
            study_id=study.study_id,
            study_name=study.study_name,
            symbol=study.symbol,
            strategy_template=study.strategy_template,
            best_params=study.best_params or {},
            best_value=study.best_value or 0.0,
            best_trial_number=study.best_trial_number or 0,
            best_metrics=study.best_metrics or {},
            validation_metrics=None,  # Could be added from trials if stored
            total_trials=study.n_trials,
            completed_trials=study.n_complete_trials,
            pruned_trials=study.n_pruned_trials,
            failed_trials=study.n_failed_trials,
            parameter_importance=None,  # Could be computed if needed
            convergence_info=stats.get("recent_performance"),
            optimization_duration=study.duration_seconds or 0.0,
            trials_per_minute=study.n_complete_trials / (study.duration_seconds / 60) if study.duration_seconds and study.duration_seconds > 0 else 0,
            completed_at=study.completed_at or study.created_at,
            configuration=study.configuration or {}
        )
        
        return optimization_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get study {study_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve study: {str(e)}")
    finally:
        db.close()

@app.get("/optimize/best-params/{study_id}")
def get_best_parameters(study_id: str):
    """
    Get best parameters from completed optimization study.
    
    Returns the parameter combination that achieved the best objective value.
    """
    _check_optimization_availability()
    
    db = SessionLocal()
    try:
        study = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.study_id == study_id
        ).first()
        
        if not study:
            raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
        
        if not study.best_params:
            raise HTTPException(status_code=400, detail=f"Study '{study_id}' has no best parameters yet")
        
        return {
            "study_id": study_id,
            "symbol": study.symbol,
            "strategy_template": study.strategy_template,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial_number": study.best_trial_number,
            "best_metrics": study.best_metrics,
            "optimization_completed": study.completed_at.isoformat() if study.completed_at else None,
            "ready_for_deployment": study.status.value == "completed" and study.best_value is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get best parameters for {study_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve best parameters: {str(e)}")
    finally:
        db.close()

@app.post("/optimize/watchlist", response_model=WatchlistOptimizationResponse)
async def optimize_watchlist(req: WatchlistOptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start batch optimization for multiple symbols and strategies (watchlist).
    
    Optimizes strategies across a user's watchlist or specified symbols.
    Returns immediately and runs optimization in background due to long duration.
    """
    _check_optimization_availability()
    
    try:
        # Get symbols from watchlist if not specified
        symbols = req.symbols
        if not symbols:
            db = SessionLocal()
            try:
                from .db import WatchlistItem, Asset
                
                watchlist_items = db.query(WatchlistItem).join(Asset).filter(
                    WatchlistItem.profile_id == req.profile_id,
                    WatchlistItem.active == True
                ).all()
                
                symbols = [item.asset.symbol for item in watchlist_items]
                
                if not symbols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No active symbols found in watchlist for profile {req.profile_id}"
                    )
            finally:
                db.close()
        
        # Generate unique optimization ID
        optimization_id = f"watchlist_{req.profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate estimated duration
        total_combinations = len(symbols) * len(req.strategy_templates)
        estimated_duration_hours = (total_combinations * req.n_trials_per_symbol * 0.5) / (60 * max(1, req.parallel_jobs))
        estimated_completion = datetime.now() + timedelta(hours=estimated_duration_hours)
        
        # Create initial response
        response = WatchlistOptimizationResponse(
            optimization_id=optimization_id,
            profile_id=req.profile_id,
            symbols=symbols,
            strategy_templates=req.strategy_templates,
            status="queued",
            progress=0.0,
            study_statuses={},
            estimated_duration_hours=estimated_duration_hours,
            estimated_completion=estimated_completion,
            parallel_jobs_used=req.parallel_jobs,
            priority_level=req.priority,
            created_at=datetime.now(),
            scheduled_for=None if not req.schedule_overnight else datetime.now().replace(hour=2, minute=0, second=0) + timedelta(days=1)
        )
        
        # Add optimization to background tasks
        if req.schedule_overnight:
            # Schedule for overnight execution (implement scheduling logic here)
            logger.info(f"Scheduled watchlist optimization {optimization_id} for overnight execution")
            # In production, you'd use a proper scheduler like Celery or Prefect
            background_tasks.add_task(_run_scheduled_watchlist_optimization, req, optimization_id)
        else:
            # Start immediately in background
            logger.info(f"Starting immediate watchlist optimization {optimization_id}")
            background_tasks.add_task(_run_watchlist_optimization, req, optimization_id)
        
        log_event("watchlist_optimization_started", {
            "optimization_id": optimization_id,
            "profile_id": req.profile_id,
            "symbols_count": len(symbols),
            "strategies_count": len(req.strategy_templates),
            "scheduled": req.schedule_overnight
        })
        
        append_session(
            supervisor="AutoLogger",
            subagents=["WatchlistOptimizationAgent"],
            actions=[f"Started batch optimization for {len(symbols)} symbols"],
            results=[f"optimization_id={optimization_id}", f"estimated_hours={estimated_duration_hours:.1f}"],
            suggestions=["Monitor optimization progress", "Set up completion notifications"],
            meta={"endpoint": "/optimize/watchlist", "optimization_id": optimization_id}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Watchlist optimization request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start watchlist optimization: {str(e)}")

async def _run_watchlist_optimization(req: WatchlistOptimizationRequest, optimization_id: str):
    """Run watchlist optimization in background"""
    try:
        symbols = req.symbols or []  # Should be populated by the calling function
        
        result = await optimize_watchlist_strategies_flow(
            symbols=symbols,
            strategy_templates=req.strategy_templates,
            profile_id=req.profile_id,
            n_trials_per_symbol=req.n_trials_per_symbol,
            parallel_jobs=req.parallel_jobs,
            lookback_days=req.lookback_days,
            timeframe=req.data_timeframe
        )
        
        logger.info(f"Watchlist optimization {optimization_id} completed: {result.get('success', False)}")
        
        # Could save detailed results to database here
        
    except Exception as e:
        logger.error(f"Watchlist optimization {optimization_id} failed: {e}")

async def _run_scheduled_watchlist_optimization(req: WatchlistOptimizationRequest, optimization_id: str):
    """Run scheduled watchlist optimization"""
    try:
        # Wait until scheduled time (simplified - in production use proper scheduling)
        import asyncio
        await asyncio.sleep(60)  # Placeholder
        
        result = await scheduled_optimization_flow(
            profile_id=req.profile_id,
            schedule_hour=2,
            max_symbols=len(req.symbols) if req.symbols else 20,
            n_trials_per_symbol=req.n_trials_per_symbol
        )
        
        logger.info(f"Scheduled optimization {optimization_id} completed: {result.get('success', False)}")
        
    except Exception as e:
        logger.error(f"Scheduled optimization {optimization_id} failed: {e}")

@app.get("/optimize/dashboard", response_model=OptimizationDashboardData)
def get_optimization_dashboard():
    """
    Get optimization dashboard data with summary statistics and insights.
    
    Provides overview of all optimization activities, performance trends,
    and resource utilization for management dashboard.
    """
    _check_optimization_availability()
    
    db = SessionLocal()
    try:
        # Get summary statistics
        total_studies = db.query(OptimizationStudyModel).count()
        active_studies = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.status == "running"
        ).count()
        completed_studies = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.status == "completed"
        ).count()
        
        # Get recent completions
        recent_completed = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.status == "completed",
            OptimizationStudyModel.completed_at >= datetime.now() - timedelta(days=7)
        ).order_by(OptimizationStudyModel.completed_at.desc()).limit(10).all()
        
        recent_completions = []
        for study in recent_completed:
            if study.best_value is not None:
                completion = OptimizationResults(
                    study_id=study.study_id,
                    study_name=study.study_name,
                    symbol=study.symbol,
                    strategy_template=study.strategy_template,
                    best_params=study.best_params or {},
                    best_value=study.best_value,
                    best_trial_number=study.best_trial_number or 0,
                    best_metrics=study.best_metrics or {},
                    total_trials=study.n_trials,
                    completed_trials=study.n_complete_trials,
                    pruned_trials=study.n_pruned_trials,
                    failed_trials=study.n_failed_trials,
                    optimization_duration=study.duration_seconds or 0.0,
                    trials_per_minute=0.0,
                    completed_at=study.completed_at or study.created_at,
                    configuration=study.configuration or {}
                )
                recent_completions.append(completion)
        
        # Get top performers
        top_performers_query = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.status == "completed",
            OptimizationStudyModel.best_value.isnot(None)
        ).order_by(OptimizationStudyModel.best_value.desc()).limit(10).all()
        
        top_performers = []
        for study in top_performers_query:
            performer = {
                "symbol": study.symbol,
                "strategy_template": study.strategy_template,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "completed_at": study.completed_at.isoformat() if study.completed_at else None,
                "study_id": study.study_id
            }
            top_performers.append(performer)
        
        # Strategy popularity
        from sqlalchemy import func
        strategy_counts = db.query(
            OptimizationStudyModel.strategy_template,
            func.count(OptimizationStudyModel.id).label('count')
        ).group_by(
            OptimizationStudyModel.strategy_template
        ).all()
        
        popular_strategies = {row.strategy_template: row.count for row in strategy_counts}
        
        # Create dashboard data
        dashboard_data = OptimizationDashboardData(
            total_studies=total_studies,
            active_studies=active_studies,
            completed_studies=completed_studies,
            recent_completions=recent_completions,
            active_optimizations=[],  # Could add real-time progress here
            top_performers=top_performers,
            parameter_insights={},  # Could add parameter importance analysis
            compute_usage={"cpu_percent": 0.0, "memory_percent": 0.0},  # Placeholder
            storage_usage={"studies_mb": 0.0, "trials_mb": 0.0},  # Placeholder  
            optimization_trends={"daily_completions": []},  # Could add trend analysis
            popular_strategies=popular_strategies,
            generated_at=datetime.now(),
            refresh_interval=30
        )
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to generate optimization dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")
    finally:
        db.close()

@app.get("/optimize/progress/{study_id}", response_model=OptimizationProgress)
def get_optimization_progress(study_id: str):
    """
    Get real-time optimization progress for a running study.
    
    Returns current trial number, best value found, and estimated completion time.
    """
    _check_optimization_availability()
    
    db = SessionLocal()
    try:
        study = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.study_id == study_id
        ).first()
        
        if not study:
            raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
        
        # Get recent trials
        recent_trials_query = db.query(OptimizationTrialModel).filter(
            OptimizationTrialModel.study_id == study_id
        ).order_by(OptimizationTrialModel.datetime_complete.desc()).limit(5)
        
        recent_trials_data = []
        for trial in recent_trials_query.all():
            trial_data = OptimizationTrial(
                trial_id=trial.trial_id,
                study_id=trial.study_id,
                params=trial.params or {},
                value=trial.value,
                values=trial.values,
                state=trial.state.value,
                datetime_start=trial.datetime_start,
                datetime_complete=trial.datetime_complete,
                duration=trial.duration_seconds,
                metrics=trial.metrics,
                user_attrs=trial.user_attrs or {},
                system_attrs=trial.system_attrs or {}
            )
            recent_trials_data.append(trial_data)
        
        # Calculate progress metrics
        total_trials_completed = study.n_complete_trials + study.n_pruned_trials + study.n_failed_trials
        progress_percentage = min(100.0, (total_trials_completed / study.n_trials) * 100.0)
        
        # Estimate remaining time based on completed trials
        estimated_remaining = None
        if study.started_at and total_trials_completed > 0:
            elapsed_time = (datetime.utcnow() - study.started_at).total_seconds()
            avg_time_per_trial = elapsed_time / total_trials_completed
            remaining_trials = study.n_trials - total_trials_completed
            estimated_remaining = remaining_trials * avg_time_per_trial
        
        # Determine status
        if study.status.value == "completed":
            status = "completed"
        elif study.status.value == "failed":
            status = "failed"
        elif study.status.value == "running":
            status = "running"
        else:
            status = "initializing"
        
        progress = OptimizationProgress(
            study_id=study_id,
            current_trial=total_trials_completed,
            total_trials=study.n_trials,
            progress_percentage=progress_percentage,
            elapsed_time=(datetime.utcnow() - study.started_at).total_seconds() if study.started_at else 0,
            estimated_remaining=estimated_remaining,
            current_best_value=study.best_value,
            current_best_params=study.best_params,
            current_best_trial=study.best_trial_number,
            recent_trials=recent_trials_data,
            improvement_streak=0,  # Could calculate from recent trials
            status=status,
            message=f"Completed {total_trials_completed}/{study.n_trials} trials",
            trials_per_minute=0.0,  # Could calculate from timing data
            updated_at=datetime.now()
        )
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress for {study_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization progress: {str(e)}")
    finally:
        db.close()

@app.delete("/optimize/study/{study_id}")
def delete_optimization_study(study_id: str):
    """
    Delete an optimization study and all associated trials.
    
    Warning: This action cannot be undone.
    """
    _check_optimization_availability()
    
    db = SessionLocal()
    try:
        study = db.query(OptimizationStudyModel).filter(
            OptimizationStudyModel.study_id == study_id
        ).first()
        
        if not study:
            raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
        
        # Check if study is still running
        if study.status.value == "running":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete running study '{study_id}'. Stop the study first."
            )
        
        # Delete the study (trials will be deleted due to cascade)
        db.delete(study)
        db.commit()
        
        log_event("study_deleted", {"study_id": study_id, "symbol": study.symbol})
        
        return {"message": f"Study '{study_id}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete study {study_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete study: {str(e)}")
    finally:
        db.close()

# ====================================
# HEALTH CHECK ENDPOINTS  
# ====================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint - overall system status."""
    try:
        from .health_monitor import get_health_status
        health_data = await get_health_status()
        
        status_code = 200
        if health_data["overall_status"] == "critical":
            status_code = 503  # Service Unavailable
        elif health_data["overall_status"] == "warning":
            status_code = 200  # OK but degraded
            
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "overall_status": "critical",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Kubernetes liveness check - is the service running?"""
    try:
        from .health_monitor import get_liveness_status
        return await get_liveness_status()
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "dead",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes readiness check - is the service ready to accept traffic?"""
    try:
        from .health_monitor import get_readiness_status
        readiness_data = await get_readiness_status()
        
        status_code = 200 if readiness_data["ready"] else 503
        
        return JSONResponse(
            status_code=status_code,
            content=readiness_data
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health/deep", tags=["Health"])
async def deep_health_check():
    """Detailed health check with component breakdown and diagnostics."""
    try:
        from .health_monitor import get_detailed_health_status
        return await get_detailed_health_status()
    except Exception as e:
        logger.error(f"Deep health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Deep health check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health/metrics", tags=["Health"])
async def health_metrics():
    """Health and performance metrics summary."""
    try:
        from .health_monitor import get_metrics_summary
        return await get_metrics_summary()
    except Exception as e:
        logger.error(f"Health metrics check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Health metrics failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health/dependencies", tags=["Health"])
async def dependency_status():
    """External service and dependency status check."""
    try:
        from .health_monitor import health_monitor
        
        # Check specific dependencies
        dependency_checkers = ["database", "llm_service", "cache", "external_services"]
        dependency_results = {}
        
        for component in dependency_checkers:
            if component in health_monitor.health_checkers:
                checker = health_monitor.health_checkers[component]
                result = await checker.check_health()
                dependency_results[component] = {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "details": result.details
                }
        
        # Determine overall dependency status
        overall_healthy = all(
            result["status"] == "healthy" 
            for result in dependency_results.values()
        )
        
        return {
            "dependencies_healthy": overall_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": dependency_results
        }
        
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Dependency check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/health/self-heal", tags=["Health"])
async def trigger_self_healing(current_user: User = Depends(get_current_user)):
    """Trigger automated self-healing processes."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions for self-healing")
    
    try:
        from .health_monitor import health_monitor
        healing_actions = await health_monitor.perform_self_healing()
        
        return {
            "self_healing_triggered": True,
            "actions_performed": healing_actions,
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_by": current_user.email
        }
        
    except Exception as e:
        logger.error(f"Self-healing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-healing failed: {str(e)}")

@app.get("/health/alerts", tags=["Health"])
async def active_alerts(current_user: User = Depends(get_current_user)):
    """Get active health alerts."""
    try:
        from .monitoring_alerts import alert_manager
        from .health_monitor import health_monitor
        
        # Combine health monitor alerts and alert manager alerts
        health_alerts = health_monitor.get_active_alerts()
        system_alerts = alert_manager.get_active_alerts()
        
        return {
            "health_alerts": health_alerts,
            "system_alerts": system_alerts,
            "total_active_alerts": len(health_alerts) + len(system_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")

@app.post("/health/alerts/{alert_id}/acknowledge", tags=["Health"])
async def acknowledge_alert(alert_id: str, current_user: User = Depends(get_current_user)):
    """Acknowledge a health alert."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to acknowledge alerts")
    
    try:
        from .monitoring_alerts import alert_manager
        success = await alert_manager.acknowledge_alert(alert_id, current_user.email)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "alert_acknowledged": True,
            "alert_id": alert_id,
            "acknowledged_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert acknowledgment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert acknowledgment failed: {str(e)}")

@app.post("/health/alerts/{alert_id}/resolve", tags=["Health"])
async def resolve_alert(alert_id: str, current_user: User = Depends(get_current_user)):
    """Resolve a health alert."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to resolve alerts")
    
    try:
        from .monitoring_alerts import alert_manager
        success = await alert_manager.resolve_alert(alert_id, current_user.email)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "alert_resolved": True,
            "alert_id": alert_id,
            "resolved_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert resolution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert resolution failed: {str(e)}")

@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    try:
        from .performance_monitor import performance_collector
        from .monitoring_alerts import alert_manager
        
        # Get system metrics
        metrics_output = []
        
        # Health check metrics
        try:
            from .health_monitor import health_monitor
            if health_monitor.health_history:
                latest_health = health_monitor.health_history[-1]
                
                metrics_output.append(f"# HELP supernova_health_score Overall system health score (0-100)")
                metrics_output.append(f"# TYPE supernova_health_score gauge")
                metrics_output.append(f"supernova_health_score {latest_health.overall_score}")
                
                metrics_output.append(f"# HELP supernova_components_healthy Number of healthy components")
                metrics_output.append(f"# TYPE supernova_components_healthy gauge")
                healthy_count = len([c for c in latest_health.components.values() if c.status.value == "healthy"])
                metrics_output.append(f"supernova_components_healthy {healthy_count}")
                
        except Exception as e:
            logger.error(f"Failed to collect health metrics: {e}")
        
        # Performance metrics
        try:
            if performance_collector:
                resource_stats = performance_collector.get_resource_stats()
                if resource_stats:
                    metrics_output.append(f"# HELP supernova_cpu_percent CPU usage percentage")
                    metrics_output.append(f"# TYPE supernova_cpu_percent gauge")
                    metrics_output.append(f"supernova_cpu_percent {resource_stats.get('avg_cpu_percent', 0)}")
                    
                    metrics_output.append(f"# HELP supernova_memory_percent Memory usage percentage")
                    metrics_output.append(f"# TYPE supernova_memory_percent gauge")
                    metrics_output.append(f"supernova_memory_percent {resource_stats.get('avg_memory_percent', 0)}")
                    
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
        
        # Alert metrics
        try:
            alert_stats = alert_manager.get_alert_stats()
            
            metrics_output.append(f"# HELP supernova_active_alerts Number of active alerts")
            metrics_output.append(f"# TYPE supernova_active_alerts gauge")
            metrics_output.append(f"supernova_active_alerts {alert_stats['active_alerts']['total']}")
            
            for level, count in alert_stats['active_alerts']['by_level'].items():
                metrics_output.append(f"supernova_active_alerts{{level=\"{level.lower()}\"}} {count}")
                
        except Exception as e:
            logger.error(f"Failed to collect alert metrics: {e}")
        
        # Uptime metric
        try:
            from .health_monitor import health_monitor
            uptime_seconds = (datetime.utcnow() - health_monitor.start_time).total_seconds()
            
            metrics_output.append(f"# HELP supernova_uptime_seconds Service uptime in seconds")
            metrics_output.append(f"# TYPE supernova_uptime_seconds counter")
            metrics_output.append(f"supernova_uptime_seconds {uptime_seconds}")
            
        except Exception as e:
            logger.error(f"Failed to collect uptime metrics: {e}")
        
        return Response(
            content="\n".join(metrics_output) + "\n",
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return Response(
            content=f"# Error collecting metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )

# ====================================
# MONITORING DASHBOARD ENDPOINTS
# ====================================

@app.get("/dashboard", tags=["Monitoring"])
async def dashboard_list():
    """Get list of available monitoring dashboards."""
    try:
        from .monitoring_dashboard import dashboard_manager
        return dashboard_manager.get_dashboard_list()
    except Exception as e:
        logger.error(f"Dashboard list failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard list failed: {str(e)}")

@app.get("/dashboard/{dashboard_id}", tags=["Monitoring"])
async def get_dashboard_html(dashboard_id: str):
    """Get dashboard HTML page."""
    try:
        from .monitoring_dashboard import dashboard_manager
        html_content = dashboard_manager.generate_dashboard_html(dashboard_id)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Dashboard HTML generation failed: {e}")
        return HTMLResponse(
            content="<html><body><h1>Dashboard Error</h1><p>Failed to load dashboard</p></body></html>",
            status_code=500
        )

@app.get("/api/dashboards/{dashboard_id}/data", tags=["Monitoring"])
async def get_dashboard_data(dashboard_id: str, time_range: str = "1h"):
    """Get dashboard data for rendering."""
    try:
        from .monitoring_dashboard import dashboard_manager
        return await dashboard_manager.get_dashboard_data(dashboard_id, time_range)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

@app.post("/api/dashboards", tags=["Monitoring"])
async def create_custom_dashboard(
    dashboard_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create a custom monitoring dashboard."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to create dashboards")
    
    try:
        from .monitoring_dashboard import dashboard_manager
        
        name = dashboard_config.get("name")
        description = dashboard_config.get("description", "")
        widgets = dashboard_config.get("widgets", [])
        
        if not name:
            raise HTTPException(status_code=400, detail="Dashboard name is required")
        
        dashboard_id = dashboard_manager.create_custom_dashboard(
            name=name,
            description=description,
            widgets=widgets,
            user=current_user.email
        )
        
        return {
            "dashboard_id": dashboard_id,
            "name": name,
            "created_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard creation failed: {str(e)}")

@app.put("/api/dashboards/{dashboard_id}", tags=["Monitoring"])
async def update_dashboard(
    dashboard_id: str,
    dashboard_updates: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update a monitoring dashboard."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to update dashboards")
    
    try:
        from .monitoring_dashboard import dashboard_manager
        
        success = dashboard_manager.update_dashboard(dashboard_id, dashboard_updates, current_user.email)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found or cannot be updated")
        
        return {
            "dashboard_id": dashboard_id,
            "updated": True,
            "updated_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard update failed: {str(e)}")

@app.delete("/api/dashboards/{dashboard_id}", tags=["Monitoring"])
async def delete_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a custom monitoring dashboard."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to delete dashboards")
    
    try:
        from .monitoring_dashboard import dashboard_manager
        
        success = dashboard_manager.delete_dashboard(dashboard_id, current_user.email)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found or cannot be deleted")
        
        return {
            "dashboard_id": dashboard_id,
            "deleted": True,
            "deleted_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard deletion failed: {str(e)}")

@app.get("/api/business/kpis", tags=["Business Metrics"])
async def get_business_kpis(current_user: User = Depends(get_current_user)):
    """Get business KPI dashboard."""
    try:
        from .business_metrics import get_business_kpi_dashboard
        return await get_business_kpi_dashboard()
    except Exception as e:
        logger.error(f"Business KPI retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Business KPI retrieval failed: {str(e)}")

@app.get("/api/business/health-score", tags=["Business Metrics"])
async def get_business_health_score(current_user: User = Depends(get_current_user)):
    """Get overall business health score."""
    try:
        from .business_metrics import get_business_health_score
        return await get_business_health_score()
    except Exception as e:
        logger.error(f"Business health score retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Business health score failed: {str(e)}")

@app.get("/api/business/user-metrics", tags=["Business Metrics"])
async def get_user_engagement_metrics_endpoint(current_user: User = Depends(get_current_user)):
    """Get user engagement metrics."""
    try:
        from .business_metrics import get_user_engagement_metrics
        user_metrics = await get_user_engagement_metrics()
        
        return {
            "total_users": user_metrics.total_users,
            "active_users_today": user_metrics.active_users_today,
            "active_users_week": user_metrics.active_users_week,
            "active_users_month": user_metrics.active_users_month,
            "new_users_today": user_metrics.new_users_today,
            "new_users_week": user_metrics.new_users_week,
            "new_users_month": user_metrics.new_users_month,
            "retention_rate_7d": user_metrics.retention_rate_7d,
            "retention_rate_30d": user_metrics.retention_rate_30d,
            "avg_session_duration_minutes": user_metrics.avg_session_duration_minutes,
            "avg_sessions_per_user": user_metrics.avg_sessions_per_user,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"User metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"User metrics failed: {str(e)}")

@app.get("/api/business/financial-metrics", tags=["Business Metrics"])
async def get_financial_metrics_endpoint(current_user: User = Depends(get_current_user)):
    """Get financial performance metrics."""
    if current_user.role not in ['admin', 'finance']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to view financial metrics")
    
    try:
        from .business_metrics import get_financial_metrics
        financial_metrics = await get_financial_metrics()
        
        return {
            "monthly_recurring_revenue": financial_metrics.monthly_recurring_revenue,
            "annual_recurring_revenue": financial_metrics.annual_recurring_revenue,
            "total_revenue_today": financial_metrics.total_revenue_today,
            "total_revenue_month": financial_metrics.total_revenue_month,
            "average_revenue_per_user": financial_metrics.average_revenue_per_user,
            "customer_acquisition_cost": financial_metrics.customer_acquisition_cost,
            "customer_lifetime_value": financial_metrics.customer_lifetime_value,
            "churn_rate": financial_metrics.churn_rate,
            "conversion_rate": financial_metrics.conversion_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Financial metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Financial metrics failed: {str(e)}")

@app.post("/api/business/start-collection", tags=["Business Metrics"])
async def start_business_metrics_collection(
    interval_minutes: int = 15,
    current_user: User = Depends(get_current_user)
):
    """Start automated business metrics collection."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to start metrics collection")
    
    try:
        from .business_metrics import business_metrics
        await business_metrics.start_collection(interval_minutes)
        
        return {
            "collection_started": True,
            "interval_minutes": interval_minutes,
            "started_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Business metrics collection start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collection start failed: {str(e)}")

@app.post("/api/business/stop-collection", tags=["Business Metrics"])
async def stop_business_metrics_collection(current_user: User = Depends(get_current_user)):
    """Stop automated business metrics collection."""
    if current_user.role not in ['admin', 'operator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions to stop metrics collection")
    
    try:
        from .business_metrics import business_metrics
        await business_metrics.stop_collection()
        
        return {
            "collection_stopped": True,
            "stopped_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Business metrics collection stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collection stop failed: {str(e)}")

# ====================================
# STARTUP AND BACKGROUND TASKS
# ====================================

@app.on_event("startup")
async def startup_monitoring():
    """Initialize monitoring systems on startup."""
    try:
        # Start health monitoring
        from .health_monitor import health_monitor
        await health_monitor.start_monitoring(interval_seconds=60)
        
        # Start alert processing
        from .monitoring_alerts import alert_manager
        await alert_manager.start_processing()
        
        # Start business metrics collection
        from .business_metrics import business_metrics
        await business_metrics.start_collection(interval_minutes=15)
        
        logger.info("Monitoring systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring systems: {e}")

@app.on_event("shutdown")
async def shutdown_monitoring():
    """Cleanup monitoring systems on shutdown."""
    try:
        # Stop health monitoring
        from .health_monitor import health_monitor
        await health_monitor.stop_monitoring()
        
        # Stop alert processing
        from .monitoring_alerts import alert_manager
        await alert_manager.stop_processing()
        
        # Stop business metrics collection
        from .business_metrics import business_metrics
        await business_metrics.stop_collection()
        
        logger.info("Monitoring systems cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Failed to cleanup monitoring systems: {e}")


# ================================
# SOCKET.IO EVENT HANDLERS
# ================================

# Store active sessions
active_socket_sessions = {}
active_subscriptions = {}

@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    logger.info(f"Socket.IO client {sid} connecting...")
    
    # Verify authentication token
    if not auth or 'token' not in auth:
        logger.warning(f"Socket.IO client {sid} connecting without token")
        await sio.disconnect(sid)
        return False
        
    try:
        token = auth['token']
        payload = auth_manager.verify_token(token, "access")
        user_id = payload["sub"]
        
        # Store session info
        active_socket_sessions[sid] = {
            'user_id': user_id,
            'connected_at': datetime.now(),
            'subscriptions': set()
        }
        
        await sio.emit('connected', {
            'status': 'success',
            'message': 'Connected to SuperNova AI',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        
        logger.info(f"Socket.IO client {sid} connected successfully for user {user_id}")
        return True
        
    except AuthenticationError as e:
        logger.warning(f"Authentication failed for Socket.IO client {sid}: {e}")
        await sio.emit('error', {'message': 'Authentication failed'}, room=sid)
        await sio.disconnect(sid)
        return False
    except Exception as e:
        logger.error(f"Error during Socket.IO connection for {sid}: {e}")
        await sio.disconnect(sid)
        return False

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Socket.IO client {sid} disconnected")
    
    # Clean up session data
    if sid in active_socket_sessions:
        session_info = active_socket_sessions[sid]
        
        # Remove from all subscriptions
        for subscription in session_info['subscriptions']:
            if subscription in active_subscriptions:
                active_subscriptions[subscription].discard(sid)
                if not active_subscriptions[subscription]:
                    del active_subscriptions[subscription]
        
        del active_socket_sessions[sid]

@sio.event
async def subscribe(sid, data):
    """Handle channel subscription"""
    if sid not in active_socket_sessions:
        await sio.emit('error', {'message': 'Not authenticated'}, room=sid)
        return
        
    channel = data.get('channel')
    if not channel:
        await sio.emit('error', {'message': 'Channel name required'}, room=sid)
        return
        
    # Add to subscriptions
    if channel not in active_subscriptions:
        active_subscriptions[channel] = set()
    active_subscriptions[channel].add(sid)
    active_socket_sessions[sid]['subscriptions'].add(channel)
    
    logger.info(f"Client {sid} subscribed to channel: {channel}")
    await sio.emit('subscribed', {'channel': channel}, room=sid)

@sio.event
async def unsubscribe(sid, data):
    """Handle channel unsubscription"""
    if sid not in active_socket_sessions:
        return
        
    channel = data.get('channel')
    if not channel:
        return
        
    # Remove from subscriptions
    if channel in active_subscriptions:
        active_subscriptions[channel].discard(sid)
        if not active_subscriptions[channel]:
            del active_subscriptions[channel]
    
    active_socket_sessions[sid]['subscriptions'].discard(channel)
    
    logger.info(f"Client {sid} unsubscribed from channel: {channel}")
    await sio.emit('unsubscribed', {'channel': channel}, room=sid)

@sio.event
async def chat_message(sid, data):
    """Handle chat messages"""
    if sid not in active_socket_sessions:
        await sio.emit('error', {'message': 'Not authenticated'}, room=sid)
        return
        
    session_info = active_socket_sessions[sid]
    user_id = session_info['user_id']
    
    session_id = data.get('sessionId')
    message = data.get('message')
    metadata = data.get('metadata', {})
    
    if not session_id or not message:
        await sio.emit('error', {'message': 'Session ID and message required'}, room=sid)
        return
        
    try:
        # Create chat request using existing logic
        from .advisor import advise
        
        # Get database session
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            # Create chat context
            chat_context = ChatContext(
                user_id=user_id,
                session_id=session_id,
                previous_messages=[],
                user_preferences={}
            )
            
            # Get advice from advisor
            advice_result = await advise(
                message,
                session_id=session_id,
                profile_id=user.profile_id if user else None,
                context=chat_context,
                db=db
            )
            
            # Emit response back to client
            response_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sessionId': session_id,
                'role': 'assistant',
                'content': advice_result.advice,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'confidence': advice_result.confidence,
                    'suggestions': advice_result.suggestions or [],
                    'charts': []
                }
            }
            
            # Send to specific session channel
            channel = f"chat:{session_id}"
            if channel in active_subscriptions:
                for client_sid in active_subscriptions[channel]:
                    await sio.emit('chat_message', response_message, room=client_sid)
            
            # Also send directly to the sender
            await sio.emit('chat_message', response_message, room=sid)
            
            logger.info(f"Processed chat message for session {session_id}")
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        await sio.emit('error', {'message': 'Failed to process message'}, room=sid)

# Helper function to broadcast to subscribed clients
async def broadcast_to_channel(channel: str, event: str, data: dict):
    """Broadcast data to all clients subscribed to a channel"""
    if channel in active_subscriptions:
        for client_sid in active_subscriptions[channel]:
            try:
                await sio.emit(event, data, room=client_sid)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_sid}: {e}")

# Helper function to send market data updates
async def broadcast_market_data(market_data: MarketDataUpdate):
    """Broadcast market data to subscribed clients"""
    await broadcast_to_channel('market_data', 'market_data', market_data.dict())
    # Also send to symbol-specific channels
    await broadcast_to_channel(f'market_data:{market_data.symbol}', 'market_data', market_data.dict())

# Helper function to send notifications
async def broadcast_notification(notification: dict):
    """Broadcast notification to subscribed clients"""
    await broadcast_to_channel('notifications', 'notification', notification)

# Helper function to send portfolio updates
async def broadcast_portfolio_update(portfolio_update: dict):
    """Broadcast portfolio update to subscribed clients"""
    await broadcast_to_channel('portfolio_updates', 'portfolio_update', portfolio_update)
