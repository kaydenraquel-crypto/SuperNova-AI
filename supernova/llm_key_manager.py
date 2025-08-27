"""
Secure API Key Management System for SuperNova AI LLM Providers

This module provides:
- Encrypted API key storage and retrieval
- Key rotation and lifecycle management  
- Secure key validation and health checking
- Backup key support for high availability
- Audit logging for key access and operations
"""

from __future__ import annotations
from typing import Dict, Optional, List, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import hashlib
import secrets
import base64
import logging
from pathlib import Path

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidToken

# Core imports
from .config import settings
from .db import SessionLocal
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

# Database models for secure key storage
Base = declarative_base()

class APIKeyStatus(str, Enum):
    """API key status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATING = "rotating"

class ProviderType(str, Enum):
    """LLM provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"

@dataclass
class KeyMetrics:
    """API key usage metrics"""
    requests_count: int = 0
    tokens_used: int = 0
    cost_incurred: float = 0.0
    last_used: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 100.0

class EncryptedAPIKey(Base):
    """Database model for encrypted API key storage"""
    __tablename__ = "encrypted_api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    key_name = Column(String(100), nullable=False, index=True)
    encrypted_key = Column(Text, nullable=False)
    key_hash = Column(String(64), nullable=False, unique=True)
    status = Column(String(20), default=APIKeyStatus.ACTIVE)
    is_primary = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    metadata = Column(Text, nullable=True)  # JSON metadata

class KeyAuditLog(Base):
    """Audit log for API key operations"""
    __tablename__ = "key_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(Integer, nullable=True)  # Reference to EncryptedAPIKey
    provider = Column(String(50), nullable=False)
    operation = Column(String(50), nullable=False)  # create, access, rotate, revoke, etc.
    status = Column(String(20), nullable=False)  # success, failure, error
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class SecureKeyManager:
    """
    Secure API key management with encryption, rotation, and audit logging
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize secure key manager
        
        Args:
            encryption_key: Master encryption key (will generate if not provided)
        """
        self.encryption_key = encryption_key or self._get_or_generate_master_key()
        self.fernet = self._init_encryption()
        self.session_factory = sessionmaker()
        self.engine = None
        self.key_cache: Dict[str, Tuple[str, datetime]] = {}
        self.metrics: Dict[str, KeyMetrics] = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("SecureKeyManager initialized with encryption enabled")
    
    def _get_or_generate_master_key(self) -> str:
        """Get or generate master encryption key"""
        if hasattr(settings, 'API_KEY_ENCRYPTION_KEY') and settings.API_KEY_ENCRYPTION_KEY:
            return settings.API_KEY_ENCRYPTION_KEY
        
        # Generate new key
        key = Fernet.generate_key().decode()
        logger.warning(f"Generated new master encryption key. Save this key: {key}")
        return key
    
    def _init_encryption(self) -> Fernet:
        """Initialize encryption cipher"""
        try:
            # If key looks like a password, derive it
            if len(self.encryption_key) < 32:
                password = self.encryption_key.encode()
                salt = b'supernova_salt_2024'  # In production, use random salt stored securely
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
            else:
                # Assume it's already a proper Fernet key
                key = self.encryption_key.encode()
            
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _init_database(self):
        """Initialize database connection and tables"""
        try:
            database_url = settings.DATABASE_URL
            if database_url.startswith("sqlite"):
                self.engine = create_engine(database_url, echo=False)
            else:
                self.engine = create_engine(database_url, pool_size=5, max_overflow=10)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Update session factory
            self.session_factory.configure(bind=self.engine)
            
            logger.info("Key management database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize key management database: {e}")
            raise
    
    def _get_session(self) -> Session:
        """Get database session"""
        return self.session_factory()
    
    def _hash_key(self, api_key: str) -> str:
        """Create secure hash of API key for identification"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _encrypt_key(self, api_key: str) -> str:
        """Encrypt API key"""
        return self.fernet.encrypt(api_key.encode()).decode()
    
    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        try:
            return self.fernet.decrypt(encrypted_key.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt API key - invalid token")
            raise ValueError("Invalid encryption token")
    
    def _log_audit_event(self, 
                        provider: str, 
                        operation: str, 
                        status: str = "success",
                        key_id: Optional[int] = None,
                        error_message: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Log audit event"""
        try:
            with self._get_session() as session:
                audit_log = KeyAuditLog(
                    key_id=key_id,
                    provider=provider,
                    operation=operation,
                    status=status,
                    error_message=error_message,
                    metadata=json.dumps(metadata) if metadata else None
                )
                session.add(audit_log)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    async def store_api_key(self,
                          provider: ProviderType,
                          api_key: str,
                          key_name: str = "primary",
                          is_primary: bool = True,
                          expires_at: Optional[datetime] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store API key securely
        
        Args:
            provider: LLM provider type
            api_key: The actual API key to store
            key_name: Identifier for the key (e.g., 'primary', 'backup')
            is_primary: Whether this is the primary key for the provider
            expires_at: Optional expiration date
            metadata: Additional metadata
            
        Returns:
            True if stored successfully
        """
        try:
            # Validate key format
            if not self._validate_key_format(provider, api_key):
                raise ValueError(f"Invalid API key format for provider {provider}")
            
            key_hash = self._hash_key(api_key)
            encrypted_key = self._encrypt_key(api_key)
            
            with self._get_session() as session:
                # Check if key already exists
                existing = session.query(EncryptedAPIKey).filter(
                    EncryptedAPIKey.key_hash == key_hash
                ).first()
                
                if existing:
                    logger.warning(f"API key already exists for {provider}")
                    return False
                
                # If this is primary, make other keys non-primary
                if is_primary:
                    session.query(EncryptedAPIKey).filter(
                        EncryptedAPIKey.provider == provider.value,
                        EncryptedAPIKey.is_primary == True
                    ).update({"is_primary": False})
                
                # Store new key
                new_key = EncryptedAPIKey(
                    provider=provider.value,
                    key_name=key_name,
                    encrypted_key=encrypted_key,
                    key_hash=key_hash,
                    is_primary=is_primary,
                    expires_at=expires_at,
                    metadata=json.dumps(metadata) if metadata else None
                )
                
                session.add(new_key)
                session.commit()
                
                # Log audit event
                self._log_audit_event(
                    provider.value,
                    "store",
                    key_id=new_key.id,
                    metadata={"key_name": key_name, "is_primary": is_primary}
                )
                
                logger.info(f"API key stored for {provider} ({key_name})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store API key for {provider}: {e}")
            self._log_audit_event(provider.value, "store", "error", error_message=str(e))
            return False
    
    async def get_api_key(self,
                         provider: ProviderType,
                         key_name: str = "primary",
                         use_cache: bool = True) -> Optional[str]:
        """
        Retrieve API key for provider
        
        Args:
            provider: LLM provider type
            key_name: Key identifier
            use_cache: Whether to use cached key
            
        Returns:
            Decrypted API key or None if not found
        """
        cache_key = f"{provider.value}:{key_name}"
        
        try:
            # Check cache first
            if use_cache and cache_key in self.key_cache:
                cached_key, cached_time = self.key_cache[cache_key]
                if datetime.utcnow() - cached_time < timedelta(minutes=30):
                    return cached_key
            
            with self._get_session() as session:
                # Get primary key for provider
                query = session.query(EncryptedAPIKey).filter(
                    EncryptedAPIKey.provider == provider.value,
                    EncryptedAPIKey.status == APIKeyStatus.ACTIVE
                )
                
                if key_name == "primary":
                    query = query.filter(EncryptedAPIKey.is_primary == True)
                else:
                    query = query.filter(EncryptedAPIKey.key_name == key_name)
                
                key_record = query.first()
                
                if not key_record:
                    logger.warning(f"No API key found for {provider} ({key_name})")
                    return None
                
                # Check expiration
                if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                    logger.warning(f"API key expired for {provider} ({key_name})")
                    key_record.status = APIKeyStatus.EXPIRED
                    session.commit()
                    return None
                
                # Decrypt key
                api_key = self._decrypt_key(key_record.encrypted_key)
                
                # Update usage stats
                key_record.last_used_at = datetime.utcnow()
                key_record.usage_count += 1
                session.commit()
                
                # Cache key
                if use_cache:
                    self.key_cache[cache_key] = (api_key, datetime.utcnow())
                
                # Log access
                self._log_audit_event(
                    provider.value,
                    "access",
                    key_id=key_record.id,
                    metadata={"key_name": key_name}
                )
                
                return api_key
                
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            self._log_audit_event(provider.value, "access", "error", error_message=str(e))
            return None
    
    async def rotate_api_key(self,
                           provider: ProviderType,
                           new_api_key: str,
                           key_name: str = "primary") -> bool:
        """
        Rotate API key (replace old with new)
        
        Args:
            provider: LLM provider type
            new_api_key: New API key
            key_name: Key identifier
            
        Returns:
            True if rotation successful
        """
        try:
            with self._get_session() as session:
                # Find existing key
                key_record = session.query(EncryptedAPIKey).filter(
                    EncryptedAPIKey.provider == provider.value,
                    EncryptedAPIKey.key_name == key_name,
                    EncryptedAPIKey.status == APIKeyStatus.ACTIVE
                ).first()
                
                if not key_record:
                    logger.error(f"No existing key to rotate for {provider} ({key_name})")
                    return False
                
                # Mark old key as rotating
                key_record.status = APIKeyStatus.ROTATING
                session.commit()
                
                # Store new key
                success = await self.store_api_key(
                    provider=provider,
                    api_key=new_api_key,
                    key_name=key_name,
                    is_primary=key_record.is_primary,
                    metadata={"rotated_from": key_record.id}
                )
                
                if success:
                    # Mark old key as inactive
                    key_record.status = APIKeyStatus.INACTIVE
                    session.commit()
                    
                    # Clear cache
                    cache_key = f"{provider.value}:{key_name}"
                    if cache_key in self.key_cache:
                        del self.key_cache[cache_key]
                    
                    self._log_audit_event(
                        provider.value,
                        "rotate",
                        key_id=key_record.id,
                        metadata={"key_name": key_name}
                    )
                    
                    logger.info(f"API key rotated for {provider} ({key_name})")
                    return True
                else:
                    # Restore old key status
                    key_record.status = APIKeyStatus.ACTIVE
                    session.commit()
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to rotate API key for {provider}: {e}")
            self._log_audit_event(provider.value, "rotate", "error", error_message=str(e))
            return False
    
    async def revoke_api_key(self,
                           provider: ProviderType,
                           key_name: str = "primary") -> bool:
        """
        Revoke API key (mark as revoked and remove from cache)
        
        Args:
            provider: LLM provider type
            key_name: Key identifier
            
        Returns:
            True if revoked successfully
        """
        try:
            with self._get_session() as session:
                key_record = session.query(EncryptedAPIKey).filter(
                    EncryptedAPIKey.provider == provider.value,
                    EncryptedAPIKey.key_name == key_name,
                    EncryptedAPIKey.status.in_([APIKeyStatus.ACTIVE, APIKeyStatus.INACTIVE])
                ).first()
                
                if not key_record:
                    logger.warning(f"No key to revoke for {provider} ({key_name})")
                    return False
                
                # Mark as revoked
                key_record.status = APIKeyStatus.REVOKED
                session.commit()
                
                # Clear from cache
                cache_key = f"{provider.value}:{key_name}"
                if cache_key in self.key_cache:
                    del self.key_cache[cache_key]
                
                self._log_audit_event(
                    provider.value,
                    "revoke",
                    key_id=key_record.id,
                    metadata={"key_name": key_name}
                )
                
                logger.info(f"API key revoked for {provider} ({key_name})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to revoke API key for {provider}: {e}")
            self._log_audit_event(provider.value, "revoke", "error", error_message=str(e))
            return False
    
    def _validate_key_format(self, provider: ProviderType, api_key: str) -> bool:
        """Validate API key format for provider"""
        if not api_key or len(api_key) < 10:
            return False
        
        # Provider-specific validation
        if provider == ProviderType.OPENAI:
            return api_key.startswith("sk-") and len(api_key) > 40
        elif provider == ProviderType.ANTHROPIC:
            return api_key.startswith("sk-ant-") and len(api_key) > 50
        elif provider == ProviderType.HUGGINGFACE:
            return api_key.startswith("hf_") and len(api_key) > 30
        elif provider == ProviderType.OLLAMA:
            return True  # Ollama typically doesn't require API keys
        
        return True
    
    async def validate_api_key(self,
                             provider: ProviderType,
                             api_key: Optional[str] = None,
                             key_name: str = "primary") -> bool:
        """
        Validate API key by making a test request
        
        Args:
            provider: LLM provider type
            api_key: API key to validate (if None, retrieves from storage)
            key_name: Key identifier
            
        Returns:
            True if key is valid
        """
        try:
            if not api_key:
                api_key = await self.get_api_key(provider, key_name)
                if not api_key:
                    return False
            
            # Make test request based on provider
            if provider == ProviderType.OPENAI:
                return await self._validate_openai_key(api_key)
            elif provider == ProviderType.ANTHROPIC:
                return await self._validate_anthropic_key(api_key)
            elif provider == ProviderType.HUGGINGFACE:
                return await self._validate_huggingface_key(api_key)
            elif provider == ProviderType.OLLAMA:
                return await self._validate_ollama_connection()
            
            return False
            
        except Exception as e:
            logger.error(f"API key validation failed for {provider}: {e}")
            return False
    
    async def _validate_openai_key(self, api_key: str) -> bool:
        """Validate OpenAI API key"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Make a minimal request
            response = client.models.list()
            return len(response.data) > 0
        except Exception:
            return False
    
    async def _validate_anthropic_key(self, api_key: str) -> bool:
        """Validate Anthropic API key"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            # Make a minimal request
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return response is not None
        except Exception:
            return False
    
    async def _validate_huggingface_key(self, api_key: str) -> bool:
        """Validate HuggingFace API token"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://huggingface.co/api/whoami-v2",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                return response.status_code == 200
        except Exception:
            return False
    
    async def _validate_ollama_connection(self) -> bool:
        """Validate Ollama server connection"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_key_metrics(self, provider: ProviderType) -> Dict[str, Any]:
        """Get API key usage metrics"""
        try:
            with self._get_session() as session:
                keys = session.query(EncryptedAPIKey).filter(
                    EncryptedAPIKey.provider == provider.value
                ).all()
                
                metrics = {
                    "total_keys": len(keys),
                    "active_keys": len([k for k in keys if k.status == APIKeyStatus.ACTIVE]),
                    "primary_key_exists": any(k.is_primary for k in keys if k.status == APIKeyStatus.ACTIVE),
                    "keys_by_status": {},
                    "oldest_key": None,
                    "newest_key": None,
                    "total_usage": 0
                }
                
                if keys:
                    # Status breakdown
                    for status in APIKeyStatus:
                        metrics["keys_by_status"][status.value] = len([k for k in keys if k.status == status])
                    
                    # Age information
                    metrics["oldest_key"] = min(k.created_at for k in keys).isoformat()
                    metrics["newest_key"] = max(k.created_at for k in keys).isoformat()
                    
                    # Usage stats
                    metrics["total_usage"] = sum(k.usage_count or 0 for k in keys)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get key metrics for {provider}: {e}")
            return {}
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired keys and return count removed"""
        try:
            with self._get_session() as session:
                expired_keys = session.query(EncryptedAPIKey).filter(
                    EncryptedAPIKey.expires_at < datetime.utcnow(),
                    EncryptedAPIKey.status != APIKeyStatus.EXPIRED
                ).all()
                
                count = len(expired_keys)
                
                for key in expired_keys:
                    key.status = APIKeyStatus.EXPIRED
                    self._log_audit_event(
                        key.provider,
                        "expire",
                        key_id=key.id,
                        metadata={"key_name": key.key_name}
                    )
                
                session.commit()
                
                # Clear cache for expired keys
                for key in expired_keys:
                    cache_key = f"{key.provider}:{key.key_name}"
                    if cache_key in self.key_cache:
                        del self.key_cache[cache_key]
                
                if count > 0:
                    logger.info(f"Marked {count} expired API keys")
                
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0

# Global key manager instance
key_manager: Optional[SecureKeyManager] = None

def get_key_manager() -> SecureKeyManager:
    """Get global key manager instance"""
    global key_manager
    if key_manager is None:
        key_manager = SecureKeyManager()
    return key_manager

async def initialize_key_manager() -> SecureKeyManager:
    """Initialize and return key manager"""
    manager = get_key_manager()
    
    # Load keys from environment if configured
    await _load_keys_from_env(manager)
    
    return manager

async def _load_keys_from_env(manager: SecureKeyManager):
    """Load API keys from environment variables"""
    try:
        # OpenAI
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            await manager.store_api_key(
                ProviderType.OPENAI,
                settings.OPENAI_API_KEY,
                "primary",
                is_primary=True,
                metadata={"source": "environment"}
            )
        
        # Anthropic
        if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
            await manager.store_api_key(
                ProviderType.ANTHROPIC,
                settings.ANTHROPIC_API_KEY,
                "primary", 
                is_primary=True,
                metadata={"source": "environment"}
            )
        
        # HuggingFace
        if hasattr(settings, 'HUGGINGFACE_API_TOKEN') and settings.HUGGINGFACE_API_TOKEN:
            await manager.store_api_key(
                ProviderType.HUGGINGFACE,
                settings.HUGGINGFACE_API_TOKEN,
                "primary",
                is_primary=True,
                metadata={"source": "environment"}
            )
        
        logger.info("API keys loaded from environment")
        
    except Exception as e:
        logger.error(f"Failed to load keys from environment: {e}")