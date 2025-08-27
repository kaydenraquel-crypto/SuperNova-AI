"""
SuperNova AI - Comprehensive Secrets Management System

This module provides secure secrets management with:
- Encryption at rest and in transit
- Automatic key rotation
- Multiple secret stores (local, cloud)
- Audit logging and compliance
- Secret lifecycle management
- Secure secret sharing and access control
"""

from __future__ import annotations
import os
import json
import logging
import hashlib
import secrets as py_secrets
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from contextlib import contextmanager

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import aiofiles

from .config_management import Environment, ConfigurationLevel

# Cloud integrations
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    TOKEN = "token"
    CONNECTION_STRING = "connection_string"
    WEBHOOK_SECRET = "webhook_secret"
    ENCRYPTION_KEY = "encryption_key"


class SecretStatus(str, Enum):
    """Secret status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    EXPIRED = "expired"
    PENDING_ROTATION = "pending_rotation"


class SecretStore(str, Enum):
    """Secret storage backends."""
    LOCAL = "local"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    HASHICORP_VAULT = "hashicorp_vault"
    DATABASE = "database"


@dataclass
class SecretMetadata:
    """Metadata for secrets."""
    secret_id: str
    name: str
    secret_type: SecretType
    status: SecretStatus
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    rotation_interval: Optional[timedelta] = None
    next_rotation: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    access_policies: Set[str] = field(default_factory=set)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    encrypted: bool = True
    version: int = 1
    previous_versions: List[str] = field(default_factory=list)


@dataclass
class SecretAccessPolicy:
    """Secret access policy."""
    policy_id: str
    name: str
    description: str
    allowed_environments: Set[Environment]
    allowed_users: Set[str] = field(default_factory=set)
    allowed_services: Set[str] = field(default_factory=set)
    allowed_operations: Set[str] = field(default_factory=set)
    ip_restrictions: Set[str] = field(default_factory=set)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class SecretAuditEvent:
    """Secret audit event."""
    
    def __init__(
        self,
        secret_id: str,
        action: str,
        user: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.secret_id = secret_id
        self.action = action
        self.user = user
        self.result = result
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.ip_address = self.details.get('ip_address')
        self.user_agent = self.details.get('user_agent')


class SecretEncryption:
    """Advanced secret encryption with key rotation support."""
    
    def __init__(self, primary_key: Optional[str] = None, keys: Optional[List[str]] = None):
        """Initialize with primary key or multiple keys for rotation."""
        
        if keys:
            # Multi-key setup for rotation
            self._keys = [key.encode() if isinstance(key, str) else key for key in keys]
            self._cipher = MultiFernet([Fernet(key) for key in self._keys])
        elif primary_key:
            key = primary_key.encode() if isinstance(primary_key, str) else primary_key
            self._cipher = Fernet(key)
        else:
            # Generate new key
            key = Fernet.generate_key()
            self._cipher = Fernet(key)
    
    def encrypt(self, value: str) -> str:
        """Encrypt a string value."""
        return self._cipher.encrypt(value.encode()).decode()
    
    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string value."""
        return self._cipher.decrypt(encrypted_value.encode()).decode()
    
    def rotate_key(self, new_key: str) -> 'SecretEncryption':
        """Add a new key for rotation."""
        if isinstance(self._cipher, MultiFernet):
            keys = list(self._cipher._fernets) + [Fernet(new_key.encode())]
        else:
            keys = [self._cipher, Fernet(new_key.encode())]
        
        return SecretEncryption(keys=[f._signing_key + f._encryption_key for f in keys])
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted."""
        try:
            self._cipher.decrypt(value.encode())
            return True
        except Exception:
            return False
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


class SecretGenerator:
    """Generate secure secrets of various types."""
    
    @staticmethod
    def generate_api_key(length: int = 32, prefix: str = "") -> str:
        """Generate a secure API key."""
        key = py_secrets.token_urlsafe(length)
        return f"{prefix}{key}" if prefix else key
    
    @staticmethod
    def generate_password(
        length: int = 16,
        include_symbols: bool = True,
        include_numbers: bool = True,
        include_uppercase: bool = True,
        include_lowercase: bool = True
    ) -> str:
        """Generate a secure password."""
        import string
        
        chars = ""
        if include_lowercase:
            chars += string.ascii_lowercase
        if include_uppercase:
            chars += string.ascii_uppercase
        if include_numbers:
            chars += string.digits
        if include_symbols:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        if not chars:
            raise ValueError("At least one character type must be included")
        
        return ''.join(py_secrets.choice(chars) for _ in range(length))
    
    @staticmethod
    def generate_webhook_secret(length: int = 32) -> str:
        """Generate a webhook secret."""
        return py_secrets.token_hex(length)
    
    @staticmethod
    def generate_jwt_secret(length: int = 64) -> str:
        """Generate a JWT secret."""
        return py_secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Generate an encryption key."""
        return Fernet.generate_key().decode()


class SecretValidator:
    """Validate secrets for security compliance."""
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength."""
        result = {
            'valid': True,
            'score': 0,
            'issues': [],
            'suggestions': []
        }
        
        # Length check
        if len(password) < 8:
            result['issues'].append("Password too short (minimum 8 characters)")
            result['valid'] = False
        elif len(password) >= 12:
            result['score'] += 2
        else:
            result['score'] += 1
        
        # Character type checks
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        char_types = sum([has_lower, has_upper, has_digit, has_symbol])
        
        if char_types < 3:
            result['issues'].append("Password should contain at least 3 character types")
            result['valid'] = False
        
        result['score'] += char_types
        
        # Common patterns check
        common_patterns = ['123', 'abc', 'password', 'admin', 'qwerty']
        if any(pattern in password.lower() for pattern in common_patterns):
            result['issues'].append("Password contains common patterns")
            result['score'] -= 2
        
        # Repetition check
        if len(set(password)) < len(password) * 0.6:
            result['issues'].append("Password has too many repeated characters")
            result['score'] -= 1
        
        # Suggestions
        if not has_upper:
            result['suggestions'].append("Add uppercase letters")
        if not has_lower:
            result['suggestions'].append("Add lowercase letters")
        if not has_digit:
            result['suggestions'].append("Add numbers")
        if not has_symbol:
            result['suggestions'].append("Add special characters")
        
        return result
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format."""
        # Should be at least 20 characters, alphanumeric + some symbols
        if len(api_key) < 20:
            return False
        
        # Should not contain obvious patterns
        if any(pattern in api_key.lower() for pattern in ['test', 'demo', 'example']):
            return False
        
        return True
    
    @staticmethod
    def check_secret_expiry(metadata: SecretMetadata) -> bool:
        """Check if a secret is expired."""
        if not metadata.expires_at:
            return False
        
        return datetime.utcnow() > metadata.expires_at
    
    @staticmethod
    def check_rotation_needed(metadata: SecretMetadata) -> bool:
        """Check if secret rotation is needed."""
        if not metadata.next_rotation:
            return False
        
        return datetime.utcnow() > metadata.next_rotation


class SecretStore_ABC(ABC):
    """Abstract base class for secret stores."""
    
    @abstractmethod
    async def store_secret(
        self,
        secret_id: str,
        value: str,
        metadata: SecretMetadata
    ) -> bool:
        """Store a secret."""
        pass
    
    @abstractmethod
    async def retrieve_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret."""
        pass
    
    @abstractmethod
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all secret IDs."""
        pass
    
    @abstractmethod
    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get secret metadata."""
        pass
    
    @abstractmethod
    async def update_metadata(self, secret_id: str, metadata: SecretMetadata) -> bool:
        """Update secret metadata."""
        pass


class LocalSecretStore(SecretStore_ABC):
    """Local file-based secret store with encryption."""
    
    def __init__(self, storage_path: Path, encryption: SecretEncryption):
        self.storage_path = storage_path
        self.encryption = encryption
        self.secrets_file = storage_path / "secrets.json"
        self.metadata_file = storage_path / "metadata.json"
        
        # Ensure storage directory exists
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._secrets: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_data()
    
    def _load_data(self):
        """Load secrets and metadata from files."""
        try:
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    self._secrets = json.load(f)
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading secret store data: {e}")
    
    async def _save_data(self):
        """Save secrets and metadata to files."""
        try:
            async with aiofiles.open(self.secrets_file, 'w') as f:
                await f.write(json.dumps(self._secrets, indent=2))
            
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(self._metadata, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Error saving secret store data: {e}")
    
    async def store_secret(
        self,
        secret_id: str,
        value: str,
        metadata: SecretMetadata
    ) -> bool:
        try:
            # Encrypt the secret
            encrypted_value = self.encryption.encrypt(value)
            self._secrets[secret_id] = encrypted_value
            
            # Store metadata
            self._metadata[secret_id] = {
                'secret_id': metadata.secret_id,
                'name': metadata.name,
                'secret_type': metadata.secret_type.value,
                'status': metadata.status.value,
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat(),
                'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                'access_count': metadata.access_count,
                'rotation_interval': metadata.rotation_interval.total_seconds() if metadata.rotation_interval else None,
                'next_rotation': metadata.next_rotation.isoformat() if metadata.next_rotation else None,
                'tags': list(metadata.tags),
                'access_policies': list(metadata.access_policies),
                'audit_log': metadata.audit_log,
                'encrypted': metadata.encrypted,
                'version': metadata.version,
                'previous_versions': metadata.previous_versions
            }
            
            await self._save_data()
            return True
            
        except Exception as e:
            logger.error(f"Error storing secret {secret_id}: {e}")
            return False
    
    async def retrieve_secret(self, secret_id: str) -> Optional[str]:
        try:
            encrypted_value = self._secrets.get(secret_id)
            if not encrypted_value:
                return None
            
            # Update access metadata
            if secret_id in self._metadata:
                self._metadata[secret_id]['last_accessed'] = datetime.utcnow().isoformat()
                self._metadata[secret_id]['access_count'] = self._metadata[secret_id].get('access_count', 0) + 1
                await self._save_data()
            
            return self.encryption.decrypt(encrypted_value)
            
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_id}: {e}")
            return None
    
    async def delete_secret(self, secret_id: str) -> bool:
        try:
            self._secrets.pop(secret_id, None)
            self._metadata.pop(secret_id, None)
            await self._save_data()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting secret {secret_id}: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        return list(self._secrets.keys())
    
    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        meta = self._metadata.get(secret_id)
        if not meta:
            return None
        
        try:
            return SecretMetadata(
                secret_id=meta['secret_id'],
                name=meta['name'],
                secret_type=SecretType(meta['secret_type']),
                status=SecretStatus(meta['status']),
                created_at=datetime.fromisoformat(meta['created_at']),
                updated_at=datetime.fromisoformat(meta['updated_at']),
                expires_at=datetime.fromisoformat(meta['expires_at']) if meta['expires_at'] else None,
                last_accessed=datetime.fromisoformat(meta['last_accessed']) if meta['last_accessed'] else None,
                access_count=meta.get('access_count', 0),
                rotation_interval=timedelta(seconds=meta['rotation_interval']) if meta['rotation_interval'] else None,
                next_rotation=datetime.fromisoformat(meta['next_rotation']) if meta['next_rotation'] else None,
                tags=set(meta.get('tags', [])),
                access_policies=set(meta.get('access_policies', [])),
                audit_log=meta.get('audit_log', []),
                encrypted=meta.get('encrypted', True),
                version=meta.get('version', 1),
                previous_versions=meta.get('previous_versions', [])
            )
        except Exception as e:
            logger.error(f"Error parsing metadata for {secret_id}: {e}")
            return None
    
    async def update_metadata(self, secret_id: str, metadata: SecretMetadata) -> bool:
        if secret_id not in self._metadata:
            return False
        
        try:
            self._metadata[secret_id].update({
                'updated_at': metadata.updated_at.isoformat(),
                'status': metadata.status.value,
                'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                'rotation_interval': metadata.rotation_interval.total_seconds() if metadata.rotation_interval else None,
                'next_rotation': metadata.next_rotation.isoformat() if metadata.next_rotation else None,
                'tags': list(metadata.tags),
                'access_policies': list(metadata.access_policies),
                'audit_log': metadata.audit_log,
                'version': metadata.version,
                'previous_versions': metadata.previous_versions
            })
            
            await self._save_data()
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata for {secret_id}: {e}")
            return False


class AWSSecretsManagerStore(SecretStore_ABC):
    """AWS Secrets Manager integration."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS Secrets Manager")
        
        self.client = boto3.client('secretsmanager', region_name=region_name)
        self.region = region_name
    
    async def store_secret(
        self,
        secret_id: str,
        value: str,
        metadata: SecretMetadata
    ) -> bool:
        try:
            secret_value = {
                'value': value,
                'metadata': {
                    'name': metadata.name,
                    'secret_type': metadata.secret_type.value,
                    'tags': list(metadata.tags)
                }
            }
            
            try:
                # Try to update existing secret
                self.client.update_secret(
                    SecretId=secret_id,
                    SecretString=json.dumps(secret_value)
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new secret
                    self.client.create_secret(
                        Name=secret_id,
                        SecretString=json.dumps(secret_value),
                        Description=metadata.name
                    )
                else:
                    raise
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing secret {secret_id} in AWS: {e}")
            return False
    
    async def retrieve_secret(self, secret_id: str) -> Optional[str]:
        try:
            response = self.client.get_secret_value(SecretId=secret_id)
            secret_data = json.loads(response['SecretString'])
            return secret_data.get('value')
            
        except Exception as e:
            logger.debug(f"Error retrieving secret {secret_id} from AWS: {e}")
            return None
    
    async def delete_secret(self, secret_id: str) -> bool:
        try:
            self.client.delete_secret(
                SecretId=secret_id,
                ForceDeleteWithoutRecovery=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Error deleting secret {secret_id} from AWS: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        try:
            response = self.client.list_secrets()
            return [secret['Name'] for secret in response['SecretList']]
        except Exception as e:
            logger.error(f"Error listing secrets from AWS: {e}")
            return []
    
    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        # AWS Secrets Manager has limited metadata support
        # This would need to be implemented with additional storage
        return None
    
    async def update_metadata(self, secret_id: str, metadata: SecretMetadata) -> bool:
        # AWS Secrets Manager has limited metadata support
        return False


class SecretsManager:
    """Main secrets management class."""
    
    def __init__(
        self,
        primary_store: SecretStore_ABC,
        backup_stores: Optional[List[SecretStore_ABC]] = None,
        encryption: Optional[SecretEncryption] = None,
        audit_callback: Optional[Callable[[SecretAuditEvent], None]] = None
    ):
        self.primary_store = primary_store
        self.backup_stores = backup_stores or []
        self.encryption = encryption or SecretEncryption()
        self.audit_callback = audit_callback
        
        # Access policies
        self._access_policies: Dict[str, SecretAccessPolicy] = {}
        
        # Rotation scheduler
        self._rotation_tasks: Dict[str, asyncio.Task] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def add_access_policy(self, policy: SecretAccessPolicy):
        """Add an access policy."""
        self._access_policies[policy.policy_id] = policy
    
    def _audit_log(
        self,
        secret_id: str,
        action: str,
        user: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit event."""
        event = SecretAuditEvent(secret_id, action, user, result, details)
        
        if self.audit_callback:
            try:
                self.audit_callback(event)
            except Exception as e:
                logger.error(f"Error in audit callback: {e}")
        
        logger.info(f"Secret audit: {action} {secret_id} by {user} - {result}")
    
    def _check_access_policy(
        self,
        secret_id: str,
        user: str,
        operation: str,
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if access is allowed based on policies."""
        
        # Get secret metadata to check assigned policies
        try:
            loop = asyncio.get_event_loop()
            metadata = loop.run_until_complete(self.primary_store.get_metadata(secret_id))
            if not metadata:
                return False
            
            # Check each assigned policy
            for policy_id in metadata.access_policies:
                policy = self._access_policies.get(policy_id)
                if not policy:
                    continue
                
                # Check environment
                if environment not in policy.allowed_environments:
                    continue
                
                # Check user
                if policy.allowed_users and user not in policy.allowed_users:
                    continue
                
                # Check operation
                if policy.allowed_operations and operation not in policy.allowed_operations:
                    continue
                
                # Check expiry
                if policy.expires_at and datetime.utcnow() > policy.expires_at:
                    continue
                
                # If we get here, access is allowed
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking access policy: {e}")
            return False
    
    async def create_secret(
        self,
        name: str,
        secret_type: SecretType,
        value: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        rotation_interval: Optional[timedelta] = None,
        tags: Optional[Set[str]] = None,
        access_policies: Optional[Set[str]] = None,
        user: str = "system",
        auto_generate: bool = True
    ) -> Optional[str]:
        """Create a new secret."""
        
        with self._lock:
            secret_id = hashlib.sha256(f"{name}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
            
            try:
                # Generate value if not provided
                if not value and auto_generate:
                    if secret_type == SecretType.API_KEY:
                        value = SecretGenerator.generate_api_key()
                    elif secret_type == SecretType.PASSWORD:
                        value = SecretGenerator.generate_password()
                    elif secret_type == SecretType.WEBHOOK_SECRET:
                        value = SecretGenerator.generate_webhook_secret()
                    elif secret_type == SecretType.ENCRYPTION_KEY:
                        value = SecretGenerator.generate_encryption_key()
                    else:
                        self._audit_log(secret_id, "create", user, "failed", {"reason": "no_value_provided"})
                        return None
                
                if not value:
                    self._audit_log(secret_id, "create", user, "failed", {"reason": "no_value_provided"})
                    return None
                
                # Validate secret
                if secret_type == SecretType.PASSWORD:
                    validation = SecretValidator.validate_password_strength(value)
                    if not validation['valid']:
                        self._audit_log(secret_id, "create", user, "failed", {"validation_issues": validation['issues']})
                        return None
                elif secret_type == SecretType.API_KEY:
                    if not SecretValidator.validate_api_key_format(value):
                        self._audit_log(secret_id, "create", user, "failed", {"reason": "invalid_api_key_format"})
                        return None
                
                # Calculate next rotation if interval is set
                next_rotation = None
                if rotation_interval:
                    next_rotation = datetime.utcnow() + rotation_interval
                
                # Create metadata
                metadata = SecretMetadata(
                    secret_id=secret_id,
                    name=name,
                    secret_type=secret_type,
                    status=SecretStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    expires_at=expires_at,
                    rotation_interval=rotation_interval,
                    next_rotation=next_rotation,
                    tags=tags or set(),
                    access_policies=access_policies or set()
                )
                
                # Store in primary store
                success = await self.primary_store.store_secret(secret_id, value, metadata)
                if not success:
                    self._audit_log(secret_id, "create", user, "failed", {"reason": "store_error"})
                    return None
                
                # Store in backup stores
                for backup_store in self.backup_stores:
                    try:
                        await backup_store.store_secret(secret_id, value, metadata)
                    except Exception as e:
                        logger.warning(f"Failed to store secret in backup store: {e}")
                
                # Schedule rotation if needed
                if next_rotation:
                    self._schedule_rotation(secret_id, next_rotation)
                
                self._audit_log(secret_id, "create", user, "success", {"secret_type": secret_type.value})
                return secret_id
                
            except Exception as e:
                logger.error(f"Error creating secret: {e}")
                self._audit_log(secret_id, "create", user, "error", {"error": str(e)})
                return None
    
    async def get_secret(
        self,
        secret_id: str,
        user: str = "system",
        environment: Environment = Environment.DEVELOPMENT
    ) -> Optional[str]:
        """Retrieve a secret."""
        
        try:
            # Check access policy
            if not self._check_access_policy(secret_id, user, "read", environment):
                self._audit_log(secret_id, "get", user, "access_denied")
                return None
            
            # Get from primary store
            value = await self.primary_store.retrieve_secret(secret_id)
            
            if value is None:
                # Try backup stores
                for backup_store in self.backup_stores:
                    try:
                        value = await backup_store.retrieve_secret(secret_id)
                        if value:
                            break
                    except Exception as e:
                        logger.debug(f"Error retrieving from backup store: {e}")
            
            if value:
                # Check if secret is expired
                metadata = await self.primary_store.get_metadata(secret_id)
                if metadata and SecretValidator.check_secret_expiry(metadata):
                    self._audit_log(secret_id, "get", user, "expired")
                    return None
                
                self._audit_log(secret_id, "get", user, "success")
                return value
            else:
                self._audit_log(secret_id, "get", user, "not_found")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_id}: {e}")
            self._audit_log(secret_id, "get", user, "error", {"error": str(e)})
            return None
    
    async def update_secret(
        self,
        secret_id: str,
        new_value: str,
        user: str = "system",
        environment: Environment = Environment.DEVELOPMENT
    ) -> bool:
        """Update a secret value."""
        
        try:
            # Check access policy
            if not self._check_access_policy(secret_id, user, "update", environment):
                self._audit_log(secret_id, "update", user, "access_denied")
                return False
            
            # Get current metadata
            metadata = await self.primary_store.get_metadata(secret_id)
            if not metadata:
                self._audit_log(secret_id, "update", user, "not_found")
                return False
            
            # Validate new value
            if metadata.secret_type == SecretType.PASSWORD:
                validation = SecretValidator.validate_password_strength(new_value)
                if not validation['valid']:
                    self._audit_log(secret_id, "update", user, "validation_failed", {"issues": validation['issues']})
                    return False
            
            # Archive current version
            current_value = await self.primary_store.retrieve_secret(secret_id)
            if current_value:
                metadata.previous_versions.append(current_value)
                # Keep only last 5 versions
                if len(metadata.previous_versions) > 5:
                    metadata.previous_versions.pop(0)
            
            # Update metadata
            metadata.updated_at = datetime.utcnow()
            metadata.version += 1
            
            # Store new value
            success = await self.primary_store.store_secret(secret_id, new_value, metadata)
            if not success:
                self._audit_log(secret_id, "update", user, "store_error")
                return False
            
            # Update backup stores
            for backup_store in self.backup_stores:
                try:
                    await backup_store.store_secret(secret_id, new_value, metadata)
                except Exception as e:
                    logger.warning(f"Failed to update secret in backup store: {e}")
            
            self._audit_log(secret_id, "update", user, "success")
            return True
            
        except Exception as e:
            logger.error(f"Error updating secret {secret_id}: {e}")
            self._audit_log(secret_id, "update", user, "error", {"error": str(e)})
            return False
    
    async def delete_secret(
        self,
        secret_id: str,
        user: str = "system",
        environment: Environment = Environment.DEVELOPMENT
    ) -> bool:
        """Delete a secret."""
        
        try:
            # Check access policy
            if not self._check_access_policy(secret_id, user, "delete", environment):
                self._audit_log(secret_id, "delete", user, "access_denied")
                return False
            
            # Delete from primary store
            success = await self.primary_store.delete_secret(secret_id)
            
            # Delete from backup stores
            for backup_store in self.backup_stores:
                try:
                    await backup_store.delete_secret(secret_id)
                except Exception as e:
                    logger.warning(f"Failed to delete secret from backup store: {e}")
            
            # Cancel rotation task if exists
            if secret_id in self._rotation_tasks:
                self._rotation_tasks[secret_id].cancel()
                del self._rotation_tasks[secret_id]
            
            if success:
                self._audit_log(secret_id, "delete", user, "success")
            else:
                self._audit_log(secret_id, "delete", user, "failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting secret {secret_id}: {e}")
            self._audit_log(secret_id, "delete", user, "error", {"error": str(e)})
            return False
    
    async def rotate_secret(
        self,
        secret_id: str,
        user: str = "system",
        environment: Environment = Environment.DEVELOPMENT
    ) -> bool:
        """Rotate a secret (generate new value)."""
        
        try:
            metadata = await self.primary_store.get_metadata(secret_id)
            if not metadata:
                return False
            
            # Generate new value based on secret type
            new_value = None
            if metadata.secret_type == SecretType.API_KEY:
                new_value = SecretGenerator.generate_api_key()
            elif metadata.secret_type == SecretType.PASSWORD:
                new_value = SecretGenerator.generate_password()
            elif metadata.secret_type == SecretType.WEBHOOK_SECRET:
                new_value = SecretGenerator.generate_webhook_secret()
            elif metadata.secret_type == SecretType.ENCRYPTION_KEY:
                new_value = SecretGenerator.generate_encryption_key()
            
            if not new_value:
                self._audit_log(secret_id, "rotate", user, "failed", {"reason": "unsupported_type"})
                return False
            
            # Update the secret
            success = await self.update_secret(secret_id, new_value, user, environment)
            
            if success:
                # Update next rotation time
                if metadata.rotation_interval:
                    metadata.next_rotation = datetime.utcnow() + metadata.rotation_interval
                    await self.primary_store.update_metadata(secret_id, metadata)
                    self._schedule_rotation(secret_id, metadata.next_rotation)
                
                self._audit_log(secret_id, "rotate", user, "success")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rotating secret {secret_id}: {e}")
            self._audit_log(secret_id, "rotate", user, "error", {"error": str(e)})
            return False
    
    def _schedule_rotation(self, secret_id: str, rotation_time: datetime):
        """Schedule automatic rotation for a secret."""
        
        async def rotation_task():
            try:
                # Wait until rotation time
                delay = (rotation_time - datetime.utcnow()).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Perform rotation
                await self.rotate_secret(secret_id, user="auto_rotation")
                
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in automatic rotation for {secret_id}: {e}")
        
        # Cancel existing task if any
        if secret_id in self._rotation_tasks:
            self._rotation_tasks[secret_id].cancel()
        
        # Schedule new task
        self._rotation_tasks[secret_id] = asyncio.create_task(rotation_task())
    
    async def list_secrets(
        self,
        user: str = "system",
        environment: Environment = Environment.DEVELOPMENT,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List secrets with metadata."""
        
        try:
            secret_ids = await self.primary_store.list_secrets()
            secrets_info = []
            
            for secret_id in secret_ids:
                # Check access policy for listing
                if not self._check_access_policy(secret_id, user, "list", environment):
                    continue
                
                metadata = await self.primary_store.get_metadata(secret_id)
                if not metadata:
                    continue
                
                # Apply filters
                if filters:
                    if filters.get('secret_type') and metadata.secret_type != filters['secret_type']:
                        continue
                    if filters.get('status') and metadata.status != filters['status']:
                        continue
                    if filters.get('tags') and not filters['tags'].intersection(metadata.tags):
                        continue
                
                secrets_info.append({
                    'secret_id': secret_id,
                    'name': metadata.name,
                    'secret_type': metadata.secret_type.value,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat(),
                    'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                    'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                    'access_count': metadata.access_count,
                    'tags': list(metadata.tags),
                    'version': metadata.version,
                    'rotation_needed': SecretValidator.check_rotation_needed(metadata),
                    'expired': SecretValidator.check_secret_expiry(metadata)
                })
            
            self._audit_log("*", "list", user, "success", {"count": len(secrets_info)})
            return secrets_info
            
        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            self._audit_log("*", "list", user, "error", {"error": str(e)})
            return []
    
    async def get_secret_health(self) -> Dict[str, Any]:
        """Get health status of secrets management."""
        
        try:
            secret_ids = await self.primary_store.list_secrets()
            
            health = {
                'total_secrets': len(secret_ids),
                'active_secrets': 0,
                'expired_secrets': 0,
                'rotation_needed': 0,
                'backup_stores': len(self.backup_stores),
                'rotation_tasks': len(self._rotation_tasks),
                'issues': []
            }
            
            for secret_id in secret_ids:
                metadata = await self.primary_store.get_metadata(secret_id)
                if not metadata:
                    continue
                
                if metadata.status == SecretStatus.ACTIVE:
                    health['active_secrets'] += 1
                
                if SecretValidator.check_secret_expiry(metadata):
                    health['expired_secrets'] += 1
                
                if SecretValidator.check_rotation_needed(metadata):
                    health['rotation_needed'] += 1
            
            # Check for issues
            if health['expired_secrets'] > 0:
                health['issues'].append(f"{health['expired_secrets']} secrets have expired")
            
            if health['rotation_needed'] > 0:
                health['issues'].append(f"{health['rotation_needed']} secrets need rotation")
            
            return health
            
        except Exception as e:
            logger.error(f"Error checking secret health: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown the secrets manager."""
        # Cancel all rotation tasks
        for task in self._rotation_tasks.values():
            task.cancel()
        
        logger.info("Secrets manager shut down")


# Global secrets manager
secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global secrets_manager
    
    if secrets_manager is None:
        # Initialize with local store by default
        storage_path = Path(os.getenv('SECRETS_STORAGE_PATH', './secrets'))
        encryption = SecretEncryption(os.getenv('SECRETS_ENCRYPTION_KEY'))
        local_store = LocalSecretStore(storage_path, encryption)
        
        # Add cloud backup stores if configured
        backup_stores = []
        
        # AWS Secrets Manager
        if os.getenv('AWS_SECRETS_MANAGER_ENABLED', 'false').lower() == 'true':
            try:
                aws_store = AWSSecretsManagerStore(os.getenv('AWS_REGION', 'us-east-1'))
                backup_stores.append(aws_store)
            except Exception as e:
                logger.warning(f"Failed to initialize AWS Secrets Manager: {e}")
        
        secrets_manager = SecretsManager(local_store, backup_stores)
    
    return secrets_manager


# Convenience functions
async def create_secret(name: str, secret_type: SecretType, **kwargs) -> Optional[str]:
    """Create a new secret."""
    manager = get_secrets_manager()
    return await manager.create_secret(name, secret_type, **kwargs)


async def get_secret(secret_id: str, **kwargs) -> Optional[str]:
    """Get a secret value."""
    manager = get_secrets_manager()
    return await manager.get_secret(secret_id, **kwargs)


async def update_secret(secret_id: str, new_value: str, **kwargs) -> bool:
    """Update a secret value."""
    manager = get_secrets_manager()
    return await manager.update_secret(secret_id, new_value, **kwargs)


async def delete_secret(secret_id: str, **kwargs) -> bool:
    """Delete a secret."""
    manager = get_secrets_manager()
    return await manager.delete_secret(secret_id, **kwargs)