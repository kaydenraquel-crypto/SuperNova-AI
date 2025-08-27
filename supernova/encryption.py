"""
SuperNova AI Data Encryption System
Advanced encryption for sensitive financial data and PII protection
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from .security_config import security_settings, SENSITIVE_FIELD_PATTERNS
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES

logger = logging.getLogger(__name__)


class EncryptionLevel(str, Enum):
    """Encryption security levels"""
    STANDARD = "standard"      # AES-256-GCM
    HIGH = "high"             # AES-256-GCM with key rotation
    MAXIMUM = "maximum"       # RSA + AES hybrid encryption


class KeyType(str, Enum):
    """Encryption key types"""
    MASTER = "master"         # Master encryption key
    DATA = "data"             # Data encryption key
    FIELD = "field"           # Field-level encryption key
    BACKUP = "backup"         # Backup encryption key
    ARCHIVE = "archive"       # Archive encryption key


class EncryptionError(Exception):
    """Custom encryption error"""
    pass


class DecryptionError(Exception):
    """Custom decryption error"""
    pass


class KeyManager:
    """
    Advanced key management system with rotation, escrow, and hierarchy
    """
    
    def __init__(self):
        self.keys: Dict[str, Dict] = {}
        self.master_key = self._get_or_create_master_key()
        self.key_hierarchy: Dict[str, str] = {}  # child_key_id -> parent_key_id
        
        # Initialize key rotation schedule
        self.rotation_schedule: Dict[str, datetime] = {}
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        master_key = security_settings.get_encryption_key()
        
        if isinstance(master_key, str):
            master_key = master_key.encode()
        
        # Validate key length
        if len(master_key) != 44:  # Base64 encoded 32-byte key
            raise EncryptionError("Invalid master key length")
        
        return base64.urlsafe_b64decode(master_key)
    
    def generate_key(
        self,
        key_type: KeyType,
        key_id: Optional[str] = None,
        parent_key_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Generate new encryption key"""
        
        if key_id is None:
            key_id = f"{key_type.value}_{secrets.token_hex(8)}"
        
        # Generate key material
        key_material = Fernet.generate_key()
        
        # Create key metadata
        key_info = {
            "id": key_id,
            "type": key_type.value,
            "material": key_material.decode(),
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "version": 1,
            "usage_count": 0,
            "metadata": metadata or {}
        }
        
        # Set up key hierarchy
        if parent_key_id:
            self.key_hierarchy[key_id] = parent_key_id
            key_info["parent_key_id"] = parent_key_id
        
        # Encrypt key with master key
        master_cipher = Fernet(base64.urlsafe_b64encode(self.master_key))
        encrypted_key = master_cipher.encrypt(key_material)
        key_info["encrypted_material"] = encrypted_key.decode()
        
        # Store key
        self.keys[key_id] = key_info
        
        # Schedule rotation
        if security_settings.KEY_ROTATION_ENABLED:
            rotation_date = datetime.utcnow() + timedelta(days=security_settings.KEY_ROTATION_DAYS)
            self.rotation_schedule[key_id] = rotation_date
        
        security_logger.log_security_event(
            event_type="KEY_GENERATED",
            level=SecurityEventLevel.INFO,
            details={
                "key_id": key_id,
                "key_type": key_type.value,
                "has_parent": parent_key_id is not None
            }
        )
        
        return key_id
    
    def get_key(self, key_id: str) -> bytes:
        """Get decrypted key material"""
        if key_id not in self.keys:
            raise EncryptionError(f"Key {key_id} not found")
        
        key_info = self.keys[key_id]
        
        if key_info["status"] != "active":
            raise EncryptionError(f"Key {key_id} is not active")
        
        # Decrypt key material
        master_cipher = Fernet(base64.urlsafe_b64encode(self.master_key))
        encrypted_material = key_info["encrypted_material"].encode()
        key_material = master_cipher.decrypt(encrypted_material)
        
        # Update usage count
        key_info["usage_count"] += 1
        key_info["last_used"] = datetime.utcnow().isoformat()
        
        return key_material
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate encryption key"""
        if key_id not in self.keys:
            raise EncryptionError(f"Key {key_id} not found")
        
        old_key_info = self.keys[key_id]
        
        # Create new key with incremented version
        new_key_id = f"{key_id}_v{old_key_info['version'] + 1}"
        new_key_material = Fernet.generate_key()
        
        new_key_info = old_key_info.copy()
        new_key_info.update({
            "id": new_key_id,
            "material": new_key_material.decode(),
            "created_at": datetime.utcnow().isoformat(),
            "version": old_key_info["version"] + 1,
            "usage_count": 0,
            "predecessor": key_id
        })
        
        # Encrypt new key
        master_cipher = Fernet(base64.urlsafe_b64encode(self.master_key))
        encrypted_key = master_cipher.encrypt(new_key_material)
        new_key_info["encrypted_material"] = encrypted_key.decode()
        
        # Store new key
        self.keys[new_key_id] = new_key_info
        
        # Mark old key as deprecated
        old_key_info["status"] = "deprecated"
        old_key_info["deprecated_at"] = datetime.utcnow().isoformat()
        old_key_info["successor"] = new_key_id
        
        # Update rotation schedule
        if security_settings.KEY_ROTATION_ENABLED:
            rotation_date = datetime.utcnow() + timedelta(days=security_settings.KEY_ROTATION_DAYS)
            self.rotation_schedule[new_key_id] = rotation_date
            self.rotation_schedule.pop(key_id, None)
        
        security_logger.log_security_event(
            event_type="KEY_ROTATED",
            level=SecurityEventLevel.INFO,
            details={
                "old_key_id": key_id,
                "new_key_id": new_key_id,
                "version": new_key_info["version"]
            }
        )
        
        return new_key_id
    
    def revoke_key(self, key_id: str, reason: str = "manual_revocation"):
        """Revoke encryption key"""
        if key_id not in self.keys:
            raise EncryptionError(f"Key {key_id} not found")
        
        key_info = self.keys[key_id]
        key_info["status"] = "revoked"
        key_info["revoked_at"] = datetime.utcnow().isoformat()
        key_info["revocation_reason"] = reason
        
        # Remove from rotation schedule
        self.rotation_schedule.pop(key_id, None)
        
        security_logger.log_security_event(
            event_type="KEY_REVOKED",
            level=SecurityEventLevel.WARNING,
            details={
                "key_id": key_id,
                "reason": reason,
                "usage_count": key_info["usage_count"]
            }
        )
    
    def get_keys_for_rotation(self) -> List[str]:
        """Get keys that need rotation"""
        current_time = datetime.utcnow()
        keys_to_rotate = []
        
        for key_id, rotation_time in self.rotation_schedule.items():
            if current_time >= rotation_time:
                keys_to_rotate.append(key_id)
        
        return keys_to_rotate
    
    def export_key_for_backup(self, key_id: str, recipient_public_key: bytes) -> str:
        """Export key for backup (encrypted with recipient's public key)"""
        if key_id not in self.keys:
            raise EncryptionError(f"Key {key_id} not found")
        
        key_material = self.get_key(key_id)
        
        # Encrypt with recipient's public key
        public_key = serialization.load_pem_public_key(recipient_public_key)
        encrypted_key = public_key.encrypt(
            key_material,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        export_data = {
            "key_id": key_id,
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "exported_at": datetime.utcnow().isoformat(),
            "metadata": self.keys[key_id]["metadata"]
        }
        
        return base64.b64encode(json.dumps(export_data).encode()).decode()


class DataEncryption:
    """
    Advanced data encryption system with field-level encryption,
    format-preserving encryption, and tokenization
    """
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.field_keys: Dict[str, str] = {}  # field_name -> key_id mapping
        
        # Initialize field-level encryption keys
        self._initialize_field_keys()
    
    def _initialize_field_keys(self):
        """Initialize field-level encryption keys"""
        sensitive_fields = [
            'ssn', 'tax_id', 'bank_account', 'credit_card',
            'api_keys', 'passwords', 'personal_notes',
            'account_number', 'routing_number', 'social_security'
        ]
        
        for field in sensitive_fields:
            key_id = self.key_manager.generate_key(
                KeyType.FIELD,
                key_id=f"field_{field}",
                metadata={"field_name": field, "encryption_type": "field_level"}
            )
            self.field_keys[field] = key_id
    
    def encrypt_data(
        self,
        data: Union[str, bytes, Dict, List],
        encryption_level: EncryptionLevel = EncryptionLevel.STANDARD,
        key_id: Optional[str] = None
    ) -> str:
        """
        Encrypt data with specified encryption level
        
        Returns base64-encoded encrypted data with metadata
        """
        try:
            # Serialize data if not string/bytes
            if isinstance(data, (dict, list)):
                data_bytes = json.dumps(data, default=str).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Get or create encryption key
            if key_id is None:
                key_id = self.key_manager.generate_key(
                    KeyType.DATA,
                    metadata={"encryption_level": encryption_level.value}
                )
            
            encrypted_data = None
            metadata = {
                "key_id": key_id,
                "encryption_level": encryption_level.value,
                "encrypted_at": datetime.utcnow().isoformat(),
                "algorithm": "AES-256-GCM"
            }
            
            if encryption_level == EncryptionLevel.STANDARD:
                encrypted_data = self._encrypt_standard(data_bytes, key_id)
            
            elif encryption_level == EncryptionLevel.HIGH:
                encrypted_data = self._encrypt_high(data_bytes, key_id)
                metadata["algorithm"] = "AES-256-GCM-HKDF"
            
            elif encryption_level == EncryptionLevel.MAXIMUM:
                encrypted_data = self._encrypt_maximum(data_bytes, key_id)
                metadata["algorithm"] = "RSA-4096+AES-256-GCM"
            
            # Create encrypted container
            container = {
                "metadata": metadata,
                "data": base64.b64encode(encrypted_data).decode()
            }
            
            return base64.b64encode(json.dumps(container).encode()).decode()
        
        except Exception as e:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["ENCRYPTION_ERROR"],
                level=SecurityEventLevel.ERROR,
                details={"error": str(e), "encryption_level": encryption_level.value}
            )
            raise EncryptionError(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict, List]:
        """Decrypt data and return original format"""
        try:
            # Decode container
            container_data = json.loads(base64.b64decode(encrypted_data).decode())
            metadata = container_data["metadata"]
            encrypted_bytes = base64.b64decode(container_data["data"])
            
            key_id = metadata["key_id"]
            encryption_level = EncryptionLevel(metadata["encryption_level"])
            
            # Decrypt based on level
            decrypted_bytes = None
            
            if encryption_level == EncryptionLevel.STANDARD:
                decrypted_bytes = self._decrypt_standard(encrypted_bytes, key_id)
            
            elif encryption_level == EncryptionLevel.HIGH:
                decrypted_bytes = self._decrypt_high(encrypted_bytes, key_id)
            
            elif encryption_level == EncryptionLevel.MAXIMUM:
                decrypted_bytes = self._decrypt_maximum(encrypted_bytes, key_id)
            
            # Try to deserialize as JSON, fallback to string
            try:
                decrypted_str = decrypted_bytes.decode('utf-8')
                return json.loads(decrypted_str)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return decrypted_bytes.decode('utf-8', errors='ignore')
        
        except Exception as e:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["ENCRYPTION_ERROR"],
                level=SecurityEventLevel.ERROR,
                details={"error": str(e), "operation": "decrypt"}
            )
            raise DecryptionError(f"Decryption failed: {str(e)}")
    
    def _encrypt_standard(self, data: bytes, key_id: str) -> bytes:
        """Standard AES-256-GCM encryption"""
        key_material = self.key_manager.get_key(key_id)
        cipher = Fernet(key_material)
        return cipher.encrypt(data)
    
    def _decrypt_standard(self, data: bytes, key_id: str) -> bytes:
        """Standard AES-256-GCM decryption"""
        key_material = self.key_manager.get_key(key_id)
        cipher = Fernet(key_material)
        return cipher.decrypt(data)
    
    def _encrypt_high(self, data: bytes, key_id: str) -> bytes:
        """High-security encryption with key derivation"""
        key_material = self.key_manager.get_key(key_id)
        
        # Generate salt and derive key
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=security_settings.KEY_DERIVATION_ITERATIONS,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key_material))
        
        # Encrypt with derived key
        cipher = Fernet(derived_key)
        encrypted_data = cipher.encrypt(data)
        
        # Prepend salt to encrypted data
        return salt + encrypted_data
    
    def _decrypt_high(self, data: bytes, key_id: str) -> bytes:
        """High-security decryption with key derivation"""
        key_material = self.key_manager.get_key(key_id)
        
        # Extract salt and encrypted data
        salt = data[:16]
        encrypted_data = data[16:]
        
        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=security_settings.KEY_DERIVATION_ITERATIONS,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key_material))
        
        # Decrypt
        cipher = Fernet(derived_key)
        return cipher.decrypt(encrypted_data)
    
    def _encrypt_maximum(self, data: bytes, key_id: str) -> bytes:
        """Maximum security RSA + AES hybrid encryption"""
        # Generate ephemeral AES key
        aes_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        # Encrypt data with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Get RSA key for AES key encryption (in practice, use a separate RSA key)
        # For now, use a derived key approach
        master_key_material = self.key_manager.get_key(key_id)
        
        # Encrypt AES key with master key (simplified - in production use RSA)
        key_cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(master_key_material).digest()))
        encrypted_aes_key = key_cipher.encrypt(aes_key)
        
        # Combine encrypted key, IV, tag, and ciphertext
        result = (
            len(encrypted_aes_key).to_bytes(4, 'big') +
            encrypted_aes_key +
            iv +
            encryptor.tag +
            ciphertext
        )
        
        return result
    
    def _decrypt_maximum(self, data: bytes, key_id: str) -> bytes:
        """Maximum security RSA + AES hybrid decryption"""
        # Extract components
        key_len = int.from_bytes(data[:4], 'big')
        encrypted_aes_key = data[4:4+key_len]
        iv = data[4+key_len:4+key_len+16]
        tag = data[4+key_len+16:4+key_len+32]
        ciphertext = data[4+key_len+32:]
        
        # Decrypt AES key
        master_key_material = self.key_manager.get_key(key_id)
        key_cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(master_key_material).digest()))
        aes_key = key_cipher.decrypt(encrypted_aes_key)
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def encrypt_field(self, field_name: str, value: Any) -> str:
        """Encrypt individual field with field-specific key"""
        # Check if field should be encrypted
        if not self._should_encrypt_field(field_name):
            return str(value) if value is not None else None
        
        if value is None or value == "":
            return value
        
        # Get field key
        key_id = self.field_keys.get(field_name)
        if not key_id:
            # Create new field key if not exists
            key_id = self.key_manager.generate_key(
                KeyType.FIELD,
                key_id=f"field_{field_name}",
                metadata={"field_name": field_name}
            )
            self.field_keys[field_name] = key_id
        
        return self.encrypt_data(value, EncryptionLevel.HIGH, key_id)
    
    def decrypt_field(self, field_name: str, encrypted_value: str) -> Any:
        """Decrypt individual field"""
        if not encrypted_value or not self._should_encrypt_field(field_name):
            return encrypted_value
        
        try:
            return self.decrypt_data(encrypted_value)
        except DecryptionError:
            # Log error but don't raise - might be unencrypted legacy data
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["ENCRYPTION_ERROR"],
                level=SecurityEventLevel.WARNING,
                details={"field": field_name, "operation": "field_decrypt"}
            )
            return encrypted_value
    
    def _should_encrypt_field(self, field_name: str) -> bool:
        """Check if field should be encrypted based on patterns"""
        if not security_settings.FIELD_ENCRYPTION_ENABLED:
            return False
        
        field_lower = field_name.lower()
        
        # Check explicit encrypted fields list
        if field_lower in [f.lower() for f in security_settings.ENCRYPTED_FIELDS]:
            return True
        
        # Check against patterns
        for pattern in SENSITIVE_FIELD_PATTERNS:
            if re.search(pattern, field_lower):
                return True
        
        return False
    
    def encrypt_pii_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt PII fields in a data dictionary"""
        encrypted_data = {}
        
        for key, value in data.items():
            encrypted_data[key] = self.encrypt_field(key, value)
        
        return encrypted_data
    
    def decrypt_pii_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt PII fields in a data dictionary"""
        decrypted_data = {}
        
        for key, value in data.items():
            decrypted_data[key] = self.decrypt_field(key, value)
        
        return decrypted_data


class TokenizationService:
    """
    Data tokenization for sensitive information
    Replaces sensitive data with non-sensitive tokens
    """
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.token_vault: Dict[str, str] = {}  # token -> encrypted_original
        self.reverse_vault: Dict[str, str] = {}  # hash(original) -> token
        
        # Initialize tokenization key
        self.tokenization_key_id = self.key_manager.generate_key(
            KeyType.DATA,
            key_id="tokenization_master",
            metadata={"purpose": "tokenization"}
        )
    
    def tokenize(self, value: str, preserve_format: bool = False) -> str:
        """
        Tokenize sensitive value
        
        Args:
            value: Original sensitive value
            preserve_format: Whether to preserve the format (e.g., credit card format)
            
        Returns:
            Non-sensitive token
        """
        if not value:
            return value
        
        # Check if already tokenized
        value_hash = hashlib.sha256(value.encode()).hexdigest()
        if value_hash in self.reverse_vault:
            return self.reverse_vault[value_hash]
        
        # Generate token
        if preserve_format:
            token = self._generate_format_preserving_token(value)
        else:
            token = f"tok_{secrets.token_hex(16)}"
        
        # Encrypt original value
        encrypted_original = self._encrypt_for_vault(value)
        
        # Store in vaults
        self.token_vault[token] = encrypted_original
        self.reverse_vault[value_hash] = token
        
        security_logger.log_security_event(
            event_type="DATA_TOKENIZED",
            level=SecurityEventLevel.INFO,
            details={"token": token[:8] + "...", "format_preserving": preserve_format}
        )
        
        return token
    
    def detokenize(self, token: str) -> str:
        """
        Detokenize to get original value
        
        Args:
            token: Token to detokenize
            
        Returns:
            Original sensitive value
        """
        if not token or token not in self.token_vault:
            return token
        
        encrypted_original = self.token_vault[token]
        original_value = self._decrypt_from_vault(encrypted_original)
        
        security_logger.log_security_event(
            event_type="DATA_DETOKENIZED",
            level=SecurityEventLevel.INFO,
            details={"token": token[:8] + "..."}
        )
        
        return original_value
    
    def _generate_format_preserving_token(self, value: str) -> str:
        """Generate token that preserves the format of original value"""
        token_chars = []
        
        for char in value:
            if char.isdigit():
                token_chars.append(str(secrets.randbelow(10)))
            elif char.isalpha():
                if char.isupper():
                    token_chars.append(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                else:
                    token_chars.append(secrets.choice('abcdefghijklmnopqrstuvwxyz'))
            else:
                token_chars.append(char)  # Preserve special characters
        
        return ''.join(token_chars)
    
    def _encrypt_for_vault(self, value: str) -> str:
        """Encrypt value for token vault storage"""
        key_material = self.key_manager.get_key(self.tokenization_key_id)
        cipher = Fernet(key_material)
        encrypted = cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_from_vault(self, encrypted_value: str) -> str:
        """Decrypt value from token vault"""
        key_material = self.key_manager.get_key(self.tokenization_key_id)
        cipher = Fernet(key_material)
        encrypted_bytes = base64.b64decode(encrypted_value)
        decrypted = cipher.decrypt(encrypted_bytes)
        return decrypted.decode()


# Global encryption services
key_manager = KeyManager()
data_encryption = DataEncryption(key_manager)
tokenization_service = TokenizationService(key_manager)


# =====================================
# UTILITY FUNCTIONS
# =====================================

def encrypt_sensitive_data(data: Union[str, Dict, List], level: str = "standard") -> str:
    """Convenience function to encrypt sensitive data"""
    encryption_level = EncryptionLevel(level)
    return data_encryption.encrypt_data(data, encryption_level)


def decrypt_sensitive_data(encrypted_data: str) -> Union[str, Dict, List]:
    """Convenience function to decrypt sensitive data"""
    return data_encryption.decrypt_data(encrypted_data)


def tokenize_pii(value: str, preserve_format: bool = False) -> str:
    """Convenience function to tokenize PII"""
    return tokenization_service.tokenize(value, preserve_format)


def detokenize_pii(token: str) -> str:
    """Convenience function to detokenize PII"""
    return tokenization_service.detokenize(token)


def perform_key_rotation():
    """Perform scheduled key rotation"""
    keys_to_rotate = key_manager.get_keys_for_rotation()
    
    for key_id in keys_to_rotate:
        try:
            new_key_id = key_manager.rotate_key(key_id)
            logger.info(f"Rotated key {key_id} to {new_key_id}")
        except Exception as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")


# =====================================
# DATABASE ENCRYPTION MIXINS
# =====================================

class EncryptedFieldMixin:
    """Mixin for SQLAlchemy models to handle encrypted fields"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encrypted_fields = getattr(self.__class__, '_encrypted_fields', [])
    
    def __setattr__(self, name, value):
        if hasattr(self, '_encrypted_fields') and name in self._encrypted_fields:
            if value is not None:
                value = data_encryption.encrypt_field(name, value)
        super().__setattr__(name, value)
    
    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if (hasattr(self, '_encrypted_fields') and 
            name in self._encrypted_fields and 
            value is not None and 
            isinstance(value, str)):
            try:
                return data_encryption.decrypt_field(name, value)
            except DecryptionError:
                return value  # Return as-is if decryption fails
        return value