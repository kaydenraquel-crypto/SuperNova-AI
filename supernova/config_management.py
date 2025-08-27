"""
SuperNova AI - Comprehensive Configuration Management System

This module provides a comprehensive configuration management framework that supports:
- Environment-specific configurations (dev, staging, prod)
- Secure secrets management with encryption and rotation
- Configuration validation and type checking with dependency checking
- Hot-reloading for dynamic settings
- Configuration versioning and rollback capabilities
- Integration with cloud secret stores (AWS, Azure, HashiCorp Vault)
- Configuration monitoring and drift detection
- Deployment configuration templates
- Configuration testing and validation suites
"""

from __future__ import annotations
import os
import json
import logging
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from contextlib import contextmanager

from pydantic import BaseSettings, Field, validator, root_validator
from cryptography.fernet import Fernet
import yaml
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .security_config import security_settings

# Cloud integrations
try:
    import boto3
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


class Environment(str, Enum):
    """Environment types for configuration management."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationLevel(str, Enum):
    """Configuration security levels."""
    PUBLIC = "public"          # Non-sensitive settings
    INTERNAL = "internal"      # Internal application settings
    CONFIDENTIAL = "confidential"  # Sensitive business logic
    SECRET = "secret"          # API keys, passwords, etc.


class ConfigurationSource(str, Enum):
    """Configuration data sources."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    CLOUD_SECRET = "cloud_secret"
    VAULT = "vault"
    CONSUL = "consul"


@dataclass
class ConfigurationMetadata:
    """Metadata for configuration entries."""
    source: ConfigurationSource
    level: ConfigurationLevel
    last_updated: datetime
    version: str
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    hot_reloadable: bool = False
    encrypted: bool = False
    audit_log: List[Dict[str, Any]] = field(default_factory=list)


class ConfigurationChangeEvent:
    """Event for configuration changes."""
    
    def __init__(self, key: str, old_value: Any, new_value: Any, metadata: ConfigurationMetadata):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.metadata = metadata
        self.timestamp = datetime.utcnow()


class ConfigurationValidator(ABC):
    """Abstract base class for configuration validators."""
    
    @abstractmethod
    def validate(self, key: str, value: Any, metadata: ConfigurationMetadata) -> tuple[bool, Optional[str]]:
        """Validate a configuration value."""
        pass


class TypeValidator(ConfigurationValidator):
    """Type-based configuration validator."""
    
    def __init__(self, expected_type: Type):
        self.expected_type = expected_type
    
    def validate(self, key: str, value: Any, metadata: ConfigurationMetadata) -> tuple[bool, Optional[str]]:
        if not isinstance(value, self.expected_type):
            return False, f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
        return True, None


class RangeValidator(ConfigurationValidator):
    """Range-based validator for numeric values."""
    
    def __init__(self, min_val: Union[int, float] = None, max_val: Union[int, float] = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, key: str, value: Any, metadata: ConfigurationMetadata) -> tuple[bool, Optional[str]]:
        if not isinstance(value, (int, float)):
            return False, "Value must be numeric for range validation"
        
        if self.min_val is not None and value < self.min_val:
            return False, f"Value {value} is below minimum {self.min_val}"
        
        if self.max_val is not None and value > self.max_val:
            return False, f"Value {value} is above maximum {self.max_val}"
        
        return True, None


class RegexValidator(ConfigurationValidator):
    """Regular expression validator."""
    
    def __init__(self, pattern: str):
        import re
        self.pattern = re.compile(pattern)
    
    def validate(self, key: str, value: Any, metadata: ConfigurationMetadata) -> tuple[bool, Optional[str]]:
        if not isinstance(value, str):
            return False, "Value must be string for regex validation"
        
        if not self.pattern.match(value):
            return False, f"Value does not match pattern {self.pattern.pattern}"
        
        return True, None


class DependencyValidator(ConfigurationValidator):
    """Dependency validator to ensure required configurations are present."""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
    
    def validate(self, key: str, value: Any, metadata: ConfigurationMetadata) -> tuple[bool, Optional[str]]:
        missing_deps = []
        
        for dep in metadata.dependencies:
            if not self.config_manager.has_configuration(dep):
                missing_deps.append(dep)
        
        if missing_deps:
            return False, f"Missing dependencies: {', '.join(missing_deps)}"
        
        return True, None


class SecretEncryption:
    """Handles encryption and decryption of sensitive configuration data."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self._cipher = Fernet(encryption_key.encode())
        else:
            self._cipher = Fernet(Fernet.generate_key())
    
    def encrypt(self, value: str) -> str:
        """Encrypt a string value."""
        return self._cipher.encrypt(value.encode()).decode()
    
    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string value."""
        return self._cipher.decrypt(encrypted_value.encode()).decode()
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted."""
        try:
            self._cipher.decrypt(value.encode())
            return True
        except Exception:
            return False


class CloudSecretManager(ABC):
    """Abstract base class for cloud secret management."""
    
    @abstractmethod
    async def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret."""
        pass
    
    @abstractmethod
    async def set_secret(self, key: str, value: str) -> bool:
        """Store a secret."""
        pass
    
    @abstractmethod
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List available secrets."""
        pass


class AWSSecretsManager(CloudSecretManager):
    """AWS Secrets Manager integration."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS Secrets Manager")
        
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    async def get_secret(self, key: str) -> Optional[str]:
        try:
            response = self.client.get_secret_value(SecretId=key)
            return response['SecretString']
        except Exception as e:
            logger.error(f"Failed to get secret {key} from AWS: {e}")
            return None
    
    async def set_secret(self, key: str, value: str) -> bool:
        try:
            try:
                # Try to update existing secret
                self.client.update_secret(SecretId=key, SecretString=value)
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret
                self.client.create_secret(Name=key, SecretString=value)
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key} in AWS: {e}")
            return False
    
    async def delete_secret(self, key: str) -> bool:
        try:
            self.client.delete_secret(SecretId=key, ForceDeleteWithoutRecovery=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key} from AWS: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        try:
            response = self.client.list_secrets()
            return [secret['Name'] for secret in response['SecretList']]
        except Exception as e:
            logger.error(f"Failed to list secrets from AWS: {e}")
            return []


class AzureKeyVault(CloudSecretManager):
    """Azure Key Vault integration."""
    
    def __init__(self, vault_url: str):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-keyvault-secrets is required for Azure Key Vault")
        
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
    
    async def get_secret(self, key: str) -> Optional[str]:
        try:
            secret = self.client.get_secret(key)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to get secret {key} from Azure: {e}")
            return None
    
    async def set_secret(self, key: str, value: str) -> bool:
        try:
            self.client.set_secret(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key} in Azure: {e}")
            return False
    
    async def delete_secret(self, key: str) -> bool:
        try:
            self.client.begin_delete_secret(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key} from Azure: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]
        except Exception as e:
            logger.error(f"Failed to list secrets from Azure: {e}")
            return []


class HashiCorpVault(CloudSecretManager):
    """HashiCorp Vault integration."""
    
    def __init__(self, vault_url: str, token: str):
        if not VAULT_AVAILABLE:
            raise ImportError("hvac is required for HashiCorp Vault")
        
        self.client = hvac.Client(url=vault_url, token=token)
    
    async def get_secret(self, key: str) -> Optional[str]:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=key)
            return response['data']['data'].get('value')
        except Exception as e:
            logger.error(f"Failed to get secret {key} from Vault: {e}")
            return None
    
    async def set_secret(self, key: str, value: str) -> bool:
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=key,
                secret={'value': value}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key} in Vault: {e}")
            return False
    
    async def delete_secret(self, key: str) -> bool:
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(path=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key} from Vault: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        try:
            response = self.client.secrets.kv.v2.list_secrets(path='')
            return response['data']['keys']
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []


class ConfigurationWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes."""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        self.last_modified = {}
        self.debounce_time = 1.0  # seconds
    
    def on_modified(self, event):
        if not event.is_directory:
            current_time = datetime.utcnow().timestamp()
            
            # Debounce rapid changes
            if event.src_path in self.last_modified:
                if current_time - self.last_modified[event.src_path] < self.debounce_time:
                    return
            
            self.last_modified[event.src_path] = current_time
            
            # Schedule reload for configuration files
            if any(event.src_path.endswith(ext) for ext in ['.env', '.yaml', '.yml', '.json']):
                asyncio.create_task(self.config_manager._handle_file_change(event.src_path))


class ConfigurationManager:
    """Main configuration management class."""
    
    def __init__(
        self,
        environment: Environment = Environment.DEVELOPMENT,
        config_dir: Optional[Path] = None,
        encryption_key: Optional[str] = None,
        enable_cloud_secrets: bool = False
    ):
        self.environment = environment
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._encryption = SecretEncryption(encryption_key)
        self._cloud_secrets: Optional[CloudSecretManager] = None
        self._validators: Dict[str, List[ConfigurationValidator]] = {}
        self._change_listeners: List[Callable[[ConfigurationChangeEvent], None]] = []
        self._cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, ConfigurationMetadata] = {}
        
        # Configuration storage
        self._configurations: Dict[str, Any] = {}
        self._metadata: Dict[str, ConfigurationMetadata] = {}
        
        # Versioning
        self._versions: Dict[str, List[dict]] = {}
        self._current_version = "1.0.0"
        
        # Hot reload support
        self._hot_reload_enabled = True
        self._observer: Optional[Observer] = None
        self._lock = threading.RLock()
        
        # Drift detection
        self._last_config_hash: Optional[str] = None
        self._drift_check_interval = 300  # 5 minutes
        
        # Initialize
        self._initialize_cloud_secrets(enable_cloud_secrets)
        self._setup_file_watcher()
        self._load_initial_configuration()
    
    def _initialize_cloud_secrets(self, enable_cloud_secrets: bool):
        """Initialize cloud secret management."""
        if not enable_cloud_secrets:
            return
        
        try:
            # AWS Secrets Manager
            aws_region = os.getenv('AWS_REGION')
            if aws_region and BOTO3_AVAILABLE:
                self._cloud_secrets = AWSSecretsManager(aws_region)
                logger.info("AWS Secrets Manager initialized")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize AWS Secrets Manager: {e}")
        
        try:
            # Azure Key Vault
            vault_url = os.getenv('AZURE_VAULT_URL')
            if vault_url and AZURE_AVAILABLE:
                self._cloud_secrets = AzureKeyVault(vault_url)
                logger.info("Azure Key Vault initialized")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize Azure Key Vault: {e}")
        
        try:
            # HashiCorp Vault
            vault_url = os.getenv('VAULT_URL')
            vault_token = os.getenv('VAULT_TOKEN')
            if vault_url and vault_token and VAULT_AVAILABLE:
                self._cloud_secrets = HashiCorpVault(vault_url, vault_token)
                logger.info("HashiCorp Vault initialized")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize HashiCorp Vault: {e}")
    
    def _setup_file_watcher(self):
        """Setup file system watcher for configuration changes."""
        if not self._hot_reload_enabled:
            return
        
        try:
            self._observer = Observer()
            watcher = ConfigurationWatcher(self)
            self._observer.schedule(watcher, str(self.config_dir), recursive=True)
            self._observer.schedule(watcher, ".", recursive=False)  # Watch for .env files
            self._observer.start()
            logger.info("Configuration file watcher started")
        except Exception as e:
            logger.warning(f"Failed to setup file watcher: {e}")
    
    async def _handle_file_change(self, file_path: str):
        """Handle configuration file changes."""
        try:
            logger.info(f"Configuration file changed: {file_path}")
            
            # Reload configuration
            await self._reload_from_source()
            
            # Check for drift
            await self._check_configuration_drift()
            
        except Exception as e:
            logger.error(f"Error handling file change: {e}")
    
    def _load_initial_configuration(self):
        """Load initial configuration from all sources."""
        try:
            # Load from environment variables
            self._load_from_environment()
            
            # Load from configuration files
            self._load_from_files()
            
            # Setup default validators
            self._setup_default_validators()
            
            logger.info(f"Loaded {len(self._configurations)} configuration items")
            
        except Exception as e:
            logger.error(f"Failed to load initial configuration: {e}")
            raise
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Load from .env files
        env_files = [
            f".env.{self.environment.value}",
            f".env.{self.environment.value}.local",
            ".env.local",
            ".env"
        ]
        
        for env_file in env_files:
            if os.path.exists(env_file):
                from dotenv import load_dotenv
                load_dotenv(env_file, override=False)
                logger.debug(f"Loaded environment from {env_file}")
        
        # Process environment variables
        for key, value in os.environ.items():
            if key.startswith('SUPERNOVA_'):
                config_key = key.lower()
                self._configurations[config_key] = value
                self._metadata[config_key] = ConfigurationMetadata(
                    source=ConfigurationSource.ENVIRONMENT,
                    level=ConfigurationLevel.INTERNAL,
                    last_updated=datetime.utcnow(),
                    version=self._current_version,
                    description=f"Environment variable: {key}"
                )
    
    def _load_from_files(self):
        """Load configuration from files."""
        config_files = [
            self.config_dir / f"{self.environment.value}.yaml",
            self.config_dir / f"{self.environment.value}.yml",
            self.config_dir / f"{self.environment.value}.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            self.config_dir / "config.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                self._load_config_file(config_file)
    
    def _load_config_file(self, config_file: Path):
        """Load configuration from a specific file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    data = json.load(f)
                else:
                    return
            
            if not isinstance(data, dict):
                return
            
            # Process configuration data
            for key, value in data.items():
                config_key = key.lower()
                self._configurations[config_key] = value
                
                # Determine configuration level based on key patterns
                level = self._determine_config_level(key, value)
                
                self._metadata[config_key] = ConfigurationMetadata(
                    source=ConfigurationSource.FILE,
                    level=level,
                    last_updated=datetime.utcnow(),
                    version=self._current_version,
                    description=f"From file: {config_file.name}",
                    encrypted=level == ConfigurationLevel.SECRET
                )
            
            logger.debug(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_file}: {e}")
    
    def _determine_config_level(self, key: str, value: Any) -> ConfigurationLevel:
        """Determine configuration security level based on key and value."""
        key_lower = key.lower()
        
        # Secret patterns
        secret_patterns = [
            'api_key', 'secret', 'password', 'token', 'credential',
            'private_key', 'certificate', 'auth', 'oauth'
        ]
        
        if any(pattern in key_lower for pattern in secret_patterns):
            return ConfigurationLevel.SECRET
        
        # Confidential patterns
        confidential_patterns = [
            'database_url', 'connection_string', 'host', 'endpoint'
        ]
        
        if any(pattern in key_lower for pattern in confidential_patterns):
            return ConfigurationLevel.CONFIDENTIAL
        
        # Internal patterns
        internal_patterns = [
            'config', 'setting', 'option', 'parameter'
        ]
        
        if any(pattern in key_lower for pattern in internal_patterns):
            return ConfigurationLevel.INTERNAL
        
        return ConfigurationLevel.PUBLIC
    
    def _setup_default_validators(self):
        """Setup default configuration validators."""
        # Temperature validator
        self.add_validator('llm_temperature', RangeValidator(0.0, 2.0))
        
        # Timeout validators
        self.add_validator('llm_timeout', RangeValidator(1, 600))
        self.add_validator('connection_timeout', RangeValidator(1, 300))
        
        # Retry validators
        self.add_validator('llm_max_retries', RangeValidator(0, 10))
        
        # Cost limit validators
        self.add_validator('llm_daily_cost_limit', RangeValidator(0.01, 10000.0))
        
        # URL validators
        url_pattern = r'^https?://.*'
        self.add_validator('openai_base_url', RegexValidator(url_pattern))
        self.add_validator('anthropic_base_url', RegexValidator(url_pattern))
        
        # Email validators
        email_pattern = r'^[^@]+@[^@]+\.[^@]+$'
        self.add_validator('notification_email', RegexValidator(email_pattern))
    
    def add_validator(self, key: str, validator: ConfigurationValidator):
        """Add a validator for a configuration key."""
        if key not in self._validators:
            self._validators[key] = []
        self._validators[key].append(validator)
    
    def add_change_listener(self, listener: Callable[[ConfigurationChangeEvent], None]):
        """Add a change listener."""
        self._change_listeners.append(listener)
    
    async def set_configuration(
        self,
        key: str,
        value: Any,
        level: ConfigurationLevel = ConfigurationLevel.PUBLIC,
        source: ConfigurationSource = ConfigurationSource.FILE,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        hot_reloadable: bool = False,
        encrypt: bool = False
    ) -> bool:
        """Set a configuration value with metadata."""
        
        with self._lock:
            # Create metadata
            metadata = ConfigurationMetadata(
                source=source,
                level=level,
                last_updated=datetime.utcnow(),
                version=self._current_version,
                description=description,
                tags=tags or set(),
                dependencies=dependencies or set(),
                hot_reloadable=hot_reloadable,
                encrypted=encrypt
            )
            
            # Validate configuration
            validation_result = await self._validate_configuration(key, value, metadata)
            if not validation_result[0]:
                logger.error(f"Configuration validation failed for {key}: {validation_result[1]}")
                return False
            
            # Get old value for change event
            old_value = self._configurations.get(key)
            
            # Encrypt value if needed
            stored_value = value
            if encrypt and isinstance(value, str):
                stored_value = self._encryption.encrypt(value)
                metadata.encrypted = True
            
            # Store in cloud if it's a secret
            if level == ConfigurationLevel.SECRET and self._cloud_secrets:
                try:
                    await self._cloud_secrets.set_secret(key, str(value))
                    logger.info(f"Secret {key} stored in cloud")
                except Exception as e:
                    logger.warning(f"Failed to store secret in cloud: {e}")
            
            # Update local storage
            self._configurations[key] = stored_value
            self._metadata[key] = metadata
            self._cache[key] = value  # Cache decrypted value
            
            # Version tracking
            if key not in self._versions:
                self._versions[key] = []
            
            self._versions[key].append({
                'value': stored_value,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Keep only last 10 versions per key
            if len(self._versions[key]) > 10:
                self._versions[key].pop(0)
            
            # Add to audit log
            metadata.audit_log.append({
                'action': 'set',
                'timestamp': datetime.utcnow().isoformat(),
                'old_value': old_value,
                'new_value': value,
                'user': os.getenv('USER', 'system')
            })
            
            # Trigger change event
            if old_value != value:
                event = ConfigurationChangeEvent(key, old_value, value, metadata)
                for listener in self._change_listeners:
                    try:
                        listener(event)
                    except Exception as e:
                        logger.error(f"Error in change listener: {e}")
            
            logger.info(f"Configuration set: {key} (level: {level.value})")
            return True
    
    async def get_configuration(
        self,
        key: str,
        default: Any = None,
        decrypt: bool = True
    ) -> Any:
        """Get a configuration value."""
        
        with self._lock:
            # Check cache first
            if key in self._cache:
                return self._cache[key]
            
            # Check local storage
            if key in self._configurations:
                value = self._configurations[key]
                metadata = self._metadata.get(key)
                
                # Decrypt if needed
                if metadata and metadata.encrypted and decrypt:
                    if isinstance(value, str) and self._encryption.is_encrypted(value):
                        value = self._encryption.decrypt(value)
                
                # Cache the value
                self._cache[key] = value
                return value
            
            # Check cloud secrets for secret-level configurations
            if self._cloud_secrets:
                try:
                    cloud_value = await self._cloud_secrets.get_secret(key)
                    if cloud_value is not None:
                        self._cache[key] = cloud_value
                        return cloud_value
                except Exception as e:
                    logger.debug(f"Could not retrieve {key} from cloud: {e}")
            
            return default
    
    async def _validate_configuration(
        self,
        key: str,
        value: Any,
        metadata: ConfigurationMetadata
    ) -> tuple[bool, Optional[str]]:
        """Validate a configuration value."""
        
        # Check validators
        if key in self._validators:
            for validator in self._validators[key]:
                result = validator.validate(key, value, metadata)
                if not result[0]:
                    return result
        
        # Check dependencies
        dependency_validator = DependencyValidator(self)
        return dependency_validator.validate(key, value, metadata)
    
    def has_configuration(self, key: str) -> bool:
        """Check if a configuration exists."""
        return key in self._configurations or key in self._cache
    
    async def delete_configuration(self, key: str) -> bool:
        """Delete a configuration."""
        
        with self._lock:
            try:
                # Remove from cloud if it was stored there
                metadata = self._metadata.get(key)
                if metadata and metadata.level == ConfigurationLevel.SECRET and self._cloud_secrets:
                    try:
                        await self._cloud_secrets.delete_secret(key)
                    except Exception as e:
                        logger.warning(f"Failed to delete secret from cloud: {e}")
                
                # Remove locally
                self._configurations.pop(key, None)
                self._metadata.pop(key, None)
                self._cache.pop(key, None)
                self._versions.pop(key, None)
                
                logger.info(f"Configuration deleted: {key}")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting configuration {key}: {e}")
                return False
    
    async def list_configurations(
        self,
        level: Optional[ConfigurationLevel] = None,
        source: Optional[ConfigurationSource] = None,
        tags: Optional[Set[str]] = None
    ) -> List[str]:
        """List configuration keys with optional filtering."""
        
        filtered_keys = []
        
        for key, metadata in self._metadata.items():
            # Level filter
            if level and metadata.level != level:
                continue
            
            # Source filter
            if source and metadata.source != source:
                continue
            
            # Tags filter
            if tags and not tags.intersection(metadata.tags):
                continue
            
            filtered_keys.append(key)
        
        return sorted(filtered_keys)
    
    async def get_configuration_metadata(self, key: str) -> Optional[ConfigurationMetadata]:
        """Get metadata for a configuration key."""
        return self._metadata.get(key)
    
    async def rollback_configuration(self, key: str, version_index: int = -2) -> bool:
        """Rollback a configuration to a previous version."""
        
        history = self._versions.get(key, [])
        if not history or abs(version_index) > len(history):
            logger.error(f"Invalid version index for configuration {key}")
            return False
        
        try:
            previous_version = history[version_index]
            metadata = previous_version['metadata']
            value = previous_version['value']
            
            # Decrypt if needed
            if metadata.encrypted and isinstance(value, str):
                if self._encryption.is_encrypted(value):
                    value = self._encryption.decrypt(value)
            
            # Set the configuration
            success = await self.set_configuration(
                key=key,
                value=value,
                level=metadata.level,
                source=metadata.source,
                description=f"Rolled back from version {len(history)}",
                tags=metadata.tags,
                dependencies=metadata.dependencies,
                hot_reloadable=metadata.hot_reloadable,
                encrypt=metadata.encrypted
            )
            
            if success:
                logger.info(f"Configuration {key} rolled back to version {version_index}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rolling back configuration {key}: {e}")
            return False
    
    async def _reload_from_source(self):
        """Reload configuration from all sources."""
        try:
            # Backup current state
            backup_config = self._configurations.copy()
            backup_metadata = self._metadata.copy()
            
            # Clear current state
            self._configurations.clear()
            self._metadata.clear()
            self._cache.clear()
            
            # Reload
            self._load_initial_configuration()
            
            # Find changes
            changed_keys = []
            for key in set(backup_config.keys()) | set(self._configurations.keys()):
                if backup_config.get(key) != self._configurations.get(key):
                    changed_keys.append(key)
            
            if changed_keys:
                logger.info(f"Configuration reloaded. Changed keys: {changed_keys}")
                
                # Trigger change events
                for key in changed_keys:
                    metadata = self._metadata.get(key)
                    if metadata:
                        event = ConfigurationChangeEvent(
                            key,
                            backup_config.get(key),
                            self._configurations.get(key),
                            metadata
                        )
                        
                        for listener in self._change_listeners:
                            try:
                                listener(event)
                            except Exception as e:
                                logger.error(f"Error in change listener: {e}")
        
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            # Restore backup on error
            self._configurations = backup_config
            self._metadata = backup_metadata
    
    async def _check_configuration_drift(self):
        """Check for configuration drift."""
        try:
            current_hash = self._calculate_config_hash()
            
            if self._last_config_hash and self._last_config_hash != current_hash:
                logger.warning("Configuration drift detected!")
                
                # Trigger drift detection event
                for listener in self._change_listeners:
                    try:
                        # Create a special drift event
                        event = ConfigurationChangeEvent(
                            "__drift_detected__",
                            self._last_config_hash,
                            current_hash,
                            ConfigurationMetadata(
                                source=ConfigurationSource.FILE,
                                level=ConfigurationLevel.INTERNAL,
                                last_updated=datetime.utcnow(),
                                version=self._current_version,
                                description="Configuration drift detected"
                            )
                        )
                        listener(event)
                    except Exception as e:
                        logger.error(f"Error in drift detection listener: {e}")
            
            self._last_config_hash = current_hash
            
        except Exception as e:
            logger.error(f"Error checking configuration drift: {e}")
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration for drift detection."""
        config_str = json.dumps(self._configurations, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def export_configuration(
        self,
        output_path: Path,
        include_secrets: bool = False,
        encrypt_export: bool = True,
        format: str = 'yaml'
    ) -> bool:
        """Export configuration to a file."""
        
        try:
            export_data = {
                'environment': self.environment.value,
                'version': self._current_version,
                'exported_at': datetime.utcnow().isoformat(),
                'configurations': {},
                'metadata': {}
            }
            
            for key, value in self._configurations.items():
                metadata = self._metadata.get(key)
                if not metadata:
                    continue
                
                # Skip secrets if not included
                if not include_secrets and metadata.level == ConfigurationLevel.SECRET:
                    continue
                
                export_data['configurations'][key] = value
                export_data['metadata'][key] = {
                    'source': metadata.source.value,
                    'level': metadata.level.value,
                    'last_updated': metadata.last_updated.isoformat(),
                    'version': metadata.version,
                    'description': metadata.description,
                    'tags': list(metadata.tags),
                    'dependencies': list(metadata.dependencies),
                    'hot_reloadable': metadata.hot_reloadable,
                    'encrypted': metadata.encrypted
                }
            
            # Write to file
            if format.lower() == 'yaml':
                content = yaml.dump(export_data, default_flow_style=False)
            else:
                content = json.dumps(export_data, indent=2)
            
            if encrypt_export:
                content = self._encryption.encrypt(content)
                output_path = output_path.with_suffix('.encrypted')
            
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(content)
            
            logger.info(f"Configuration exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    async def import_configuration(
        self,
        input_path: Path,
        merge: bool = True,
        decrypt_import: bool = True
    ) -> bool:
        """Import configuration from a file."""
        
        try:
            async with aiofiles.open(input_path, 'r') as f:
                content = await f.read()
            
            # Decrypt if needed
            if input_path.suffix == '.encrypted' and decrypt_import:
                content = self._encryption.decrypt(content)
            
            # Parse content
            if input_path.name.endswith('.yaml') or input_path.name.endswith('.yml'):
                import_data = yaml.safe_load(content)
            else:
                import_data = json.loads(content)
            
            configurations = import_data.get('configurations', {})
            metadata_dict = import_data.get('metadata', {})
            
            success_count = 0
            
            for key, value in configurations.items():
                # Skip if not merging and key exists
                if not merge and self.has_configuration(key):
                    continue
                
                meta = metadata_dict.get(key, {})
                
                success = await self.set_configuration(
                    key=key,
                    value=value,
                    level=ConfigurationLevel(meta.get('level', 'public')),
                    source=ConfigurationSource(meta.get('source', 'file')),
                    description=meta.get('description'),
                    tags=set(meta.get('tags', [])),
                    dependencies=set(meta.get('dependencies', [])),
                    hot_reloadable=meta.get('hot_reloadable', False),
                    encrypt=meta.get('encrypted', False)
                )
                
                if success:
                    success_count += 1
            
            logger.info(f"Imported {success_count} configurations from {input_path}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    async def get_configuration_health(self) -> Dict[str, Any]:
        """Get health status of the configuration system."""
        
        health = {
            'environment': self.environment.value,
            'total_configurations': len(self._configurations),
            'cache_size': len(self._cache),
            'hot_reload_enabled': self._hot_reload_enabled,
            'encryption_enabled': True,
            'cloud_secrets_enabled': self._cloud_secrets is not None,
            'file_watcher_active': self._observer is not None and self._observer.is_alive(),
            'issues': []
        }
        
        # Check for missing dependencies
        for key, metadata in self._metadata.items():
            missing_deps = []
            for dep in metadata.dependencies:
                if not self.has_configuration(dep):
                    missing_deps.append(dep)
            
            if missing_deps:
                health['issues'].append(f"Configuration {key} has missing dependencies: {missing_deps}")
        
        # Check for configuration drift
        if self._last_config_hash:
            current_hash = self._calculate_config_hash()
            if current_hash != self._last_config_hash:
                health['issues'].append("Configuration drift detected")
        
        return health
    
    def shutdown(self):
        """Shutdown the configuration manager."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        logger.info("Configuration manager shut down")
    
    @contextmanager
    def configuration_context(self, **overrides):
        """Context manager for temporary configuration overrides."""
        original_values = {}
        
        try:
            # Set overrides and store original values
            for key, value in overrides.items():
                original_values[key] = self._cache.get(key)
                self._cache[key] = value
            
            yield self
            
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    self._cache.pop(key, None)
                else:
                    self._cache[key] = original_value


# Global configuration manager instance
config_manager = ConfigurationManager(
    environment=Environment(os.getenv('SUPERNOVA_ENV', 'development')),
    encryption_key=os.getenv('CONFIG_ENCRYPTION_KEY'),
    enable_cloud_secrets=os.getenv('ENABLE_CLOUD_SECRETS', 'false').lower() == 'true'
)


# Convenience functions
async def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return await config_manager.get_configuration(key, default)


async def set_config(key: str, value: Any, **kwargs) -> bool:
    """Set a configuration value."""
    return await config_manager.set_configuration(key, value, **kwargs)


def get_config_sync(key: str, default: Any = None) -> Any:
    """Synchronous configuration getter for backwards compatibility."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, use cached value
            return config_manager._cache.get(key, default)
        else:
            return loop.run_until_complete(config_manager.get_configuration(key, default))
    except RuntimeError:
        # No event loop, use cached value
        return config_manager._cache.get(key, default)


# Configuration health check
async def check_configuration_health() -> Dict[str, Any]:
    """Check configuration system health."""
    return await config_manager.get_configuration_health()