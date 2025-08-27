"""
Enhanced Configuration Management with Validation and Hot Reloading

This module provides:
- Configuration validation and type checking
- Hot configuration reloading without restart
- Environment-specific configuration management
- Configuration change monitoring and notifications
- Secure configuration backup and recovery
- Configuration versioning and rollback
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import os
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Core imports
from .config import settings, Settings
from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, ValidationError, validator
from typing_extensions import Literal

logger = logging.getLogger(__name__)

class ConfigEnvironment(str, Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ConfigValidationLevel(str, Enum):
    """Configuration validation levels"""
    STRICT = "strict"      # All validations must pass
    RELAXED = "relaxed"    # Warnings for failures, continue
    DISABLED = "disabled"  # No validation

@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    changed_keys: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)

class LLMProviderConfig(BaseModel):
    """LLM provider configuration validation model"""
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0 or v > 600:
            raise ValueError('Timeout must be between 1 and 600 seconds')
        return v
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Max retries must be between 0 and 10')
        return v

class CostTrackingConfig(BaseModel):
    """Cost tracking configuration validation model"""
    enabled: bool = True
    daily_limit: float = 50.0
    monthly_limit: float = 1000.0
    alert_threshold: float = 0.8
    
    @validator('daily_limit', 'monthly_limit')
    def validate_limits(cls, v):
        if v <= 0:
            raise ValueError('Cost limits must be positive')
        return v
    
    @validator('alert_threshold')
    def validate_threshold(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Alert threshold must be between 0 and 1')
        return v

class SuperNovaConfig(BaseModel):
    """Main SuperNova configuration validation model"""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # LLM Configuration
    llm_enabled: bool = True
    llm_provider: str = "openai"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4000
    
    # Database
    database_url: str = "sqlite:///./supernova.db"
    
    # Security
    api_key_encryption_enabled: bool = True
    
    @validator('llm_temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v
    
    @validator('llm_max_tokens')
    def validate_max_tokens(cls, v):
        if v <= 0 or v > 32000:
            raise ValueError('Max tokens must be between 1 and 32000')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager: 'EnhancedConfigManager'):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not file_path.endswith(('.env', '.yaml', '.yml', '.json')):
            return
        
        # Debounce rapid changes
        current_time = time.time()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1.0:
                return
        
        self.last_modified[file_path] = current_time
        
        logger.info(f"Configuration file changed: {file_path}")
        
        # Schedule reload
        asyncio.create_task(self.config_manager.reload_configuration())

class EnhancedConfigManager:
    """
    Enhanced configuration manager with validation and hot reloading
    """
    
    def __init__(self):
        """Initialize configuration manager"""
        self.config_cache: Dict[str, Any] = {}
        self.config_history: List[Dict[str, Any]] = []
        self.validation_level = ConfigValidationLevel.RELAXED
        self.change_listeners: List[Callable[[Dict[str, Any]], None]] = []
        
        # File watching
        self.observer: Optional[Observer] = None
        self.watched_paths: List[str] = []
        
        # Configuration sources
        self.env_file_path = ".env"
        self.config_file_path = "config.yaml"
        self.backup_dir = Path("config_backups")
        
        # Hot reload settings
        self.hot_reload_enabled = True
        self.reload_lock = threading.Lock()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize configuration manager"""
        try:
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Load initial configuration
            self._load_configuration()
            
            # Start file watching if enabled
            if self.hot_reload_enabled:
                self._start_file_watching()
            
            logger.info("Enhanced configuration manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            raise
    
    def _load_configuration(self) -> ConfigValidationResult:
        """Load configuration from all sources"""
        result = ConfigValidationResult()
        
        try:
            # Load environment variables
            if os.path.exists(self.env_file_path):
                load_dotenv(self.env_file_path, override=True)
                logger.info(f"Loaded environment from {self.env_file_path}")
            
            # Load YAML configuration if exists
            yaml_config = {}
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded YAML config from {self.config_file_path}")
            
            # Merge configurations (env vars take precedence)
            merged_config = self._merge_configurations(yaml_config)
            
            # Validate configuration
            validation_result = self._validate_configuration(merged_config)
            result.errors.extend(validation_result.errors)
            result.warnings.extend(validation_result.warnings)
            result.valid = validation_result.valid
            
            # Cache configuration
            old_config = self.config_cache.copy()
            self.config_cache = merged_config
            
            # Track changes
            result.changed_keys = self._find_changed_keys(old_config, merged_config)
            
            # Backup current configuration
            if result.valid:
                self._backup_configuration(merged_config)
            
            # Add to history
            self.config_history.append({
                'timestamp': datetime.utcnow(),
                'config': merged_config.copy(),
                'validation_result': result
            })
            
            # Keep only last 10 configurations
            if len(self.config_history) > 10:
                self.config_history.pop(0)
            
            logger.info(f"Configuration loaded with {len(result.errors)} errors and {len(result.warnings)} warnings")
            
        except Exception as e:
            result.add_error(f"Failed to load configuration: {str(e)}")
            logger.error(f"Configuration loading failed: {e}")
        
        return result
    
    def _merge_configurations(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML config with environment variables"""
        merged = yaml_config.copy()
        
        # Get current settings values
        for attr_name in dir(settings):
            if not attr_name.startswith('_'):
                value = getattr(settings, attr_name)
                # Convert to lowercase with underscores for consistency
                key = attr_name.lower()
                merged[key] = value
        
        return merged
    
    def _validate_configuration(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """Validate configuration against schema"""
        result = ConfigValidationResult()
        
        try:
            # Main configuration validation
            main_config = SuperNovaConfig(**{
                k: v for k, v in config.items() 
                if k in SuperNovaConfig.__fields__
            })
            
            # LLM provider validations
            self._validate_llm_providers(config, result)
            
            # Cost tracking validation
            self._validate_cost_tracking(config, result)
            
            # Security validation
            self._validate_security_settings(config, result)
            
            # Environment-specific validation
            self._validate_environment_specific(config, result)
            
        except ValidationError as e:
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                message = error['msg']
                result.add_error(f"Validation error in {field}: {message}")
        
        except Exception as e:
            result.add_error(f"Unexpected validation error: {str(e)}")
        
        return result
    
    def _validate_llm_providers(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate LLM provider configurations"""
        providers = ['openai', 'anthropic', 'ollama', 'huggingface']
        
        for provider in providers:
            provider_config = {}
            
            # Collect provider-specific settings
            for key, value in config.items():
                if key.startswith(f'{provider}_'):
                    setting_name = key.replace(f'{provider}_', '')
                    if setting_name == 'api_key':
                        provider_config['api_key'] = value
                    elif setting_name == 'base_url':
                        provider_config['base_url'] = value
            
            # Add common settings
            provider_config['enabled'] = config.get(f'{provider}_enabled', True)
            provider_config['timeout'] = config.get('llm_timeout', 60)
            provider_config['max_retries'] = config.get('llm_max_retries', 3)
            
            try:
                LLMProviderConfig(**provider_config)
            except ValidationError as e:
                for error in e.errors():
                    result.add_error(f"LLM {provider} config error: {error['msg']}")
    
    def _validate_cost_tracking(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate cost tracking configuration"""
        cost_config = {
            'enabled': config.get('cost_tracking_enabled', True),
            'daily_limit': config.get('llm_daily_cost_limit', 50.0),
            'monthly_limit': config.get('llm_monthly_cost_limit', 1000.0),
            'alert_threshold': config.get('llm_cost_alert_threshold', 0.8)
        }
        
        try:
            CostTrackingConfig(**cost_config)
        except ValidationError as e:
            for error in e.errors():
                result.add_error(f"Cost tracking config error: {error['msg']}")
    
    def _validate_security_settings(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate security-related settings"""
        # Check for potential security issues
        if config.get('debug', False) and config.get('supernova_env') == 'production':
            result.add_warning("Debug mode is enabled in production environment")
        
        if not config.get('api_key_encryption_enabled', True):
            result.add_warning("API key encryption is disabled")
        
        # Check for weak encryption keys
        encryption_key = config.get('api_key_encryption_key')
        if encryption_key and len(encryption_key) < 32:
            result.add_error("API key encryption key must be at least 32 characters")
        
        # Check database URL security
        database_url = config.get('database_url', '')
        if 'production' in config.get('supernova_env', '') and 'sqlite' in database_url:
            result.add_warning("Using SQLite in production environment")
    
    def _validate_environment_specific(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate environment-specific configurations"""
        env = config.get('supernova_env', 'development')
        
        if env == 'production':
            # Production-specific validations
            required_prod_settings = [
                'openai_api_key',
                'anthropic_api_key',
                'database_url',
                'api_key_encryption_key'
            ]
            
            for setting in required_prod_settings:
                if not config.get(setting):
                    result.add_warning(f"Production setting '{setting}' is not configured")
            
            # Check for development settings in production
            if config.get('debug', False):
                result.add_error("Debug mode should not be enabled in production")
        
        elif env == 'development':
            # Development-specific validations
            if not config.get('debug', True):
                result.add_warning("Debug mode is typically enabled in development")
    
    def _find_changed_keys(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[str]:
        """Find keys that have changed between configurations"""
        changed = []
        
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                changed.append(key)
        
        return changed
    
    def _backup_configuration(self, config: Dict[str, Any]):
        """Backup current configuration"""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"config_backup_{timestamp}.json"
            
            with open(backup_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            # Clean old backups (keep last 20)
            backup_files = sorted(self.backup_dir.glob("config_backup_*.json"))
            if len(backup_files) > 20:
                for old_backup in backup_files[:-20]:
                    old_backup.unlink()
            
            logger.debug(f"Configuration backed up to {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
    
    def _start_file_watching(self):
        """Start watching configuration files for changes"""
        try:
            self.observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            # Watch current directory for .env files
            watch_paths = ['.', 'config']
            
            for path in watch_paths:
                if os.path.exists(path):
                    self.observer.schedule(event_handler, path, recursive=False)
                    self.watched_paths.append(path)
            
            self.observer.start()
            logger.info(f"Started watching configuration files in: {self.watched_paths}")
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            self.observer = None
    
    async def reload_configuration(self) -> ConfigValidationResult:
        """Hot reload configuration"""
        with self.reload_lock:
            logger.info("Hot reloading configuration...")
            
            result = self._load_configuration()
            
            if result.valid or self.validation_level == ConfigValidationLevel.RELAXED:
                # Notify listeners of configuration change
                for listener in self.change_listeners:
                    try:
                        listener(self.config_cache)
                    except Exception as e:
                        logger.error(f"Configuration change listener error: {e}")
                
                # Update settings object
                self._update_settings_object()
                
                logger.info(f"Configuration reloaded successfully. Changed keys: {result.changed_keys}")
            else:
                logger.error(f"Configuration reload failed validation: {result.errors}")
            
            return result
    
    def _update_settings_object(self):
        """Update the global settings object with new configuration"""
        try:
            for key, value in self.config_cache.items():
                attr_name = key.upper()
                if hasattr(settings, attr_name):
                    setattr(settings, attr_name, value)
            
            logger.debug("Settings object updated with new configuration")
            
        except Exception as e:
            logger.error(f"Failed to update settings object: {e}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config_cache.get(key.lower(), default)
    
    def set_config_value(self, key: str, value: Any) -> bool:
        """Set configuration value (in memory only)"""
        try:
            self.config_cache[key.lower()] = value
            
            # Update settings object
            attr_name = key.upper()
            if hasattr(settings, attr_name):
                setattr(settings, attr_name, value)
            
            logger.info(f"Configuration value updated: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration value: {e}")
            return False
    
    def add_change_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """Add configuration change listener"""
        self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """Remove configuration change listener"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        if not self.config_history:
            return {"error": "No configuration history available"}
        
        latest = self.config_history[-1]
        result = latest['validation_result']
        
        return {
            "timestamp": latest['timestamp'].isoformat(),
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "changed_keys": result.changed_keys,
            "total_settings": len(latest['config']),
            "validation_level": self.validation_level.value
        }
    
    def rollback_configuration(self, steps_back: int = 1) -> bool:
        """Rollback configuration to previous version"""
        if len(self.config_history) <= steps_back:
            logger.error("Not enough configuration history for rollback")
            return False
        
        try:
            target_config = self.config_history[-(steps_back + 1)]['config']
            
            # Validate the rollback configuration
            validation_result = self._validate_configuration(target_config)
            
            if validation_result.valid or self.validation_level == ConfigValidationLevel.RELAXED:
                self.config_cache = target_config.copy()
                self._update_settings_object()
                
                # Notify listeners
                for listener in self.change_listeners:
                    try:
                        listener(self.config_cache)
                    except Exception as e:
                        logger.error(f"Configuration rollback listener error: {e}")
                
                logger.info(f"Configuration rolled back {steps_back} steps")
                return True
            else:
                logger.error(f"Rollback configuration is invalid: {validation_result.errors}")
                return False
                
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False
    
    def export_configuration(self, file_path: str, format: str = 'json') -> bool:
        """Export current configuration to file"""
        try:
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(self.config_cache, f, indent=2, default=str)
            elif format.lower() == 'yaml':
                with open(file_path, 'w') as f:
                    yaml.dump(self.config_cache, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration manager statistics"""
        return {
            "total_settings": len(self.config_cache),
            "history_length": len(self.config_history),
            "watched_paths": self.watched_paths,
            "hot_reload_enabled": self.hot_reload_enabled,
            "validation_level": self.validation_level.value,
            "change_listeners": len(self.change_listeners),
            "backup_directory": str(self.backup_dir),
            "backup_count": len(list(self.backup_dir.glob("config_backup_*.json")))
        }
    
    def shutdown(self):
        """Shutdown configuration manager"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                logger.info("Configuration file watching stopped")
            
            # Final backup
            self._backup_configuration(self.config_cache)
            
            logger.info("Configuration manager shut down")
            
        except Exception as e:
            logger.error(f"Error during configuration manager shutdown: {e}")

# Global configuration manager instance
config_manager: Optional[EnhancedConfigManager] = None

def get_config_manager() -> EnhancedConfigManager:
    """Get global configuration manager instance"""
    global config_manager
    if config_manager is None:
        config_manager = EnhancedConfigManager()
    return config_manager

def reload_config() -> ConfigValidationResult:
    """Convenient function to reload configuration"""
    manager = get_config_manager()
    return asyncio.run(manager.reload_configuration())

def get_config(key: str, default: Any = None) -> Any:
    """Convenient function to get configuration value"""
    manager = get_config_manager()
    return manager.get_config_value(key, default)

def set_config(key: str, value: Any) -> bool:
    """Convenient function to set configuration value"""
    manager = get_config_manager()
    return manager.set_config_value(key, value)