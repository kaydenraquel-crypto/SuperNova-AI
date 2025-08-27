"""
SuperNova AI - Configuration Versioning and Rollback System

This module provides comprehensive configuration versioning with:
- Semantic versioning for configuration changes
- Git-like branching and merging for configuration management
- Rollback capabilities with impact analysis
- Configuration diffing and change tracking
- Automated backup and restore functionality
- Integration with deployment and CI/CD systems
"""

from __future__ import annotations
import os
import json
import logging
import hashlib
import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback
import tempfile
import subprocess
import re

import yaml
from packaging import version
import aiofiles
import git
from git import Repo, InvalidGitRepositoryError

from .config_management import config_manager, Environment, ConfigurationLevel
from .config_validation_enhanced import validator, ValidationResult
from .secrets_management import get_secrets_manager

logger = logging.getLogger(__name__)


class VersionType(str, Enum):
    """Configuration version types."""
    MAJOR = "major"      # Breaking changes
    MINOR = "minor"      # New features, backwards compatible
    PATCH = "patch"      # Bug fixes, maintenance
    HOTFIX = "hotfix"    # Emergency fixes


class ChangeType(str, Enum):
    """Configuration change types."""
    ADDITION = "addition"
    MODIFICATION = "modification"
    REMOVAL = "removal"
    RENAME = "rename"
    TYPE_CHANGE = "type_change"


@dataclass
class ConfigurationChange:
    """Represents a single configuration change."""
    key: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    old_key: Optional[str] = None  # For renames
    impact_level: str = "low"  # low, medium, high, critical
    requires_restart: bool = False
    affects_services: Set[str] = field(default_factory=set)
    security_sensitive: bool = False
    compliance_relevant: bool = False


@dataclass
class ConfigurationVersion:
    """Configuration version metadata."""
    version: str
    environment: Environment
    timestamp: datetime
    author: str
    message: str
    version_type: VersionType
    changes: List[ConfigurationChange] = field(default_factory=list)
    parent_version: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    validation_result: Optional[ValidationResult] = None
    deployment_target: Optional[str] = None
    rollback_safe: bool = True
    config_hash: Optional[str] = None
    size_bytes: int = 0
    change_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class RollbackPlan:
    """Plan for rolling back configuration changes."""
    target_version: str
    current_version: str
    affected_keys: Set[str]
    impact_analysis: Dict[str, Any]
    safety_checks: List[str]
    estimated_downtime: timedelta
    rollback_steps: List[Dict[str, Any]]
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


class ConfigurationDiffer:
    """Computes differences between configuration versions."""
    
    def __init__(self):
        self.service_mapping = {
            # Database configurations
            'database': ['database', 'orm', 'migrations', 'backup'],
            'db_': ['database', 'orm', 'migrations'],
            'timescale': ['timescale', 'analytics', 'monitoring'],
            
            # LLM configurations
            'llm': ['llm_service', 'chat', 'advisor', 'analysis'],
            'openai': ['llm_service', 'chat', 'advisor'],
            'anthropic': ['llm_service', 'chat', 'advisor'],
            
            # Cache configurations
            'redis': ['cache', 'session', 'queue'],
            'cache': ['cache', 'performance'],
            
            # Security configurations
            'jwt': ['auth', 'security'],
            'mfa': ['auth', 'security'],
            'ssl': ['security', 'network'],
            'cors': ['security', 'api'],
            'rate_limit': ['security', 'api'],
            
            # Monitoring configurations
            'log': ['logging', 'monitoring'],
            'metrics': ['monitoring', 'observability'],
            'alert': ['monitoring', 'alerting'],
            'health': ['monitoring', 'health'],
            
            # Performance configurations
            'pool': ['performance', 'database'],
            'timeout': ['performance', 'network'],
            'concurrent': ['performance', 'scaling'],
        }
        
        self.security_patterns = [
            'secret', 'key', 'password', 'token', 'credential',
            'cert', 'ssl', 'auth', 'jwt', 'encryption'
        ]
        
        self.critical_patterns = [
            'database_url', 'redis_url', 'secret_key',
            'encryption_key', 'jwt_secret_key'
        ]
    
    def compare_configurations(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> List[ConfigurationChange]:
        """Compare two configurations and return changes."""
        changes = []
        
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if key not in old_config:
                # Addition
                change = ConfigurationChange(
                    key=key,
                    change_type=ChangeType.ADDITION,
                    new_value=new_value
                )
            elif key not in new_config:
                # Removal
                change = ConfigurationChange(
                    key=key,
                    change_type=ChangeType.REMOVAL,
                    old_value=old_value
                )
            elif old_value != new_value:
                # Modification or type change
                change_type = ChangeType.TYPE_CHANGE if type(old_value) != type(new_value) else ChangeType.MODIFICATION
                change = ConfigurationChange(
                    key=key,
                    change_type=change_type,
                    old_value=old_value,
                    new_value=new_value
                )
            else:
                continue  # No change
            
            # Analyze impact
            self._analyze_change_impact(change)
            changes.append(change)
        
        return changes
    
    def _analyze_change_impact(self, change: ConfigurationChange):
        """Analyze the impact of a configuration change."""
        key_lower = change.key.lower()
        
        # Determine impact level
        if any(pattern in key_lower for pattern in self.critical_patterns):
            change.impact_level = "critical"
            change.requires_restart = True
        elif any(pattern in key_lower for pattern in self.security_patterns):
            change.impact_level = "high"
            change.security_sensitive = True
        elif key_lower.startswith(('database', 'redis', 'llm')):
            change.impact_level = "medium"
        else:
            change.impact_level = "low"
        
        # Determine affected services
        for pattern, services in self.service_mapping.items():
            if pattern in key_lower:
                change.affects_services.update(services)
                break
        
        # Check if restart required
        restart_patterns = [
            'database_url', 'redis_url', 'port', 'host',
            'ssl', 'encryption_key', 'secret_key'
        ]
        
        if any(pattern in key_lower for pattern in restart_patterns):
            change.requires_restart = True
        
        # Check compliance relevance
        compliance_patterns = [
            'audit', 'log', 'retention', 'encryption', 'gdpr', 'ccpa'
        ]
        
        if any(pattern in key_lower for pattern in compliance_patterns):
            change.compliance_relevant = True
    
    def generate_diff_summary(self, changes: List[ConfigurationChange]) -> Dict[str, Any]:
        """Generate a summary of configuration changes."""
        summary = {
            'total_changes': len(changes),
            'additions': 0,
            'modifications': 0,
            'removals': 0,
            'renames': 0,
            'type_changes': 0,
            'impact_levels': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'requires_restart': 0,
            'security_sensitive': 0,
            'compliance_relevant': 0,
            'affected_services': set()
        }
        
        for change in changes:
            # Count by type
            if change.change_type == ChangeType.ADDITION:
                summary['additions'] += 1
            elif change.change_type == ChangeType.MODIFICATION:
                summary['modifications'] += 1
            elif change.change_type == ChangeType.REMOVAL:
                summary['removals'] += 1
            elif change.change_type == ChangeType.RENAME:
                summary['renames'] += 1
            elif change.change_type == ChangeType.TYPE_CHANGE:
                summary['type_changes'] += 1
            
            # Count by impact
            summary['impact_levels'][change.impact_level] += 1
            
            # Count flags
            if change.requires_restart:
                summary['requires_restart'] += 1
            if change.security_sensitive:
                summary['security_sensitive'] += 1
            if change.compliance_relevant:
                summary['compliance_relevant'] += 1
            
            # Collect affected services
            summary['affected_services'].update(change.affects_services)
        
        # Convert set to list for JSON serialization
        summary['affected_services'] = list(summary['affected_services'])
        
        return summary


class ConfigurationVersionManager:
    """Manages configuration versions and history."""
    
    def __init__(
        self,
        repository_path: Path,
        environment: Optional[Environment] = None,
        use_git: bool = True
    ):
        self.repository_path = repository_path
        self.environment = environment or Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        self.use_git = use_git
        
        # Initialize repository
        self.repository_path.mkdir(parents=True, exist_ok=True)
        
        # Version storage
        self.versions_dir = self.repository_path / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.repository_path / "versions.json"
        self.versions_metadata: Dict[str, ConfigurationVersion] = {}
        
        # Components
        self.differ = ConfigurationDiffer()
        
        # Git repository (optional)
        self.git_repo: Optional[Repo] = None
        if self.use_git:
            self._initialize_git_repo()
        
        # Load existing metadata
        self._load_versions_metadata()
        
        logger.info(f"Configuration version manager initialized for {self.environment.value}")
    
    def _initialize_git_repo(self):
        """Initialize or open Git repository."""
        try:
            self.git_repo = Repo(self.repository_path)
            logger.info("Opened existing Git repository")
        except InvalidGitRepositoryError:
            try:
                self.git_repo = Repo.init(self.repository_path)
                logger.info("Initialized new Git repository")
                
                # Create initial gitignore
                gitignore_path = self.repository_path / ".gitignore"
                with open(gitignore_path, 'w') as f:
                    f.write("*.tmp\n*.log\n__pycache__/\n.pytest_cache/\n")
                
                # Initial commit
                self.git_repo.index.add([".gitignore"])
                self.git_repo.index.commit("Initial commit")
                
            except Exception as e:
                logger.warning(f"Could not initialize Git repository: {e}")
                self.git_repo = None
                self.use_git = False
    
    def _load_versions_metadata(self):
        """Load versions metadata from storage."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for version_str, version_data in data.items():
                    # Reconstruct ConfigurationVersion objects
                    version_obj = ConfigurationVersion(
                        version=version_data['version'],
                        environment=Environment(version_data['environment']),
                        timestamp=datetime.fromisoformat(version_data['timestamp']),
                        author=version_data['author'],
                        message=version_data['message'],
                        version_type=VersionType(version_data['version_type']),
                        parent_version=version_data.get('parent_version'),
                        tags=set(version_data.get('tags', [])),
                        deployment_target=version_data.get('deployment_target'),
                        rollback_safe=version_data.get('rollback_safe', True),
                        config_hash=version_data.get('config_hash'),
                        size_bytes=version_data.get('size_bytes', 0),
                        change_summary=version_data.get('change_summary', {})
                    )
                    
                    # Reconstruct changes
                    changes = []
                    for change_data in version_data.get('changes', []):
                        change = ConfigurationChange(
                            key=change_data['key'],
                            change_type=ChangeType(change_data['change_type']),
                            old_value=change_data.get('old_value'),
                            new_value=change_data.get('new_value'),
                            old_key=change_data.get('old_key'),
                            impact_level=change_data.get('impact_level', 'low'),
                            requires_restart=change_data.get('requires_restart', False),
                            affects_services=set(change_data.get('affects_services', [])),
                            security_sensitive=change_data.get('security_sensitive', False),
                            compliance_relevant=change_data.get('compliance_relevant', False)
                        )
                        changes.append(change)
                    
                    version_obj.changes = changes
                    self.versions_metadata[version_str] = version_obj
                
                logger.info(f"Loaded {len(self.versions_metadata)} version records")
        
        except Exception as e:
            logger.error(f"Error loading versions metadata: {e}")
    
    def _save_versions_metadata(self):
        """Save versions metadata to storage."""
        try:
            # Convert to serializable format
            data = {}
            for version_str, version_obj in self.versions_metadata.items():
                version_data = asdict(version_obj)
                
                # Convert non-serializable types
                version_data['environment'] = version_obj.environment.value
                version_data['timestamp'] = version_obj.timestamp.isoformat()
                version_data['version_type'] = version_obj.version_type.value
                version_data['tags'] = list(version_obj.tags)
                
                # Convert changes
                changes_data = []
                for change in version_obj.changes:
                    change_data = asdict(change)
                    change_data['change_type'] = change.change_type.value
                    change_data['affects_services'] = list(change.affects_services)
                    changes_data.append(change_data)
                
                version_data['changes'] = changes_data
                data[version_str] = version_data
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving versions metadata: {e}")
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _get_next_version(
        self,
        version_type: VersionType,
        current_version: Optional[str] = None
    ) -> str:
        """Calculate next version number."""
        if not current_version:
            # Get latest version
            versions = [v.version for v in self.versions_metadata.values()]
            if not versions:
                return "1.0.0"
            
            # Find highest version
            try:
                sorted_versions = sorted(versions, key=lambda x: version.parse(x), reverse=True)
                current_version = sorted_versions[0]
            except Exception:
                # If versions are not parseable, start fresh
                return "1.0.0"
        
        try:
            current = version.parse(current_version)
            
            if version_type == VersionType.MAJOR:
                return f"{current.major + 1}.0.0"
            elif version_type == VersionType.MINOR:
                return f"{current.major}.{current.minor + 1}.0"
            elif version_type in [VersionType.PATCH, VersionType.HOTFIX]:
                return f"{current.major}.{current.minor}.{current.micro + 1}"
            
        except Exception:
            # Fallback to timestamp-based versioning
            timestamp = datetime.utcnow().strftime("%Y.%m.%d")
            return f"{timestamp}.1"
    
    async def create_version(
        self,
        config: Dict[str, Any],
        message: str,
        version_type: VersionType = VersionType.PATCH,
        author: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        deployment_target: Optional[str] = None,
        validate: bool = True
    ) -> ConfigurationVersion:
        """Create a new configuration version."""
        
        try:
            # Get current version
            current_version = self.get_latest_version()
            parent_version = current_version.version if current_version else None
            
            # Calculate new version number
            new_version_number = self._get_next_version(version_type, parent_version)
            
            # Calculate changes
            changes = []
            if current_version:
                current_config = self.load_configuration_version(current_version.version)
                changes = self.differ.compare_configurations(current_config, config)
            
            # Validate configuration if requested
            validation_result = None
            if validate:
                validation_result = await validator.validate_environment(
                    environment=self.environment,
                    config=config
                )
                
                if not validation_result.passed and self.environment == Environment.PRODUCTION:
                    raise ValueError(f"Configuration validation failed: {len(validation_result.errors)} errors")
            
            # Create version metadata
            version_obj = ConfigurationVersion(
                version=new_version_number,
                environment=self.environment,
                timestamp=datetime.utcnow(),
                author=author or os.getenv('USER', 'system'),
                message=message,
                version_type=version_type,
                changes=changes,
                parent_version=parent_version,
                tags=tags or set(),
                validation_result=validation_result,
                deployment_target=deployment_target,
                config_hash=self._calculate_config_hash(config),
                size_bytes=len(json.dumps(config, default=str)),
                change_summary=self.differ.generate_diff_summary(changes)
            )
            
            # Determine if rollback safe
            version_obj.rollback_safe = self._is_rollback_safe(changes)
            
            # Save configuration to file
            config_file = self.versions_dir / f"{new_version_number}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            # Save metadata
            self.versions_metadata[new_version_number] = version_obj
            self._save_versions_metadata()
            
            # Git commit if enabled
            if self.use_git and self.git_repo:
                try:
                    self.git_repo.index.add([
                        str(config_file.relative_to(self.repository_path)),
                        str(self.metadata_file.relative_to(self.repository_path))
                    ])
                    
                    commit_message = f"{version_type.value}: {message}\n\nVersion: {new_version_number}"
                    if changes:
                        commit_message += f"\nChanges: {len(changes)} modifications"
                    
                    self.git_repo.index.commit(commit_message)
                    
                    # Add tag
                    self.git_repo.create_tag(
                        f"v{new_version_number}",
                        message=f"Version {new_version_number}: {message}"
                    )
                    
                    logger.info(f"Created Git commit and tag for version {new_version_number}")
                
                except Exception as e:
                    logger.warning(f"Git operations failed: {e}")
            
            logger.info(f"Created configuration version {new_version_number} with {len(changes)} changes")
            return version_obj
        
        except Exception as e:
            logger.error(f"Error creating configuration version: {e}")
            raise
    
    def _is_rollback_safe(self, changes: List[ConfigurationChange]) -> bool:
        """Determine if configuration changes are safe for rollback."""
        
        # Check for dangerous change patterns
        unsafe_patterns = [
            'database_version', 'schema_version', 'migration_version'
        ]
        
        for change in changes:
            # Critical changes are generally not safe to rollback
            if change.impact_level == "critical":
                return False
            
            # Check for unsafe patterns
            if any(pattern in change.key.lower() for pattern in unsafe_patterns):
                return False
            
            # Additions are generally safe to rollback (remove them)
            # Modifications might be safe depending on the change
            # Removals might not be safe (data loss)
            if change.change_type == ChangeType.REMOVAL:
                # Check if it's a critical configuration
                if change.impact_level in ["high", "critical"]:
                    return False
        
        return True
    
    def get_version(self, version_number: str) -> Optional[ConfigurationVersion]:
        """Get version metadata by version number."""
        return self.versions_metadata.get(version_number)
    
    def get_latest_version(self) -> Optional[ConfigurationVersion]:
        """Get the latest configuration version."""
        if not self.versions_metadata:
            return None
        
        try:
            # Sort by semantic version
            sorted_versions = sorted(
                self.versions_metadata.items(),
                key=lambda x: version.parse(x[0]),
                reverse=True
            )
            return sorted_versions[0][1]
        
        except Exception:
            # Fallback to timestamp sorting
            sorted_versions = sorted(
                self.versions_metadata.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
            return sorted_versions[0] if sorted_versions else None
    
    def list_versions(
        self,
        limit: Optional[int] = None,
        environment: Optional[Environment] = None,
        tags: Optional[Set[str]] = None
    ) -> List[ConfigurationVersion]:
        """List configuration versions with optional filtering."""
        
        versions = list(self.versions_metadata.values())
        
        # Filter by environment
        if environment:
            versions = [v for v in versions if v.environment == environment]
        
        # Filter by tags
        if tags:
            versions = [v for v in versions if tags.intersection(v.tags)]
        
        # Sort by version number (descending)
        try:
            versions = sorted(versions, key=lambda x: version.parse(x.version), reverse=True)
        except Exception:
            # Fallback to timestamp sorting
            versions = sorted(versions, key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def load_configuration_version(self, version_number: str) -> Dict[str, Any]:
        """Load configuration for a specific version."""
        config_file = self.versions_dir / f"{version_number}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file for version {version_number} not found")
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading configuration for version {version_number}: {e}")
            raise
    
    def compare_versions(
        self,
        from_version: str,
        to_version: str
    ) -> List[ConfigurationChange]:
        """Compare two configuration versions."""
        
        from_config = self.load_configuration_version(from_version)
        to_config = self.load_configuration_version(to_version)
        
        return self.differ.compare_configurations(from_config, to_config)
    
    async def create_rollback_plan(
        self,
        target_version: str,
        current_version: Optional[str] = None
    ) -> RollbackPlan:
        """Create a rollback plan to a target version."""
        
        if current_version is None:
            latest = self.get_latest_version()
            if not latest:
                raise ValueError("No current version available")
            current_version = latest.version
        
        # Get version metadata
        target_version_obj = self.get_version(target_version)
        if not target_version_obj:
            raise ValueError(f"Target version {target_version} not found")
        
        current_version_obj = self.get_version(current_version)
        if not current_version_obj:
            raise ValueError(f"Current version {current_version} not found")
        
        # Calculate changes needed for rollback
        changes = self.compare_versions(current_version, target_version)
        affected_keys = {change.key for change in changes}
        
        # Analyze impact
        impact_analysis = {
            'total_changes': len(changes),
            'critical_changes': len([c for c in changes if c.impact_level == 'critical']),
            'requires_restart': any(c.requires_restart for c in changes),
            'affected_services': list({service for c in changes for service in c.affects_services}),
            'security_impact': any(c.security_sensitive for c in changes),
            'compliance_impact': any(c.compliance_relevant for c in changes),
            'data_loss_risk': any(c.change_type == ChangeType.REMOVAL for c in changes)
        }
        
        # Determine safety checks needed
        safety_checks = [
            'configuration_validation',
            'database_connectivity',
            'service_health_check'
        ]
        
        if impact_analysis['security_impact']:
            safety_checks.append('security_validation')
        
        if impact_analysis['requires_restart']:
            safety_checks.append('graceful_shutdown_test')
        
        # Estimate downtime
        base_downtime = timedelta(seconds=30)  # Base rollback time
        if impact_analysis['requires_restart']:
            base_downtime += timedelta(minutes=2)  # Additional restart time
        
        complexity_factor = min(len(changes) / 10, 3)  # Max 3x multiplier
        estimated_downtime = base_downtime * (1 + complexity_factor)
        
        # Generate rollback steps
        rollback_steps = [
            {
                'step': 1,
                'action': 'create_backup',
                'description': f'Create backup of current configuration version {current_version}',
                'estimated_time': timedelta(seconds=10)
            },
            {
                'step': 2,
                'action': 'validate_target',
                'description': f'Validate target configuration version {target_version}',
                'estimated_time': timedelta(seconds=30)
            },
            {
                'step': 3,
                'action': 'run_safety_checks',
                'description': f'Run safety checks: {", ".join(safety_checks)}',
                'estimated_time': timedelta(seconds=60)
            },
            {
                'step': 4,
                'action': 'apply_configuration',
                'description': f'Apply configuration changes ({len(changes)} changes)',
                'estimated_time': timedelta(seconds=30)
            }
        ]
        
        if impact_analysis['requires_restart']:
            rollback_steps.append({
                'step': 5,
                'action': 'restart_services',
                'description': f'Restart affected services: {", ".join(impact_analysis["affected_services"])}',
                'estimated_time': timedelta(minutes=2)
            })
        
        rollback_steps.append({
            'step': len(rollback_steps) + 1,
            'action': 'verify_rollback',
            'description': 'Verify rollback success and system health',
            'estimated_time': timedelta(seconds=30)
        })
        
        # Determine if approval required
        requires_approval = (
            impact_analysis['critical_changes'] > 0 or
            impact_analysis['security_impact'] or
            impact_analysis['data_loss_risk'] or
            self.environment == Environment.PRODUCTION
        )
        
        return RollbackPlan(
            target_version=target_version,
            current_version=current_version,
            affected_keys=affected_keys,
            impact_analysis=impact_analysis,
            safety_checks=safety_checks,
            estimated_downtime=estimated_downtime,
            rollback_steps=rollback_steps,
            requires_approval=requires_approval
        )
    
    async def execute_rollback(
        self,
        rollback_plan: RollbackPlan,
        dry_run: bool = False,
        force: bool = False
    ) -> Dict[str, Any]:
        """Execute a rollback plan."""
        
        if rollback_plan.requires_approval and not rollback_plan.approved_by and not force:
            raise ValueError("Rollback requires approval but none provided")
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Executing rollback from {rollback_plan.current_version} to {rollback_plan.target_version}")
        
        results = {
            'success': False,
            'steps_completed': 0,
            'total_steps': len(rollback_plan.rollback_steps),
            'errors': [],
            'rollback_performed': False,
            'dry_run': dry_run
        }
        
        try:
            # Execute rollback steps
            for step in rollback_plan.rollback_steps:
                step_num = step['step']
                action = step['action']
                description = step['description']
                
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Step {step_num}: {description}")
                
                try:
                    if not dry_run:
                        if action == 'create_backup':
                            # Current configuration should already be versioned
                            pass
                        
                        elif action == 'validate_target':
                            target_config = self.load_configuration_version(rollback_plan.target_version)
                            validation_result = await validator.validate_environment(
                                environment=self.environment,
                                config=target_config
                            )
                            
                            if not validation_result.passed:
                                raise ValueError(f"Target configuration validation failed: {len(validation_result.errors)} errors")
                        
                        elif action == 'run_safety_checks':
                            # Safety checks would be implemented here
                            pass
                        
                        elif action == 'apply_configuration':
                            # Apply the target configuration
                            target_config = self.load_configuration_version(rollback_plan.target_version)
                            
                            # Update configuration manager
                            for key, value in target_config.items():
                                await config_manager.set_configuration(
                                    key=key,
                                    value=value,
                                    description=f"Rollback to version {rollback_plan.target_version}"
                                )
                        
                        elif action == 'restart_services':
                            # Service restart logic would be implemented here
                            logger.info("Services would be restarted here")
                        
                        elif action == 'verify_rollback':
                            # Verification logic would be implemented here
                            pass
                    
                    results['steps_completed'] += 1
                    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Step {step_num} completed successfully")
                
                except Exception as e:
                    error_msg = f"Step {step_num} failed: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                    raise
            
            results['success'] = True
            results['rollback_performed'] = not dry_run
            
            if not dry_run:
                logger.info(f"Rollback to version {rollback_plan.target_version} completed successfully")
            else:
                logger.info(f"Dry run rollback simulation completed successfully")
        
        except Exception as e:
            results['success'] = False
            error_msg = f"Rollback failed: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def get_version_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get version history for display."""
        versions = self.list_versions(limit=limit)
        
        history = []
        for ver in versions:
            history.append({
                'version': ver.version,
                'timestamp': ver.timestamp.isoformat(),
                'author': ver.author,
                'message': ver.message,
                'version_type': ver.version_type.value,
                'environment': ver.environment.value,
                'changes_count': len(ver.changes),
                'change_summary': ver.change_summary,
                'rollback_safe': ver.rollback_safe,
                'tags': list(ver.tags),
                'config_hash': ver.config_hash,
                'size_bytes': ver.size_bytes
            })
        
        return history


# Global version manager
_version_manager: Optional[ConfigurationVersionManager] = None


def get_version_manager() -> ConfigurationVersionManager:
    """Get global version manager instance."""
    global _version_manager
    
    if _version_manager is None:
        repo_path = Path(os.getenv('CONFIG_REPOSITORY_PATH', './config_repository'))
        environment = Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        
        _version_manager = ConfigurationVersionManager(
            repository_path=repo_path,
            environment=environment,
            use_git=os.getenv('CONFIG_USE_GIT', 'true').lower() == 'true'
        )
    
    return _version_manager


# Convenience functions
async def create_configuration_version(
    config: Dict[str, Any],
    message: str,
    version_type: VersionType = VersionType.PATCH,
    **kwargs
) -> ConfigurationVersion:
    """Create a new configuration version."""
    manager = get_version_manager()
    return await manager.create_version(config, message, version_type, **kwargs)


async def rollback_to_version(
    target_version: str,
    dry_run: bool = False,
    force: bool = False
) -> Dict[str, Any]:
    """Rollback configuration to a specific version."""
    manager = get_version_manager()
    rollback_plan = await manager.create_rollback_plan(target_version)
    return await manager.execute_rollback(rollback_plan, dry_run=dry_run, force=force)


def list_configuration_versions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """List configuration versions."""
    manager = get_version_manager()
    return manager.get_version_history(limit=limit)


if __name__ == "__main__":
    """CLI interface for version management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperNova AI Configuration Version Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List versions command
    list_parser = subparsers.add_parser('list', help='List configuration versions')
    list_parser.add_argument('--limit', type=int, help='Limit number of versions to show')
    
    # Show version command
    show_parser = subparsers.add_parser('show', help='Show version details')
    show_parser.add_argument('version', help='Version number to show')
    
    # Compare versions command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('from_version', help='Source version')
    compare_parser.add_argument('to_version', help='Target version')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to version')
    rollback_parser.add_argument('version', help='Target version for rollback')
    rollback_parser.add_argument('--dry-run', action='store_true', help='Simulate rollback without applying changes')
    rollback_parser.add_argument('--force', action='store_true', help='Force rollback without approval')
    
    args = parser.parse_args()
    
    async def main():
        manager = get_version_manager()
        
        if args.command == 'list':
            versions = manager.get_version_history(limit=args.limit)
            print(f"{'Version':<15} {'Date':<20} {'Author':<15} {'Type':<10} {'Changes':<8} {'Message'}")
            print("-" * 90)
            
            for ver in versions:
                date_str = datetime.fromisoformat(ver['timestamp']).strftime('%Y-%m-%d %H:%M')
                print(f"{ver['version']:<15} {date_str:<20} {ver['author']:<15} {ver['version_type']:<10} {ver['changes_count']:<8} {ver['message'][:30]}")
        
        elif args.command == 'show':
            version_obj = manager.get_version(args.version)
            if not version_obj:
                print(f"Version {args.version} not found")
                return
            
            print(f"Version: {version_obj.version}")
            print(f"Environment: {version_obj.environment.value}")
            print(f"Timestamp: {version_obj.timestamp}")
            print(f"Author: {version_obj.author}")
            print(f"Type: {version_obj.version_type.value}")
            print(f"Message: {version_obj.message}")
            print(f"Changes: {len(version_obj.changes)}")
            print(f"Rollback Safe: {version_obj.rollback_safe}")
            
            if version_obj.change_summary:
                print("\nChange Summary:")
                for key, value in version_obj.change_summary.items():
                    if isinstance(value, list):
                        print(f"  {key}: {', '.join(value)}")
                    else:
                        print(f"  {key}: {value}")
        
        elif args.command == 'compare':
            changes = manager.compare_versions(args.from_version, args.to_version)
            print(f"Comparing {args.from_version} → {args.to_version}")
            print(f"Total changes: {len(changes)}")
            print()
            
            for change in changes:
                print(f"{change.change_type.value.upper()}: {change.key}")
                if change.old_value is not None:
                    print(f"  Old: {change.old_value}")
                if change.new_value is not None:
                    print(f"  New: {change.new_value}")
                print(f"  Impact: {change.impact_level}")
                print()
        
        elif args.command == 'rollback':
            rollback_plan = await manager.create_rollback_plan(args.version)
            
            print(f"Rollback Plan: {rollback_plan.current_version} → {rollback_plan.target_version}")
            print(f"Estimated Downtime: {rollback_plan.estimated_downtime}")
            print(f"Requires Approval: {rollback_plan.requires_approval}")
            print()
            
            print("Impact Analysis:")
            for key, value in rollback_plan.impact_analysis.items():
                print(f"  {key}: {value}")
            print()
            
            if args.dry_run:
                print("Executing dry run...")
                result = await manager.execute_rollback(rollback_plan, dry_run=True)
                print(f"Dry run completed: {result['success']}")
            else:
                confirm = input("Execute rollback? (y/N): ")
                if confirm.lower() == 'y':
                    result = await manager.execute_rollback(rollback_plan, force=args.force)
                    print(f"Rollback completed: {result['success']}")
                    if result['errors']:
                        print("Errors:")
                        for error in result['errors']:
                            print(f"  {error}")
        
        else:
            parser.print_help()
    
    asyncio.run(main())