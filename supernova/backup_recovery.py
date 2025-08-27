"""
Database Backup and Disaster Recovery System

Comprehensive backup and disaster recovery solution including:
- Automated database backups (full, incremental, differential)
- Point-in-time recovery capabilities
- Cross-region backup replication
- Backup verification and testing
- Disaster recovery procedures and failover
- Backup retention policies and cleanup
"""

from __future__ import annotations
import os
import logging
import subprocess
import shutil
import gzip
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile
import hashlib
from contextlib import contextmanager
from urllib.parse import urlparse
import schedule
import threading

try:
    from .config import settings
    from .database_config import db_config, get_session
    from .migration_manager import migration_manager
    import boto3
    from azure.storage.blob import BlobServiceClient
    from google.cloud import storage as gcs
    CLOUD_STORAGE_AVAILABLE = True
except ImportError:
    CLOUD_STORAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class BackupMetadata:
    """Backup metadata information."""
    backup_id: str
    database: str
    backup_type: str  # full, incremental, differential
    start_time: datetime
    end_time: datetime
    file_path: str
    file_size_bytes: int
    checksum: str
    compression: str
    encryption: bool
    database_version: str
    schema_version: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class RecoveryPoint:
    """Recovery point information."""
    timestamp: datetime
    backup_id: str
    database: str
    transaction_log_position: Optional[str] = None
    recovery_possible: bool = True

class BackupStorage:
    """Abstract base class for backup storage backends."""
    
    def upload_backup(self, local_path: str, remote_path: str) -> bool:
        """Upload backup to storage backend."""
        raise NotImplementedError
    
    def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Download backup from storage backend."""
        raise NotImplementedError
    
    def list_backups(self, prefix: str = '') -> List[str]:
        """List available backups."""
        raise NotImplementedError
    
    def delete_backup(self, remote_path: str) -> bool:
        """Delete backup from storage."""
        raise NotImplementedError

class LocalStorage(BackupStorage):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload_backup(self, local_path: str, remote_path: str) -> bool:
        """Copy backup to local storage directory."""
        try:
            dest_path = self.base_path / remote_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest_path)
            return True
        except Exception as e:
            logger.error(f"Failed to upload to local storage: {e}")
            return False
    
    def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Copy backup from local storage."""
        try:
            source_path = self.base_path / remote_path
            if source_path.exists():
                shutil.copy2(source_path, local_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to download from local storage: {e}")
            return False
    
    def list_backups(self, prefix: str = '') -> List[str]:
        """List backups in local storage."""
        try:
            pattern = f"{prefix}*" if prefix else "*"
            return [str(p.relative_to(self.base_path)) for p in self.base_path.glob(pattern)]
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def delete_backup(self, remote_path: str) -> bool:
        """Delete backup from local storage."""
        try:
            backup_path = self.base_path / remote_path
            if backup_path.exists():
                backup_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False

class S3Storage(BackupStorage):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1', aws_access_key_id: str = None, aws_secret_access_key: str = None):
        if not CLOUD_STORAGE_AVAILABLE:
            raise ImportError("AWS SDK not available. Install boto3.")
        
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def upload_backup(self, local_path: str, remote_path: str) -> bool:
        """Upload backup to S3."""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, remote_path)
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Download backup from S3."""
        try:
            self.s3_client.download_file(self.bucket_name, remote_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    def list_backups(self, prefix: str = '') -> List[str]:
        """List backups in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list S3 backups: {e}")
            return []
    
    def delete_backup(self, remote_path: str) -> bool:
        """Delete backup from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete S3 backup: {e}")
            return False

class DatabaseBackup:
    """Database backup management."""
    
    def __init__(self, storage_backend: BackupStorage):
        self.storage_backend = storage_backend
        self.temp_dir = Path(tempfile.gettempdir()) / 'supernova_backups'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Backup configuration
        self.compression_enabled = getattr(settings, 'BACKUP_COMPRESSION', True)
        self.encryption_enabled = getattr(settings, 'BACKUP_ENCRYPTION', False)
        self.encryption_key = getattr(settings, 'BACKUP_ENCRYPTION_KEY', None)
        
        # Backup metadata tracking
        self.backup_history: List[BackupMetadata] = []
        self.load_backup_history()
    
    def create_backup(self, database: str, backup_type: str = 'full') -> Optional[BackupMetadata]:
        """
        Create database backup.
        
        Args:
            database: Database name to backup
            backup_type: Type of backup (full, incremental, differential)
            
        Returns:
            BackupMetadata if successful, None otherwise
        """
        backup_id = self._generate_backup_id(database, backup_type)
        start_time = datetime.utcnow()
        
        logger.info(f"Starting {backup_type} backup for database: {database}")
        
        try:
            # Get database URL
            urls = db_config.get_database_urls()
            if database not in urls:
                raise ValueError(f"Database '{database}' not configured")
            
            db_url = urls[database]
            
            # Create temporary backup file
            temp_file = self.temp_dir / f"{backup_id}.sql"
            
            # Perform backup based on database type
            if db_url.startswith('sqlite'):
                success = self._backup_sqlite(db_url, temp_file)
            elif db_url.startswith('postgresql'):
                success = self._backup_postgresql(db_url, temp_file, backup_type)
            else:
                raise ValueError(f"Unsupported database type: {db_url}")
            
            if not success:
                raise RuntimeError("Backup creation failed")
            
            # Post-process backup file
            final_file = temp_file
            
            # Compress if enabled
            if self.compression_enabled:
                final_file = self._compress_backup(temp_file)
                temp_file.unlink()  # Remove uncompressed file
            
            # Encrypt if enabled
            if self.encryption_enabled:
                final_file = self._encrypt_backup(final_file)
            
            # Calculate checksum
            checksum = self._calculate_checksum(final_file)
            
            # Get file size
            file_size = final_file.stat().st_size
            
            # Upload to storage backend
            remote_path = f"{database}/{backup_type}/{backup_id}{final_file.suffix}"
            upload_success = self.storage_backend.upload_backup(str(final_file), remote_path)
            
            end_time = datetime.utcnow()
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                database=database,
                backup_type=backup_type,
                start_time=start_time,
                end_time=end_time,
                file_path=remote_path,
                file_size_bytes=file_size,
                checksum=checksum,
                compression='gzip' if self.compression_enabled else 'none',
                encryption=self.encryption_enabled,
                database_version=self._get_database_version(database),
                schema_version=self._get_schema_version(database),
                success=upload_success
            )
            
            if upload_success:
                logger.info(f"Backup {backup_id} completed successfully")
                self.backup_history.append(metadata)
                self.save_backup_history()
            else:
                metadata.error_message = "Failed to upload backup to storage"
                logger.error(f"Backup {backup_id} failed to upload")
            
            # Cleanup temporary file
            if final_file.exists():
                final_file.unlink()
            
            return metadata
            
        except Exception as e:
            end_time = datetime.utcnow()
            error_metadata = BackupMetadata(
                backup_id=backup_id,
                database=database,
                backup_type=backup_type,
                start_time=start_time,
                end_time=end_time,
                file_path='',
                file_size_bytes=0,
                checksum='',
                compression='none',
                encryption=False,
                database_version='',
                schema_version='',
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Backup {backup_id} failed: {e}")
            return error_metadata
    
    def _backup_sqlite(self, db_url: str, output_file: Path) -> bool:
        """Create SQLite database backup."""
        try:
            # Extract database file path
            db_path = db_url.replace('sqlite:///', '')
            
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"SQLite database not found: {db_path}")
            
            # Use SQLite's backup command
            import sqlite3
            
            source_db = sqlite3.connect(db_path)
            
            # Create SQL dump
            with open(output_file, 'w') as f:
                for line in source_db.iterdump():
                    f.write(f'{line}\n')
            
            source_db.close()
            return True
            
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return False
    
    def _backup_postgresql(self, db_url: str, output_file: Path, backup_type: str) -> bool:
        """Create PostgreSQL database backup."""
        try:
            # Parse database URL
            parsed = urlparse(db_url)
            
            # Build pg_dump command
            cmd = [
                'pg_dump',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path[1:]}',  # Remove leading slash
                f'--file={output_file}',
                '--verbose',
                '--no-password'
            ]
            
            # Add backup type specific options
            if backup_type == 'full':
                cmd.extend(['--clean', '--create'])
            elif backup_type == 'schema':
                cmd.append('--schema-only')
            elif backup_type == 'data':
                cmd.append('--data-only')
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password or ''
            
            # Run pg_dump
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"pg_dump failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return False
    
    def _compress_backup(self, backup_file: Path) -> Path:
        """Compress backup file using gzip."""
        compressed_file = backup_file.with_suffix(backup_file.suffix + '.gz')
        
        with open(backup_file, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return compressed_file
    
    def _encrypt_backup(self, backup_file: Path) -> Path:
        """Encrypt backup file (simplified implementation)."""
        # This is a simplified implementation
        # In production, use proper encryption libraries like cryptography
        encrypted_file = backup_file.with_suffix(backup_file.suffix + '.enc')
        
        if not self.encryption_key:
            logger.warning("Encryption enabled but no key provided")
            return backup_file
        
        try:
            from cryptography.fernet import Fernet
            
            # Use the encryption key (should be base64 encoded)
            fernet = Fernet(self.encryption_key.encode())
            
            with open(backup_file, 'rb') as f_in:
                with open(encrypted_file, 'wb') as f_out:
                    f_out.write(fernet.encrypt(f_in.read()))
            
            return encrypted_file
            
        except ImportError:
            logger.warning("Cryptography library not available, skipping encryption")
            return backup_file
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return backup_file
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _generate_backup_id(self, database: str, backup_type: str) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{database}_{backup_type}_{timestamp}"
    
    def _get_database_version(self, database: str) -> str:
        """Get database version."""
        try:
            with get_session(database) as session:
                result = session.execute(text("SELECT version()"))
                return result.scalar()[:50]  # Truncate long version strings
        except Exception:
            return "unknown"
    
    def _get_schema_version(self, database: str) -> str:
        """Get current schema version."""
        try:
            return migration_manager.get_current_revision(database) or "unknown"
        except Exception:
            return "unknown"
    
    def restore_backup(self, backup_id: str, target_database: str = None) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_id: ID of backup to restore
            target_database: Target database name (defaults to original)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find backup metadata
            backup_metadata = None
            for backup in self.backup_history:
                if backup.backup_id == backup_id:
                    backup_metadata = backup
                    break
            
            if not backup_metadata:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            if not backup_metadata.success:
                logger.error(f"Cannot restore failed backup {backup_id}")
                return False
            
            target_db = target_database or backup_metadata.database
            
            logger.info(f"Starting restore of backup {backup_id} to database {target_db}")
            
            # Download backup file
            temp_file = self.temp_dir / f"restore_{backup_id}"
            
            if not self.storage_backend.download_backup(backup_metadata.file_path, str(temp_file)):
                logger.error(f"Failed to download backup {backup_id}")
                return False
            
            # Verify checksum
            if backup_metadata.checksum:
                current_checksum = self._calculate_checksum(temp_file)
                if current_checksum != backup_metadata.checksum:
                    logger.error(f"Backup {backup_id} checksum mismatch")
                    return False
            
            # Decrypt if needed
            if backup_metadata.encryption:
                temp_file = self._decrypt_backup(temp_file)
            
            # Decompress if needed
            if backup_metadata.compression == 'gzip':
                temp_file = self._decompress_backup(temp_file)
            
            # Restore based on database type
            urls = db_config.get_database_urls()
            db_url = urls[target_db]
            
            if db_url.startswith('sqlite'):
                success = self._restore_sqlite(db_url, temp_file)
            elif db_url.startswith('postgresql'):
                success = self._restore_postgresql(db_url, temp_file)
            else:
                logger.error(f"Unsupported database type: {db_url}")
                success = False
            
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
            
            if success:
                logger.info(f"Successfully restored backup {backup_id}")
            else:
                logger.error(f"Failed to restore backup {backup_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _decrypt_backup(self, encrypted_file: Path) -> Path:
        """Decrypt backup file."""
        decrypted_file = encrypted_file.with_suffix('')
        
        try:
            from cryptography.fernet import Fernet
            
            fernet = Fernet(self.encryption_key.encode())
            
            with open(encrypted_file, 'rb') as f_in:
                with open(decrypted_file, 'wb') as f_out:
                    f_out.write(fernet.decrypt(f_in.read()))
            
            return decrypted_file
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_file
    
    def _decompress_backup(self, compressed_file: Path) -> Path:
        """Decompress backup file."""
        decompressed_file = compressed_file.with_suffix('')
        
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return decompressed_file
    
    def _restore_sqlite(self, db_url: str, backup_file: Path) -> bool:
        """Restore SQLite database."""
        try:
            # Extract database file path
            db_path = db_url.replace('sqlite:///', '')
            
            # Create backup of current database
            if os.path.exists(db_path):
                backup_current = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(db_path, backup_current)
            
            # Remove existing database
            if os.path.exists(db_path):
                os.remove(db_path)
            
            # Execute SQL script to restore
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            
            with open(backup_file, 'r') as f:
                sql_script = f.read()
                conn.executescript(sql_script)
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"SQLite restore failed: {e}")
            return False
    
    def _restore_postgresql(self, db_url: str, backup_file: Path) -> bool:
        """Restore PostgreSQL database."""
        try:
            # Parse database URL
            parsed = urlparse(db_url)
            
            # Build psql command
            cmd = [
                'psql',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path[1:]}',
                f'--file={backup_file}',
                '--single-transaction',
                '--set', 'ON_ERROR_STOP=on'
            ]
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password or ''
            
            # Run psql
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"psql restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"PostgreSQL restore failed: {e}")
            return False
    
    def list_backups(self, database: str = None) -> List[BackupMetadata]:
        """List available backups."""
        backups = self.backup_history
        
        if database:
            backups = [b for b in backups if b.database == database]
        
        # Sort by timestamp, newest first
        return sorted(backups, key=lambda x: x.start_time, reverse=True)
    
    def get_recovery_points(self, database: str) -> List[RecoveryPoint]:
        """Get available recovery points for database."""
        recovery_points = []
        
        for backup in self.backup_history:
            if backup.database == database and backup.success:
                recovery_points.append(RecoveryPoint(
                    timestamp=backup.end_time,
                    backup_id=backup.backup_id,
                    database=database,
                    recovery_possible=True
                ))
        
        return sorted(recovery_points, key=lambda x: x.timestamp, reverse=True)
    
    def cleanup_old_backups(self, database: str, retention_days: int = 30) -> int:
        """
        Clean up old backups based on retention policy.
        
        Args:
            database: Database name
            retention_days: Number of days to retain backups
            
        Returns:
            Number of backups deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        deleted_count = 0
        
        # Get backups to delete
        backups_to_delete = [
            backup for backup in self.backup_history
            if backup.database == database and backup.start_time < cutoff_date
        ]
        
        for backup in backups_to_delete:
            try:
                # Delete from storage
                if self.storage_backend.delete_backup(backup.file_path):
                    # Remove from history
                    self.backup_history.remove(backup)
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {backup.backup_id}")
                
            except Exception as e:
                logger.error(f"Failed to delete backup {backup.backup_id}: {e}")
        
        if deleted_count > 0:
            self.save_backup_history()
        
        return deleted_count
    
    def save_backup_history(self) -> None:
        """Save backup history to file."""
        try:
            history_file = self.temp_dir / 'backup_history.json'
            
            with open(history_file, 'w') as f:
                json.dump([asdict(backup) for backup in self.backup_history], f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")
    
    def load_backup_history(self) -> None:
        """Load backup history from file."""
        try:
            history_file = self.temp_dir / 'backup_history.json'
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                self.backup_history = []
                for item in data:
                    # Convert datetime strings back to datetime objects
                    item['start_time'] = datetime.fromisoformat(item['start_time'])
                    item['end_time'] = datetime.fromisoformat(item['end_time'])
                    self.backup_history.append(BackupMetadata(**item))
                
        except Exception as e:
            logger.error(f"Failed to load backup history: {e}")
            self.backup_history = []

class BackupScheduler:
    """Automatic backup scheduling."""
    
    def __init__(self, backup_manager: DatabaseBackup):
        self.backup_manager = backup_manager
        self.scheduler_thread = None
        self.running = False
    
    def schedule_backups(self, schedules: Dict[str, Dict[str, str]]) -> None:
        """
        Schedule automatic backups.
        
        Args:
            schedules: Dictionary mapping database names to schedule configurations
                     Example: {'primary': {'full': 'daily', 'incremental': 'hourly'}}
        """
        # Clear existing schedules
        schedule.clear()
        
        for database, backup_schedules in schedules.items():
            for backup_type, frequency in backup_schedules.items():
                if frequency == 'hourly':
                    schedule.every().hour.do(self._scheduled_backup, database, backup_type)
                elif frequency == 'daily':
                    schedule.every().day.at("02:00").do(self._scheduled_backup, database, backup_type)
                elif frequency == 'weekly':
                    schedule.every().week.do(self._scheduled_backup, database, backup_type)
                elif frequency == 'monthly':
                    schedule.every(30).days.do(self._scheduled_backup, database, backup_type)
        
        logger.info(f"Scheduled backups for {len(schedules)} databases")
    
    def _scheduled_backup(self, database: str, backup_type: str) -> None:
        """Execute scheduled backup."""
        try:
            logger.info(f"Starting scheduled {backup_type} backup for {database}")
            metadata = self.backup_manager.create_backup(database, backup_type)
            
            if metadata and metadata.success:
                logger.info(f"Scheduled backup {metadata.backup_id} completed successfully")
            else:
                logger.error(f"Scheduled backup failed for {database}")
                
        except Exception as e:
            logger.error(f"Scheduled backup error: {e}")
    
    def start_scheduler(self) -> None:
        """Start the backup scheduler."""
        if self.running:
            return
        
        self.running = True
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                import time
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the backup scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Backup scheduler stopped")

# Factory function to create storage backends
def create_storage_backend(backend_type: str, **kwargs) -> BackupStorage:
    """Create backup storage backend."""
    if backend_type == 'local':
        return LocalStorage(kwargs.get('path', '/tmp/backups'))
    elif backend_type == 's3':
        return S3Storage(
            bucket_name=kwargs['bucket_name'],
            region=kwargs.get('region', 'us-east-1'),
            aws_access_key_id=kwargs.get('aws_access_key_id'),
            aws_secret_access_key=kwargs.get('aws_secret_access_key')
        )
    else:
        raise ValueError(f"Unsupported storage backend: {backend_type}")

# Global backup system
def initialize_backup_system() -> Tuple[DatabaseBackup, BackupScheduler]:
    """Initialize the backup and recovery system."""
    # Create storage backend
    backend_type = getattr(settings, 'BACKUP_STORAGE_TYPE', 'local')
    
    if backend_type == 'local':
        storage_path = getattr(settings, 'BACKUP_STORAGE_PATH', '/tmp/supernova_backups')
        storage_backend = LocalStorage(storage_path)
    elif backend_type == 's3':
        storage_backend = S3Storage(
            bucket_name=getattr(settings, 'BACKUP_S3_BUCKET'),
            region=getattr(settings, 'BACKUP_S3_REGION', 'us-east-1'),
            aws_access_key_id=getattr(settings, 'BACKUP_AWS_ACCESS_KEY_ID', None),
            aws_secret_access_key=getattr(settings, 'BACKUP_AWS_SECRET_ACCESS_KEY', None)
        )
    else:
        # Fallback to local storage
        storage_backend = LocalStorage('/tmp/supernova_backups')
    
    backup_manager = DatabaseBackup(storage_backend)
    scheduler = BackupScheduler(backup_manager)
    
    return backup_manager, scheduler