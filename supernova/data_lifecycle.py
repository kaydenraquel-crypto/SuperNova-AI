"""
Data Lifecycle Management and Retention Policies

Comprehensive data lifecycle management including:
- Data retention policies and automatic cleanup
- Data archival and compression
- Compliance-based data handling
- Data anonymization and pseudonymization
- Audit trails for data lifecycle events
- Automated data tiering (hot/warm/cold)
"""

from __future__ import annotations
import logging
import json
import threading
import schedule
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import gzip
import shutil
import tempfile
import hashlib

from sqlalchemy import text, create_engine, MetaData
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import settings
    from .database_config import db_config, get_session
    from .backup_recovery import BackupStorage, LocalStorage, S3Storage
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    raise e

logger = logging.getLogger(__name__)

class DataCategory(Enum):
    """Data categories for lifecycle management."""
    PERSONAL = "personal"           # Personal identifiable information
    FINANCIAL = "financial"        # Financial transaction data
    OPERATIONAL = "operational"    # Operational/business data
    ANALYTICAL = "analytical"      # Analytics and metrics data
    SYSTEM = "system"              # System logs and metadata
    TEMPORARY = "temporary"        # Temporary/cache data

class DataTier(Enum):
    """Data storage tiers."""
    HOT = "hot"         # Frequently accessed, high-performance storage
    WARM = "warm"       # Occasionally accessed, standard storage
    COLD = "cold"       # Rarely accessed, archived storage
    FROZEN = "frozen"   # Long-term archival, compliance storage

class RetentionAction(Enum):
    """Actions to take when retention period expires."""
    DELETE = "delete"           # Permanently delete data
    ARCHIVE = "archive"         # Move to archive storage
    ANONYMIZE = "anonymize"     # Remove/hash personal identifiers
    COMPRESS = "compress"       # Compress and keep in place
    NOTIFY = "notify"          # Send notification for manual review

@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    name: str
    description: str
    table_name: str
    database: str
    category: DataCategory
    retention_days: int
    action: RetentionAction
    date_column: str = "created_at"
    where_clause: Optional[str] = None
    archive_location: Optional[str] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    records_processed: int = 0

@dataclass
class DataLifecycleEvent:
    """Data lifecycle event record."""
    event_id: str
    timestamp: datetime
    database: str
    table_name: str
    policy_name: str
    action: RetentionAction
    records_affected: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class DataAnonymizer:
    """Data anonymization and pseudonymization utilities."""
    
    @staticmethod
    def hash_field(value: str, salt: str = "supernova") -> str:
        """Hash a field value for pseudonymization."""
        if not value:
            return value
        
        hash_input = f"{value}{salt}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    @staticmethod
    def anonymize_email(email: str) -> str:
        """Anonymize email address."""
        if not email or "@" not in email:
            return "anonymized@example.com"
        
        local, domain = email.split("@", 1)
        return f"user_{DataAnonymizer.hash_field(email)[:8]}@{domain}"
    
    @staticmethod
    def anonymize_name(name: str) -> str:
        """Anonymize personal name."""
        if not name:
            return "Anonymous"
        
        return f"User_{DataAnonymizer.hash_field(name)[:8]}"
    
    @staticmethod
    def mask_ip_address(ip: str) -> str:
        """Mask IP address for privacy."""
        if not ip:
            return ip
        
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        
        return "masked.ip.address"

class DataArchiver:
    """Data archival system."""
    
    def __init__(self, storage_backend: BackupStorage):
        self.storage_backend = storage_backend
        self.temp_dir = Path(tempfile.gettempdir()) / 'supernova_archive'
        self.temp_dir.mkdir(exist_ok=True)
    
    def archive_data(
        self, 
        database: str, 
        table_name: str, 
        where_clause: str,
        archive_path: str,
        compression: bool = True
    ) -> Tuple[bool, int, str]:
        """
        Archive data to external storage.
        
        Returns:
            Tuple of (success, records_archived, archive_file_path)
        """
        try:
            # Export data to temporary file
            temp_file = self.temp_dir / f"archive_{table_name}_{int(datetime.utcnow().timestamp())}.json"
            
            records_archived = 0
            
            with get_session(database) as session:
                # Get data to archive
                query = f"SELECT * FROM {table_name}"
                if where_clause:
                    query += f" WHERE {where_clause}"
                
                result = session.execute(text(query))
                records = result.fetchall()
                
                if not records:
                    return True, 0, ""
                
                # Convert to JSON
                column_names = result.keys()
                data_to_archive = []
                
                for record in records:
                    record_dict = dict(zip(column_names, record))
                    # Convert datetime objects to ISO strings
                    for key, value in record_dict.items():
                        if isinstance(value, datetime):
                            record_dict[key] = value.isoformat()
                    data_to_archive.append(record_dict)
                
                records_archived = len(data_to_archive)
                
                # Write to temporary file
                with open(temp_file, 'w') as f:
                    json.dump({
                        'metadata': {
                            'table': table_name,
                            'database': database,
                            'archived_at': datetime.utcnow().isoformat(),
                            'where_clause': where_clause,
                            'record_count': records_archived
                        },
                        'data': data_to_archive
                    }, f, indent=2, default=str)
                
                # Compress if requested
                final_file = temp_file
                if compression:
                    compressed_file = temp_file.with_suffix('.json.gz')
                    with open(temp_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    temp_file.unlink()  # Remove uncompressed file
                    final_file = compressed_file
                
                # Upload to storage
                success = self.storage_backend.upload_backup(str(final_file), archive_path)
                
                # Cleanup temporary file
                if final_file.exists():
                    final_file.unlink()
                
                return success, records_archived, archive_path
                
        except Exception as e:
            logger.error(f"Error archiving data from {table_name}: {e}")
            return False, 0, str(e)
    
    def retrieve_archived_data(
        self,
        archive_path: str,
        output_file: Optional[Path] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieve archived data from storage.
        
        Returns:
            Tuple of (success, archived_data)
        """
        try:
            # Download from storage
            temp_file = self.temp_dir / f"retrieve_{int(datetime.utcnow().timestamp())}"
            
            if not self.storage_backend.download_backup(archive_path, str(temp_file)):
                return False, {}
            
            # Decompress if needed
            if archive_path.endswith('.gz'):
                decompressed_file = temp_file.with_suffix('.json')
                with gzip.open(temp_file, 'rt') as f_in:
                    with open(decompressed_file, 'w') as f_out:
                        f_out.write(f_in.read())
                temp_file.unlink()
                temp_file = decompressed_file
            
            # Load data
            with open(temp_file, 'r') as f:
                archived_data = json.load(f)
            
            # Copy to output file if specified
            if output_file:
                shutil.copy2(temp_file, output_file)
            
            # Cleanup
            temp_file.unlink()
            
            return True, archived_data
            
        except Exception as e:
            logger.error(f"Error retrieving archived data from {archive_path}: {e}")
            return False, {}

class RetentionPolicyEngine:
    """Data retention policy execution engine."""
    
    def __init__(self):
        self.policies: List[RetentionPolicy] = []
        self.event_history: List[DataLifecycleEvent] = []
        self.anonymizer = DataAnonymizer()
        
        # Create archiver with default storage
        storage_backend = self._create_storage_backend()
        self.archiver = DataArchiver(storage_backend)
        
        # Scheduling
        self.scheduler_active = False
        self.scheduler_thread = None
        
        # Load default policies
        self._load_default_policies()
    
    def _create_storage_backend(self) -> BackupStorage:
        """Create storage backend for archival."""
        backend_type = getattr(settings, 'ARCHIVE_STORAGE_TYPE', 'local')
        
        if backend_type == 'local':
            storage_path = getattr(settings, 'ARCHIVE_STORAGE_PATH', '/tmp/supernova_archives')
            return LocalStorage(storage_path)
        elif backend_type == 's3':
            return S3Storage(
                bucket_name=getattr(settings, 'ARCHIVE_S3_BUCKET', 'supernova-archives'),
                region=getattr(settings, 'ARCHIVE_S3_REGION', 'us-east-1'),
                aws_access_key_id=getattr(settings, 'ARCHIVE_AWS_ACCESS_KEY_ID', None),
                aws_secret_access_key=getattr(settings, 'ARCHIVE_AWS_SECRET_ACCESS_KEY', None)
            )
        else:
            return LocalStorage('/tmp/supernova_archives')
    
    def _load_default_policies(self) -> None:
        """Load default retention policies."""
        default_policies = [
            RetentionPolicy(
                name="Chat Messages Retention",
                description="Delete old chat messages after 1 year",
                table_name="conversation_messages",
                database="primary",
                category=DataCategory.OPERATIONAL,
                retention_days=365,
                action=RetentionAction.DELETE,
                date_column="timestamp"
            ),
            RetentionPolicy(
                name="Sentiment Data Archival",
                description="Archive old sentiment data after 90 days",
                table_name="sentiment_data",
                database="timescale",
                category=DataCategory.ANALYTICAL,
                retention_days=90,
                action=RetentionAction.ARCHIVE,
                date_column="timestamp"
            ),
            RetentionPolicy(
                name="System Logs Cleanup",
                description="Compress old system logs after 30 days",
                table_name="audit_log",
                database="primary",
                category=DataCategory.SYSTEM,
                retention_days=30,
                action=RetentionAction.COMPRESS,
                date_column="created_at"
            ),
            RetentionPolicy(
                name="Temporary Data Cleanup",
                description="Delete temporary session data after 1 day",
                table_name="conversation_sessions",
                database="primary",
                category=DataCategory.TEMPORARY,
                retention_days=1,
                action=RetentionAction.DELETE,
                date_column="created_at",
                where_clause="is_active = 0"
            ),
            RetentionPolicy(
                name="Old Alerts Cleanup",
                description="Delete resolved alerts after 30 days",
                table_name="sentiment_alerts",
                database="timescale",
                category=DataCategory.SYSTEM,
                retention_days=30,
                action=RetentionAction.DELETE,
                date_column="triggered_at",
                where_clause="resolved_at IS NOT NULL"
            ),
            RetentionPolicy(
                name="User Data Anonymization",
                description="Anonymize inactive user data after 2 years",
                table_name="users",
                database="primary",
                category=DataCategory.PERSONAL,
                retention_days=730,
                action=RetentionAction.ANONYMIZE,
                date_column="last_login",
                where_clause="last_login < NOW() - INTERVAL '2 years'"
            )
        ]
        
        self.policies.extend(default_policies)
    
    def add_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy."""
        # Check if policy with same name already exists
        existing_policy = next((p for p in self.policies if p.name == policy.name), None)
        if existing_policy:
            # Update existing policy
            self.policies.remove(existing_policy)
        
        self.policies.append(policy)
        logger.info(f"Added retention policy: {policy.name}")
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a retention policy."""
        policy = next((p for p in self.policies if p.name == policy_name), None)
        if policy:
            self.policies.remove(policy)
            logger.info(f"Removed retention policy: {policy_name}")
            return True
        return False
    
    def execute_policy(self, policy_name: str) -> DataLifecycleEvent:
        """Execute a specific retention policy."""
        policy = next((p for p in self.policies if p.name == policy_name), None)
        
        if not policy:
            raise ValueError(f"Policy '{policy_name}' not found")
        
        if not policy.enabled:
            logger.info(f"Policy '{policy_name}' is disabled, skipping")
            return self._create_event(policy, 0, True, "Policy disabled")
        
        logger.info(f"Executing retention policy: {policy_name}")
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_days)
            
            # Build WHERE clause
            where_conditions = [f"{policy.date_column} < '{cutoff_date.isoformat()}'"]
            
            if policy.where_clause:
                where_conditions.append(policy.where_clause)
            
            where_clause = " AND ".join(where_conditions)
            
            # Execute action
            if policy.action == RetentionAction.DELETE:
                records_affected = self._delete_records(policy, where_clause)
            elif policy.action == RetentionAction.ARCHIVE:
                records_affected = self._archive_records(policy, where_clause)
            elif policy.action == RetentionAction.ANONYMIZE:
                records_affected = self._anonymize_records(policy, where_clause)
            elif policy.action == RetentionAction.COMPRESS:
                records_affected = self._compress_records(policy, where_clause)
            elif policy.action == RetentionAction.NOTIFY:
                records_affected = self._notify_records(policy, where_clause)
            else:
                raise ValueError(f"Unknown retention action: {policy.action}")
            
            # Update policy statistics
            policy.last_run = datetime.utcnow()
            policy.records_processed += records_affected
            
            # Create success event
            event = self._create_event(policy, records_affected, True)
            self.event_history.append(event)
            
            logger.info(f"Policy '{policy_name}' executed successfully, {records_affected} records affected")
            return event
            
        except Exception as e:
            logger.error(f"Error executing policy '{policy_name}': {e}")
            
            # Create failure event
            event = self._create_event(policy, 0, False, str(e))
            self.event_history.append(event)
            return event
    
    def _delete_records(self, policy: RetentionPolicy, where_clause: str) -> int:
        """Delete records based on policy."""
        with get_session(policy.database) as session:
            # Count records first
            count_query = f"SELECT COUNT(*) FROM {policy.table_name} WHERE {where_clause}"
            result = session.execute(text(count_query))
            count = result.scalar() or 0
            
            if count == 0:
                return 0
            
            # Delete records
            delete_query = f"DELETE FROM {policy.table_name} WHERE {where_clause}"
            session.execute(text(delete_query))
            session.commit()
            
            return count
    
    def _archive_records(self, policy: RetentionPolicy, where_clause: str) -> int:
        """Archive records to external storage."""
        # Generate archive path
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        archive_path = f"{policy.database}/{policy.table_name}/{policy.name}_{timestamp}.json.gz"
        
        # Archive data
        success, records_archived, _ = self.archiver.archive_data(
            database=policy.database,
            table_name=policy.table_name,
            where_clause=where_clause,
            archive_path=archive_path,
            compression=True
        )
        
        if success and records_archived > 0:
            # Delete archived records from primary table
            with get_session(policy.database) as session:
                delete_query = f"DELETE FROM {policy.table_name} WHERE {where_clause}"
                session.execute(text(delete_query))
                session.commit()
        
        return records_archived if success else 0
    
    def _anonymize_records(self, policy: RetentionPolicy, where_clause: str) -> int:
        """Anonymize records by removing/hashing personal identifiers."""
        anonymized_count = 0
        
        try:
            with get_session(policy.database) as session:
                # Get records to anonymize
                select_query = f"SELECT * FROM {policy.table_name} WHERE {where_clause}"
                result = session.execute(text(select_query))
                records = result.fetchall()
                column_names = result.keys()
                
                for record in records:
                    record_dict = dict(zip(column_names, record))
                    
                    # Apply anonymization based on table and column names
                    updates = {}
                    
                    # Common personal data fields to anonymize
                    if 'email' in record_dict and record_dict['email']:
                        updates['email'] = self.anonymizer.anonymize_email(record_dict['email'])
                    
                    if 'name' in record_dict and record_dict['name']:
                        updates['name'] = self.anonymizer.anonymize_name(record_dict['name'])
                    
                    if 'first_name' in record_dict and record_dict['first_name']:
                        updates['first_name'] = self.anonymizer.anonymize_name(record_dict['first_name'])
                    
                    if 'last_name' in record_dict and record_dict['last_name']:
                        updates['last_name'] = self.anonymizer.anonymize_name(record_dict['last_name'])
                    
                    if 'username' in record_dict and record_dict['username']:
                        updates['username'] = f"user_{self.anonymizer.hash_field(record_dict['username'])[:8]}"
                    
                    # Apply updates if any
                    if updates:
                        # Build update query
                        set_clause = ", ".join([f"{col} = '{val}'" for col, val in updates.items()])
                        
                        # Get primary key for WHERE clause
                        pk_column = 'id'  # Assume 'id' is primary key
                        pk_value = record_dict.get(pk_column)
                        
                        if pk_value:
                            update_query = f"UPDATE {policy.table_name} SET {set_clause} WHERE {pk_column} = {pk_value}"
                            session.execute(text(update_query))
                            anonymized_count += 1
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error anonymizing records: {e}")
            raise e
        
        return anonymized_count
    
    def _compress_records(self, policy: RetentionPolicy, where_clause: str) -> int:
        """Compress records (implementation depends on database features)."""
        # For now, this is a placeholder
        # In a real implementation, this might involve database-specific compression
        logger.info(f"Compression requested for {policy.table_name} (not implemented)")
        return 0
    
    def _notify_records(self, policy: RetentionPolicy, where_clause: str) -> int:
        """Send notification about records needing attention."""
        with get_session(policy.database) as session:
            # Count records
            count_query = f"SELECT COUNT(*) FROM {policy.table_name} WHERE {where_clause}"
            result = session.execute(text(count_query))
            count = result.scalar() or 0
            
            if count > 0:
                # Send notification (placeholder - integrate with notification system)
                logger.warning(
                    f"Data retention notification: {count} records in {policy.table_name} "
                    f"need attention according to policy '{policy.name}'"
                )
            
            return count
    
    def _create_event(
        self,
        policy: RetentionPolicy,
        records_affected: int,
        success: bool,
        error_message: str = None
    ) -> DataLifecycleEvent:
        """Create a data lifecycle event record."""
        event_id = f"{policy.name}_{int(datetime.utcnow().timestamp())}"
        
        return DataLifecycleEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            database=policy.database,
            table_name=policy.table_name,
            policy_name=policy.name,
            action=policy.action,
            records_affected=records_affected,
            success=success,
            error_message=error_message,
            metadata={
                'retention_days': policy.retention_days,
                'category': policy.category.value,
                'where_clause': getattr(policy, 'where_clause', None)
            }
        )
    
    def execute_all_policies(self) -> List[DataLifecycleEvent]:
        """Execute all enabled policies."""
        events = []
        
        for policy in self.policies:
            if policy.enabled:
                try:
                    event = self.execute_policy(policy.name)
                    events.append(event)
                except Exception as e:
                    logger.error(f"Error executing policy {policy.name}: {e}")
                    # Create error event
                    error_event = self._create_event(policy, 0, False, str(e))
                    events.append(error_event)
        
        return events
    
    def start_scheduler(self, daily_execution_time: str = "02:00") -> None:
        """Start automatic policy execution scheduler."""
        if self.scheduler_active:
            logger.warning("Scheduler is already active")
            return
        
        # Schedule daily execution
        schedule.every().day.at(daily_execution_time).do(self._scheduled_execution)
        
        self.scheduler_active = True
        
        def run_scheduler():
            while self.scheduler_active:
                schedule.run_pending()
                import time
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Data lifecycle scheduler started (daily execution at {daily_execution_time})")
    
    def stop_scheduler(self) -> None:
        """Stop the policy execution scheduler."""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        logger.info("Data lifecycle scheduler stopped")
    
    def _scheduled_execution(self) -> None:
        """Scheduled execution of all policies."""
        try:
            logger.info("Starting scheduled data lifecycle policy execution")
            events = self.execute_all_policies()
            
            # Log summary
            total_records = sum(event.records_affected for event in events)
            successful_policies = sum(1 for event in events if event.success)
            
            logger.info(
                f"Scheduled execution completed: {successful_policies}/{len(events)} policies successful, "
                f"{total_records} total records processed"
            )
            
        except Exception as e:
            logger.error(f"Error in scheduled policy execution: {e}")
    
    def get_policy_status(self) -> Dict[str, Any]:
        """Get status of all retention policies."""
        return {
            'total_policies': len(self.policies),
            'enabled_policies': len([p for p in self.policies if p.enabled]),
            'policies': [asdict(policy) for policy in self.policies],
            'scheduler_active': self.scheduler_active,
            'total_events': len(self.event_history),
            'recent_events': [asdict(event) for event in self.event_history[-10:]]  # Last 10 events
        }
    
    def get_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report for data lifecycle activities."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_events = [
            event for event in self.event_history
            if event.timestamp > cutoff_date
        ]
        
        # Group by action type
        actions_summary = {}
        for event in recent_events:
            action = event.action.value
            if action not in actions_summary:
                actions_summary[action] = {
                    'count': 0,
                    'records_affected': 0,
                    'successful': 0,
                    'failed': 0
                }
            
            actions_summary[action]['count'] += 1
            actions_summary[action]['records_affected'] += event.records_affected
            
            if event.success:
                actions_summary[action]['successful'] += 1
            else:
                actions_summary[action]['failed'] += 1
        
        # Group by database
        database_summary = {}
        for event in recent_events:
            db = event.database
            if db not in database_summary:
                database_summary[db] = {
                    'policies_executed': 0,
                    'records_affected': 0
                }
            
            database_summary[db]['policies_executed'] += 1
            database_summary[db]['records_affected'] += event.records_affected
        
        return {
            'report_period_days': days,
            'report_generated': datetime.utcnow().isoformat(),
            'total_events': len(recent_events),
            'actions_summary': actions_summary,
            'database_summary': database_summary,
            'compliance_status': 'compliant' if len(recent_events) > 0 else 'no_activity',
            'recent_events': [asdict(event) for event in recent_events[-20:]]  # Last 20 events
        }

# Global lifecycle manager
data_lifecycle_manager = RetentionPolicyEngine()

# Convenience functions
def add_retention_policy(policy: RetentionPolicy) -> None:
    """Add a retention policy."""
    data_lifecycle_manager.add_policy(policy)

def execute_retention_policy(policy_name: str) -> DataLifecycleEvent:
    """Execute a specific retention policy."""
    return data_lifecycle_manager.execute_policy(policy_name)

def execute_all_retention_policies() -> List[DataLifecycleEvent]:
    """Execute all retention policies."""
    return data_lifecycle_manager.execute_all_policies()

def start_lifecycle_scheduler(daily_time: str = "02:00") -> None:
    """Start automatic data lifecycle management."""
    data_lifecycle_manager.start_scheduler(daily_time)

def stop_lifecycle_scheduler() -> None:
    """Stop automatic data lifecycle management."""
    data_lifecycle_manager.stop_scheduler()

def get_lifecycle_status() -> Dict[str, Any]:
    """Get data lifecycle management status."""
    return data_lifecycle_manager.get_policy_status()

def generate_compliance_report(days: int = 30) -> Dict[str, Any]:
    """Generate compliance report."""
    return data_lifecycle_manager.get_compliance_report(days)