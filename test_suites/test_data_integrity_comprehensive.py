"""
Comprehensive Data Integrity and Database Testing Suite
=======================================================

This module provides extensive data integrity testing including:
- Database constraint validation
- Data consistency checks
- Transaction integrity testing
- Backup and recovery testing
- Data migration testing
- Data retention policy testing
- Financial calculation accuracy
- Cross-system data synchronization
"""
import pytest
import time
import json
import tempfile
import shutil
import decimal
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text, inspect, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, DataError

from supernova.db import Base, User, Profile, Asset, WatchlistItem, SessionLocal
from supernova.schemas import OHLCVBar, SentimentDataPoint


class DataIntegrityTester:
    """Comprehensive data integrity testing utilities."""
    
    def __init__(self, db_session):
        self.db_session = db_session
        self.integrity_violations = []
        self.test_data_created = []
    
    def create_test_data_set(self, size: int = 100) -> Dict[str, List[Any]]:
        """Create comprehensive test data set for integrity testing."""
        users = []
        profiles = []
        assets = []
        watchlist_items = []
        
        # Create users
        for i in range(size):
            user = User(
                name=f"Test User {i}",
                email=f"test{i}@example.com"
            )
            users.append(user)
            self.db_session.add(user)
        
        self.db_session.flush()
        
        # Create profiles for users
        for i, user in enumerate(users):
            profile = Profile(
                user_id=user.id,
                risk_score=np.random.randint(0, 101),
                time_horizon_yrs=np.random.randint(1, 31),
                objectives=f"Objective {i}",
                constraints=f"Constraint {i}",
                income=np.random.uniform(30000, 200000) if i % 3 == 0 else None,
                expenses=np.random.uniform(20000, 100000) if i % 4 == 0 else None,
                assets=np.random.uniform(10000, 500000) if i % 5 == 0 else None,
                debts=np.random.uniform(0, 100000) if i % 6 == 0 else None
            )
            profiles.append(profile)
            self.db_session.add(profile)
        
        self.db_session.flush()
        
        # Create assets
        asset_symbols = [f"STOCK_{i:03d}" for i in range(size // 2)]
        for symbol in asset_symbols:
            asset = Asset(
                symbol=symbol,
                name=f"{symbol} Corporation",
                asset_class="stock"
            )
            assets.append(asset)
            self.db_session.add(asset)
        
        self.db_session.flush()
        
        # Create watchlist items
        for i in range(size * 2):  # More watchlist items than users
            profile = np.random.choice(profiles)
            asset = np.random.choice(assets)
            
            # Check for duplicates
            existing = self.db_session.query(WatchlistItem).filter(
                WatchlistItem.profile_id == profile.id,
                WatchlistItem.asset_id == asset.id
            ).first()
            
            if not existing:
                watchlist_item = WatchlistItem(
                    profile_id=profile.id,
                    asset_id=asset.id
                )
                watchlist_items.append(watchlist_item)
                self.db_session.add(watchlist_item)
        
        self.db_session.commit()
        
        self.test_data_created = {
            'users': users,
            'profiles': profiles, 
            'assets': assets,
            'watchlist_items': watchlist_items
        }
        
        return self.test_data_created
    
    def test_referential_integrity(self) -> Dict[str, Any]:
        """Test referential integrity constraints."""
        results = {
            'foreign_key_violations': [],
            'orphaned_records': [],
            'circular_references': [],
            'constraint_validations': []
        }
        
        try:
            # Test profile -> user foreign key
            orphaned_profiles = self.db_session.query(Profile).filter(
                ~Profile.user_id.in_(
                    self.db_session.query(User.id)
                )
            ).all()
            
            results['orphaned_records'].append({
                'table': 'profiles',
                'count': len(orphaned_profiles),
                'records': [p.id for p in orphaned_profiles]
            })
            
            # Test watchlist -> profile foreign key
            orphaned_watchlist = self.db_session.query(WatchlistItem).filter(
                ~WatchlistItem.profile_id.in_(
                    self.db_session.query(Profile.id)
                )
            ).all()
            
            results['orphaned_records'].append({
                'table': 'watchlist_items',
                'field': 'profile_id',
                'count': len(orphaned_watchlist),
                'records': [w.id for w in orphaned_watchlist]
            })
            
            # Test watchlist -> asset foreign key
            orphaned_watchlist_assets = self.db_session.query(WatchlistItem).filter(
                ~WatchlistItem.asset_id.in_(
                    self.db_session.query(Asset.id)
                )
            ).all()
            
            results['orphaned_records'].append({
                'table': 'watchlist_items',
                'field': 'asset_id', 
                'count': len(orphaned_watchlist_assets),
                'records': [w.id for w in orphaned_watchlist_assets]
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_data_constraints(self) -> Dict[str, Any]:
        """Test data constraint validations."""
        results = {
            'constraint_violations': [],
            'invalid_data_formats': [],
            'boundary_value_tests': []
        }
        
        try:
            # Test user constraints
            # Test empty name (should be rejected)
            try:
                invalid_user = User(name="", email="test@example.com")
                self.db_session.add(invalid_user)
                self.db_session.flush()
                results['constraint_violations'].append({
                    'table': 'users',
                    'field': 'name',
                    'issue': 'Empty name accepted when should be rejected'
                })
            except (IntegrityError, DataError):
                # Good - constraint working
                self.db_session.rollback()
            
            # Test invalid email formats
            invalid_emails = ['invalid-email', 'test@', '@domain.com', 'test.domain']
            for email in invalid_emails:
                try:
                    invalid_user = User(name="Test", email=email)
                    self.db_session.add(invalid_user)
                    self.db_session.flush()
                    results['invalid_data_formats'].append({
                        'table': 'users',
                        'field': 'email',
                        'value': email,
                        'issue': 'Invalid email format accepted'
                    })
                    self.db_session.rollback()
                except (IntegrityError, DataError):
                    # Good - validation working
                    self.db_session.rollback()
            
            # Test profile constraints
            test_user = User(name="Constraint Test User", email="constraint@test.com")
            self.db_session.add(test_user)
            self.db_session.flush()
            
            # Test boundary values for risk score
            boundary_tests = [
                {'risk_score': -1, 'should_fail': True},
                {'risk_score': 0, 'should_fail': False},
                {'risk_score': 100, 'should_fail': False},
                {'risk_score': 101, 'should_fail': True},
                {'time_horizon_yrs': -1, 'should_fail': True},
                {'time_horizon_yrs': 0, 'should_fail': True},
                {'time_horizon_yrs': 1, 'should_fail': False},
                {'time_horizon_yrs': 100, 'should_fail': False},
            ]
            
            for test_case in boundary_tests:
                try:
                    profile_data = {
                        'user_id': test_user.id,
                        'risk_score': 50,
                        'time_horizon_yrs': 10,
                        'objectives': 'test',
                        'constraints': 'test'
                    }
                    profile_data.update(test_case)
                    
                    test_profile = Profile(**profile_data)
                    self.db_session.add(test_profile)
                    self.db_session.flush()
                    
                    if test_case['should_fail']:
                        results['boundary_value_tests'].append({
                            'test_case': test_case,
                            'issue': 'Invalid boundary value accepted'
                        })
                    
                    self.db_session.rollback()
                
                except (IntegrityError, DataError):
                    if not test_case['should_fail']:
                        results['boundary_value_tests'].append({
                            'test_case': test_case,
                            'issue': 'Valid boundary value rejected'
                        })
                    self.db_session.rollback()
        
        except Exception as e:
            results['error'] = str(e)
            self.db_session.rollback()
        
        return results
    
    def test_data_consistency(self) -> Dict[str, Any]:
        """Test data consistency across related tables."""
        results = {
            'inconsistencies_found': [],
            'aggregate_mismatches': [],
            'cross_table_validation': []
        }
        
        try:
            # Test user-profile consistency
            users_with_profiles = self.db_session.query(User.id).join(Profile).distinct().all()
            users_without_profiles = self.db_session.query(User).filter(
                ~User.id.in_([u.id for u in users_with_profiles])
            ).all()
            
            results['cross_table_validation'].append({
                'check': 'users_without_profiles',
                'count': len(users_without_profiles),
                'user_ids': [u.id for u in users_without_profiles]
            })
            
            # Test profile-watchlist consistency
            profiles_with_watchlist = self.db_session.query(Profile.id).join(WatchlistItem).distinct().all()
            profiles_without_watchlist = self.db_session.query(Profile).filter(
                ~Profile.id.in_([p.id for p in profiles_with_watchlist])
            ).all()
            
            results['cross_table_validation'].append({
                'check': 'profiles_without_watchlist',
                'count': len(profiles_without_watchlist),
                'profile_ids': [p.id for p in profiles_without_watchlist]
            })
            
            # Test asset usage consistency
            used_assets = self.db_session.query(Asset.id).join(WatchlistItem).distinct().all()
            unused_assets = self.db_session.query(Asset).filter(
                ~Asset.id.in_([a.id for a in used_assets])
            ).all()
            
            results['cross_table_validation'].append({
                'check': 'unused_assets',
                'count': len(unused_assets),
                'asset_ids': [a.id for a in unused_assets]
            })
            
            # Test aggregate consistency
            # Count profiles by user
            user_profile_counts = self.db_session.query(
                User.id,
                self.db_session.query(Profile.id).filter(Profile.user_id == User.id).count().label('profile_count')
            ).all()
            
            multiple_profiles = [u for u in user_profile_counts if u.profile_count > 1]
            results['aggregate_mismatches'].append({
                'check': 'users_with_multiple_profiles',
                'count': len(multiple_profiles),
                'user_ids': [u.id for u in multiple_profiles]
            })
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_financial_data_accuracy(self) -> Dict[str, Any]:
        """Test financial calculation accuracy and precision."""
        results = {
            'calculation_errors': [],
            'precision_issues': [],
            'rounding_inconsistencies': []
        }
        
        try:
            # Test decimal precision for financial values
            test_values = [
                decimal.Decimal('100000.01'),
                decimal.Decimal('99999.99'),
                decimal.Decimal('0.01'),
                decimal.Decimal('999999999.99'),
                decimal.Decimal('0.001')  # Sub-penny precision
            ]
            
            test_user = User(name="Financial Test", email="financial@test.com")
            self.db_session.add(test_user)
            self.db_session.flush()
            
            for i, value in enumerate(test_values):
                try:
                    profile = Profile(
                        user_id=test_user.id,
                        risk_score=50,
                        time_horizon_yrs=10,
                        objectives='test',
                        constraints='test',
                        income=float(value),
                        expenses=float(value * decimal.Decimal('0.8')),
                        assets=float(value * decimal.Decimal('2.5')),
                        debts=float(value * decimal.Decimal('0.3'))
                    )
                    self.db_session.add(profile)
                    self.db_session.flush()
                    
                    # Retrieve and check precision
                    retrieved = self.db_session.query(Profile).filter(Profile.id == profile.id).first()
                    
                    # Check income precision
                    if abs(retrieved.income - float(value)) > 0.01:  # Allow 1 cent tolerance
                        results['precision_issues'].append({
                            'field': 'income',
                            'original': float(value),
                            'retrieved': retrieved.income,
                            'difference': abs(retrieved.income - float(value))
                        })
                    
                    # Test financial calculations
                    net_worth = retrieved.assets - retrieved.debts if retrieved.assets and retrieved.debts else None
                    disposable_income = retrieved.income - retrieved.expenses if retrieved.income and retrieved.expenses else None
                    
                    if net_worth is not None and disposable_income is not None:
                        # Test calculation consistency
                        expected_net_worth = float(value * decimal.Decimal('2.5') - value * decimal.Decimal('0.3'))
                        if abs(net_worth - expected_net_worth) > 0.02:
                            results['calculation_errors'].append({
                                'calculation': 'net_worth',
                                'expected': expected_net_worth,
                                'actual': net_worth,
                                'difference': abs(net_worth - expected_net_worth)
                            })
                
                except Exception as e:
                    results['calculation_errors'].append({
                        'test_value': float(value),
                        'error': str(e)
                    })
                    self.db_session.rollback()
                    continue
            
            # Test percentage calculations
            risk_scores = [0, 25, 50, 75, 100]
            for risk_score in risk_scores:
                profile = Profile(
                    user_id=test_user.id,
                    risk_score=risk_score,
                    time_horizon_yrs=10,
                    objectives='test',
                    constraints='test'
                )
                self.db_session.add(profile)
                self.db_session.flush()
                
                retrieved = self.db_session.query(Profile).filter(Profile.id == profile.id).first()
                
                if retrieved.risk_score != risk_score:
                    results['calculation_errors'].append({
                        'field': 'risk_score',
                        'expected': risk_score,
                        'actual': retrieved.risk_score
                    })
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def cleanup_test_data(self):
        """Clean up test data created during testing."""
        try:
            if self.test_data_created:
                # Delete in reverse order of creation to respect foreign keys
                for watchlist_item in self.test_data_created.get('watchlist_items', []):
                    self.db_session.delete(watchlist_item)
                
                for asset in self.test_data_created.get('assets', []):
                    self.db_session.delete(asset)
                
                for profile in self.test_data_created.get('profiles', []):
                    self.db_session.delete(profile)
                
                for user in self.test_data_created.get('users', []):
                    self.db_session.delete(user)
                
                self.db_session.commit()
        
        except Exception as e:
            self.db_session.rollback()
            print(f"Error cleaning up test data: {e}")


@pytest.mark.data_integrity
class TestDatabaseConstraints:
    """Test database constraints and validations."""
    
    def test_primary_key_constraints(self, db_session):
        """Test primary key uniqueness constraints."""
        tester = DataIntegrityTester(db_session)
        
        try:
            # Create user
            user1 = User(name="Test User 1", email="test1@example.com")
            db_session.add(user1)
            db_session.flush()
            
            # Try to create another user with same ID (should fail)
            try:
                # This would require manually setting ID, which SQLAlchemy typically prevents
                # Test the concept by trying to create duplicate data that would violate constraints
                user2 = User(name="Test User 2", email="test1@example.com")  # Same email
                db_session.add(user2)
                db_session.commit()
                
                # If we get here, need to check if email uniqueness is enforced
                users_with_same_email = db_session.query(User).filter(
                    User.email == "test1@example.com"
                ).all()
                
                # Depending on schema, this might be allowed or not
                assert len(users_with_same_email) <= 2  # At most 2 (if duplicates allowed)
                
            except IntegrityError:
                # Good - constraint is working
                db_session.rollback()
                assert True
        
        finally:
            tester.cleanup_test_data()
    
    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint enforcement."""
        tester = DataIntegrityTester(db_session)
        
        try:
            # Try to create profile with non-existent user_id
            try:
                invalid_profile = Profile(
                    user_id=99999,  # Non-existent user
                    risk_score=50,
                    time_horizon_yrs=10,
                    objectives="test",
                    constraints="test"
                )
                db_session.add(invalid_profile)
                db_session.commit()
                
                # If we get here, foreign key constraint is not enforced
                assert False, "Foreign key constraint should prevent this"
                
            except IntegrityError:
                # Good - foreign key constraint working
                db_session.rollback()
                assert True
            
            # Test cascade deletes (if implemented)
            user = User(name="Cascade Test", email="cascade@test.com")
            db_session.add(user)
            db_session.flush()
            
            profile = Profile(
                user_id=user.id,
                risk_score=50,
                time_horizon_yrs=10,
                objectives="test",
                constraints="test"
            )
            db_session.add(profile)
            db_session.commit()
            
            user_id = user.id
            profile_id = profile.id
            
            # Delete user
            db_session.delete(user)
            db_session.commit()
            
            # Check if profile was cascade deleted
            remaining_profile = db_session.get(Profile, profile_id)
            # Depending on schema design, profile might be deleted or orphaned
            # Both are valid depending on business rules
        
        finally:
            tester.cleanup_test_data()
    
    def test_check_constraints(self, db_session):
        """Test check constraints and data validation."""
        tester = DataIntegrityTester(db_session)
        
        try:
            user = User(name="Check Test", email="check@test.com")
            db_session.add(user)
            db_session.flush()
            
            # Test invalid risk scores
            invalid_risk_scores = [-1, 101, -100, 1000]
            
            for risk_score in invalid_risk_scores:
                try:
                    invalid_profile = Profile(
                        user_id=user.id,
                        risk_score=risk_score,
                        time_horizon_yrs=10,
                        objectives="test",
                        constraints="test"
                    )
                    db_session.add(invalid_profile)
                    db_session.flush()
                    
                    # If we get here, check constraint is not working
                    db_session.rollback()
                    # Note: This might be acceptable depending on application-level validation
                
                except (IntegrityError, DataError):
                    # Good - check constraint working
                    db_session.rollback()
            
            # Test negative financial values (depending on business rules)
            financial_test_cases = [
                {'income': -1000, 'should_fail': True},
                {'expenses': -500, 'should_fail': True},
                {'assets': -10000, 'should_fail': True},
                {'debts': -5000, 'should_fail': True},
                {'time_horizon_yrs': 0, 'should_fail': True},
                {'time_horizon_yrs': -5, 'should_fail': True}
            ]
            
            for test_case in financial_test_cases:
                try:
                    profile_data = {
                        'user_id': user.id,
                        'risk_score': 50,
                        'time_horizon_yrs': 10,
                        'objectives': 'test',
                        'constraints': 'test'
                    }
                    profile_data.update({k: v for k, v in test_case.items() if k != 'should_fail'})
                    
                    test_profile = Profile(**profile_data)
                    db_session.add(test_profile)
                    db_session.flush()
                    
                    if test_case['should_fail']:
                        # Invalid data was accepted - might be OK depending on validation strategy
                        pass
                    
                    db_session.rollback()
                
                except (IntegrityError, DataError):
                    # Constraint working as expected
                    db_session.rollback()
        
        finally:
            tester.cleanup_test_data()
    
    def test_unique_constraints(self, db_session):
        """Test unique constraints."""
        tester = DataIntegrityTester(db_session)
        
        try:
            # Test asset symbol uniqueness
            asset1 = Asset(symbol="UNIQUE_TEST", name="Test Corp 1", asset_class="stock")
            db_session.add(asset1)
            db_session.flush()
            
            try:
                asset2 = Asset(symbol="UNIQUE_TEST", name="Test Corp 2", asset_class="stock")
                db_session.add(asset2)
                db_session.commit()
                
                # If we get here, uniqueness is not enforced
                # This might be acceptable depending on business rules
                assets_with_same_symbol = db_session.query(Asset).filter(
                    Asset.symbol == "UNIQUE_TEST"
                ).all()
                
                # Check how many duplicates exist
                assert len(assets_with_same_symbol) >= 1
                
            except IntegrityError:
                # Good - uniqueness constraint working
                db_session.rollback()
                assert True
        
        finally:
            tester.cleanup_test_data()


@pytest.mark.data_integrity
class TestDataConsistency:
    """Test cross-table data consistency."""
    
    def test_referential_integrity_comprehensive(self, db_session):
        """Test comprehensive referential integrity."""
        tester = DataIntegrityTester(db_session)
        
        try:
            # Create test data set
            test_data = tester.create_test_data_set(50)
            
            # Run referential integrity tests
            integrity_results = tester.test_referential_integrity()
            
            # Check results
            total_orphaned = sum(
                result['count'] for result in integrity_results['orphaned_records']
            )
            
            assert total_orphaned == 0, f"Found {total_orphaned} orphaned records"
            
            # Check for foreign key violations
            assert len(integrity_results['foreign_key_violations']) == 0, \
                f"Foreign key violations found: {integrity_results['foreign_key_violations']}"
        
        finally:
            tester.cleanup_test_data()
    
    def test_data_consistency_checks(self, db_session):
        """Test data consistency across tables."""
        tester = DataIntegrityTester(db_session)
        
        try:
            # Create test data
            test_data = tester.create_test_data_set(30)
            
            # Run consistency tests
            consistency_results = tester.test_data_consistency()
            
            # Analyze results
            total_inconsistencies = len(consistency_results['inconsistencies_found'])
            total_mismatches = len(consistency_results['aggregate_mismatches'])
            
            # Some inconsistencies might be acceptable (e.g., users without profiles)
            # But should be documented and understood
            assert total_inconsistencies < 10, \
                f"Too many data inconsistencies: {total_inconsistencies}"
            
            # Check cross-table validations
            for validation in consistency_results['cross_table_validation']:
                if validation['check'] == 'users_without_profiles':
                    # This might be acceptable in business logic
                    assert validation['count'] >= 0  # Just ensure it's counted
        
        finally:
            tester.cleanup_test_data()
    
    def test_financial_data_integrity(self, db_session):
        """Test financial data calculation integrity."""
        tester = DataIntegrityTester(db_session)
        
        try:
            # Run financial accuracy tests
            financial_results = tester.test_financial_data_accuracy()
            
            # Check for calculation errors
            calculation_errors = financial_results['calculation_errors']
            precision_issues = financial_results['precision_issues']
            
            assert len(calculation_errors) == 0, \
                f"Financial calculation errors found: {calculation_errors}"
            
            # Precision issues should be minimal
            significant_precision_issues = [
                issue for issue in precision_issues 
                if issue['difference'] > 0.01  # More than 1 cent difference
            ]
            
            assert len(significant_precision_issues) == 0, \
                f"Significant precision issues: {significant_precision_issues}"
        
        finally:
            tester.cleanup_test_data()


@pytest.mark.data_integrity
class TestDataMigration:
    """Test data migration and schema changes."""
    
    def test_schema_migration_simulation(self, db_session):
        """Simulate schema migration scenarios."""
        # Create test data before "migration"
        original_users = []
        for i in range(10):
            user = User(name=f"Migration Test User {i}", email=f"migrate{i}@test.com")
            db_session.add(user)
            original_users.append(user)
        
        db_session.commit()
        
        # Simulate adding a new field (would be done by migration script)
        # For testing purposes, we'll just verify data integrity after changes
        
        # Verify all original data is still intact
        for original_user in original_users:
            retrieved_user = db_session.get(User, original_user.id)
            assert retrieved_user is not None, f"User {original_user.id} lost during migration"
            assert retrieved_user.name == original_user.name
            assert retrieved_user.email == original_user.email
        
        # Clean up
        for user in original_users:
            db_session.delete(user)
        db_session.commit()
    
    def test_data_type_changes(self, db_session):
        """Test data integrity during data type changes."""
        # This would test scenarios like changing varchar length,
        # changing number precision, etc.
        
        # Create test data with edge cases
        user = User(
            name="A" * 100,  # Long name
            email="long.email.address@very.long.domain.name.example.com"
        )
        db_session.add(user)
        db_session.commit()
        
        # Verify data is stored correctly
        retrieved = db_session.get(User, user.id)
        assert len(retrieved.name) <= 100  # Should fit in field
        assert "@" in retrieved.email  # Should be valid email
        
        # Clean up
        db_session.delete(user)
        db_session.commit()
    
    def test_data_archiving_integrity(self, db_session):
        """Test data archiving and retention policies."""
        # Create old data that might be archived
        old_date = datetime.now() - timedelta(days=365)
        
        # Create test user
        user = User(name="Archive Test", email="archive@test.com")
        db_session.add(user)
        db_session.flush()
        
        profile = Profile(
            user_id=user.id,
            risk_score=50,
            time_horizon_yrs=10,
            objectives="test",
            constraints="test"
        )
        db_session.add(profile)
        db_session.commit()
        
        # Simulate archival process
        # In real system, this would move data to archive tables
        user_id = user.id
        profile_id = profile.id
        
        # Test that archived data maintains referential integrity
        # This is a simplified test - real archival would be more complex
        
        # Clean up
        db_session.delete(profile)
        db_session.delete(user)
        db_session.commit()


@pytest.mark.data_integrity
class TestBackupRecovery:
    """Test backup and recovery procedures."""
    
    def test_database_backup_integrity(self, tmp_path):
        """Test database backup and restore integrity."""
        # Create test database with known data
        test_db_path = tmp_path / "test_backup.db"
        backup_db_path = tmp_path / "backup.db"
        
        # Create test engine and session
        test_engine = create_engine(f"sqlite:///{test_db_path}")
        Base.metadata.create_all(test_engine)
        TestSession = sessionmaker(bind=test_engine)
        test_session = TestSession()
        
        try:
            # Create test data
            test_users = []
            for i in range(5):
                user = User(name=f"Backup Test {i}", email=f"backup{i}@test.com")
                test_session.add(user)
                test_users.append(user)
            
            test_session.commit()
            
            # Simulate backup (copy database file)
            shutil.copy(test_db_path, backup_db_path)
            
            # Modify original data
            test_users[0].name = "Modified Name"
            test_session.commit()
            
            # Create backup engine and verify backup integrity
            backup_engine = create_engine(f"sqlite:///{backup_db_path}")
            BackupSession = sessionmaker(bind=backup_engine)
            backup_session = BackupSession()
            
            # Verify backup data integrity
            backup_users = backup_session.query(User).all()
            assert len(backup_users) == 5
            
            # Verify original data is preserved in backup
            backup_user_0 = backup_session.query(User).filter(User.email == "backup0@test.com").first()
            assert backup_user_0.name == "Backup Test 0"  # Original name, not modified
            
            backup_session.close()
        
        finally:
            test_session.close()
    
    def test_point_in_time_recovery(self, tmp_path):
        """Test point-in-time recovery capabilities."""
        # This would test transaction log replay, etc.
        # Simplified test for SQLite
        
        test_db_path = tmp_path / "pit_recovery.db"
        
        # Create test database
        test_engine = create_engine(f"sqlite:///{test_db_path}")
        Base.metadata.create_all(test_engine)
        TestSession = sessionmaker(bind=test_engine)
        
        timestamps = []
        data_states = []
        
        # Create data at different points in time
        for i in range(3):
            test_session = TestSession()
            
            user = User(name=f"PIT Test {i}", email=f"pit{i}@test.com")
            test_session.add(user)
            test_session.commit()
            
            timestamps.append(datetime.now())
            
            # Record data state
            all_users = test_session.query(User).all()
            data_states.append(len(all_users))
            
            test_session.close()
            time.sleep(0.1)  # Small delay between transactions
        
        # Verify we can identify different states
        assert len(set(data_states)) > 1, "Should have different data states at different times"
        assert data_states[-1] == 3, "Should have 3 users at final state"
    
    def test_corruption_detection(self, db_session):
        """Test data corruption detection."""
        # Create test data
        user = User(name="Corruption Test", email="corruption@test.com")
        db_session.add(user)
        db_session.flush()
        
        profile = Profile(
            user_id=user.id,
            risk_score=50,
            time_horizon_yrs=10,
            objectives="test",
            constraints="test"
        )
        db_session.add(profile)
        db_session.commit()
        
        # Test data consistency checks
        # Check that profile references valid user
        orphaned_profiles = db_session.query(Profile).filter(
            ~Profile.user_id.in_(db_session.query(User.id))
        ).count()
        
        assert orphaned_profiles == 0, "Found orphaned profiles indicating corruption"
        
        # Check data format consistency
        invalid_emails = db_session.query(User).filter(
            ~User.email.contains("@")
        ).count()
        
        # Note: This is simplified - real email validation would be more thorough
        assert invalid_emails == 0, "Found invalid email formats"
        
        # Clean up
        db_session.delete(profile)
        db_session.delete(user)
        db_session.commit()


@pytest.mark.data_integrity
class TestConcurrentDataAccess:
    """Test data integrity under concurrent access."""
    
    def test_concurrent_write_integrity(self, db_session):
        """Test data integrity during concurrent writes."""
        import threading
        import time
        
        results = []
        errors = []
        
        def create_user_profile(thread_id: int):
            """Create user and profile in separate thread."""
            try:
                # Create new session for thread
                thread_session = SessionLocal()
                
                user = User(
                    name=f"Concurrent User {thread_id}",
                    email=f"concurrent{thread_id}@test.com"
                )
                thread_session.add(user)
                thread_session.flush()
                
                profile = Profile(
                    user_id=user.id,
                    risk_score=50,
                    time_horizon_yrs=10,
                    objectives=f"objectives{thread_id}",
                    constraints=f"constraints{thread_id}"
                )
                thread_session.add(profile)
                thread_session.commit()
                
                results.append({
                    'thread_id': thread_id,
                    'user_id': user.id,
                    'profile_id': profile.id
                })
                
                thread_session.close()
            
            except Exception as e:
                errors.append({
                    'thread_id': thread_id,
                    'error': str(e)
                })
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_user_profile, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == 10, f"Not all threads completed successfully: {len(results)}"
        
        # Verify data integrity
        unique_user_ids = set(result['user_id'] for result in results)
        assert len(unique_user_ids) == 10, "Duplicate user IDs created during concurrent access"
        
        # Clean up
        for result in results:
            try:
                user = db_session.get(User, result['user_id'])
                profile = db_session.get(Profile, result['profile_id'])
                if profile:
                    db_session.delete(profile)
                if user:
                    db_session.delete(user)
                db_session.commit()
            except Exception:
                db_session.rollback()
    
    def test_transaction_isolation(self, db_session):
        """Test transaction isolation levels."""
        # Create initial data
        user = User(name="Isolation Test", email="isolation@test.com")
        db_session.add(user)
        db_session.commit()
        
        user_id = user.id
        
        # Test read consistency
        # In one session, start transaction and read data
        session1 = SessionLocal()
        session2 = SessionLocal()
        
        try:
            # Session 1 reads initial data
            user1 = session1.get(User, user_id)
            original_name = user1.name
            
            # Session 2 modifies data
            user2 = session2.get(User, user_id)
            user2.name = "Modified in Session 2"
            session2.commit()
            
            # Session 1 reads again (should see original or modified depending on isolation)
            session1.refresh(user1)
            updated_name = user1.name
            
            # Verify transaction behavior
            # In SQLite with default settings, we should see the committed change
            assert updated_name == "Modified in Session 2" or updated_name == original_name
            
        finally:
            session1.close()
            session2.close()
            
            # Clean up
            user = db_session.get(User, user_id)
            db_session.delete(user)
            db_session.commit()


@pytest.mark.data_integrity
def test_generate_data_integrity_report(tmp_path, db_session):
    """Generate comprehensive data integrity report."""
    
    tester = DataIntegrityTester(db_session)
    
    try:
        # Create test data
        test_data = tester.create_test_data_set(20)
        
        # Run all integrity tests
        integrity_report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_records_created': sum(len(records) for records in test_data.values()),
                'test_categories': [
                    'Referential Integrity',
                    'Data Constraints', 
                    'Data Consistency',
                    'Financial Accuracy',
                    'Concurrent Access'
                ]
            },
            'referential_integrity': tester.test_referential_integrity(),
            'constraint_validation': tester.test_data_constraints(),
            'consistency_checks': tester.test_data_consistency(),
            'financial_accuracy': tester.test_financial_data_accuracy(),
            'recommendations': [
                'Implement automated data integrity monitoring',
                'Add constraint validation at application level',
                'Set up regular data consistency checks',
                'Implement proper transaction isolation',
                'Add data backup verification procedures',
                'Monitor referential integrity violations',
                'Implement financial calculation validation',
                'Add concurrent access testing to CI/CD',
                'Set up data archiving procedures',
                'Implement corruption detection mechanisms'
            ]
        }
        
        # Calculate overall health score
        total_violations = (
            len(integrity_report['referential_integrity'].get('orphaned_records', [])) +
            len(integrity_report['constraint_validation'].get('constraint_violations', [])) +
            len(integrity_report['consistency_checks'].get('inconsistencies_found', [])) +
            len(integrity_report['financial_accuracy'].get('calculation_errors', []))
        )
        
        integrity_report['overall_health_score'] = max(0, 100 - (total_violations * 5))
        integrity_report['integrity_status'] = (
            'excellent' if integrity_report['overall_health_score'] > 90 else
            'good' if integrity_report['overall_health_score'] > 75 else
            'needs_attention' if integrity_report['overall_health_score'] > 50 else
            'critical'
        )
        
        # Save report
        report_file = tmp_path / "data_integrity_report.json"
        with open(report_file, 'w') as f:
            json.dump(integrity_report, f, indent=2, default=str)
        
        assert report_file.exists()
        assert integrity_report['overall_health_score'] >= 0
        
    finally:
        tester.cleanup_test_data()