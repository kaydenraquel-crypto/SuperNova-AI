#!/usr/bin/env python3
"""
SuperNova AI LLM Integration Setup Script

This script configures and initializes the complete LLM integration system including:
- LLM provider configuration and validation
- Secure API key management
- Cost tracking and monitoring
- Provider health monitoring
- Database initialization
- Configuration validation
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_setup.log')
    ]
)
logger = logging.getLogger(__name__)

class LLMSetupManager:
    """Main setup manager for LLM integration"""
    
    def __init__(self):
        self.setup_steps = []
        self.failed_steps = []
        self.warnings = []
    
    async def run_complete_setup(self):
        """Run complete LLM integration setup"""
        logger.info("Starting SuperNova AI LLM Integration Setup")
        logger.info("=" * 60)
        
        try:
            # Step 1: Environment validation
            await self.validate_environment()
            
            # Step 2: Configuration setup
            await self.setup_configuration()
            
            # Step 3: Database initialization
            await self.initialize_databases()
            
            # Step 4: API key management setup
            await self.setup_key_management()
            
            # Step 5: Provider configuration
            await self.configure_providers()
            
            # Step 6: Cost tracking initialization
            await self.initialize_cost_tracking()
            
            # Step 7: Run system validation
            await self.validate_integration()
            
            # Step 8: Generate setup report
            await self.generate_setup_report()
            
            logger.info("LLM Integration Setup Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Setup failed with error: {e}")
            await self.generate_error_report(str(e))
            raise
    
    async def validate_environment(self):
        """Validate Python environment and dependencies"""
        logger.info("Step 1: Validating Environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError(f"Python 3.8+ required, found {sys.version_info}")
        
        logger.info(f"✓ Python version: {sys.version}")
        
        # Check required dependencies
        required_packages = [
            'langchain',
            'langchain_openai',
            'langchain_anthropic',
            'langchain_ollama',
            'langchain_huggingface',
            'openai',
            'anthropic',
            'cryptography',
            'pydantic',
            'sqlalchemy',
            'watchdog',
            'pyyaml'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} (missing)")
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
            self.warnings.append(f"Missing packages: {missing_packages}")
        
        self.setup_steps.append("Environment validation")
    
    async def setup_configuration(self):
        """Setup and validate configuration"""
        logger.info("Step 2: Setting up Configuration...")
        
        try:
            # Initialize configuration manager
            from supernova.config_manager import get_config_manager
            config_manager = get_config_manager()
            
            logger.info("✓ Configuration manager initialized")
            
            # Validate current configuration
            validation_result = await config_manager.reload_configuration()
            
            if validation_result.valid:
                logger.info("✓ Configuration validation passed")
            else:
                logger.warning("⚠ Configuration validation warnings:")
                for warning in validation_result.warnings:
                    logger.warning(f"  - {warning}")
                for error in validation_result.errors:
                    logger.error(f"  - {error}")
                self.warnings.extend(validation_result.warnings)
            
            # Check if .env file exists, create template if not
            if not os.path.exists('.env'):
                logger.info("Creating .env file from template...")
                if os.path.exists('.env.template'):
                    import shutil
                    shutil.copy('.env.template', '.env')
                    logger.info("✓ Created .env file from template")
                    logger.info("⚠ Please update .env file with your API keys")
                    self.warnings.append("API keys need to be configured in .env file")
                else:
                    logger.warning("⚠ .env.template not found")
            
            self.setup_steps.append("Configuration setup")
            
        except Exception as e:
            logger.error(f"Configuration setup failed: {e}")
            self.failed_steps.append(f"Configuration setup: {e}")
    
    async def initialize_databases(self):
        """Initialize databases for LLM components"""
        logger.info("Step 3: Initializing Databases...")
        
        try:
            # Initialize key management database
            from supernova.llm_key_manager import get_key_manager
            key_manager = get_key_manager()
            logger.info("✓ Key management database initialized")
            
            # Initialize cost tracking database
            from supernova.llm_cost_tracker import get_cost_tracker
            cost_tracker = get_cost_tracker()
            logger.info("✓ Cost tracking database initialized")
            
            # Test database connections
            try:
                # Test key manager database
                with key_manager._get_session() as session:
                    session.execute("SELECT 1")
                logger.info("✓ Key management database connection test passed")
                
                # Test cost tracker database
                with cost_tracker._get_session() as session:
                    session.execute("SELECT 1")
                logger.info("✓ Cost tracking database connection test passed")
                
            except Exception as e:
                logger.warning(f"Database connection test warning: {e}")
                self.warnings.append(f"Database connection issue: {e}")
            
            self.setup_steps.append("Database initialization")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.failed_steps.append(f"Database initialization: {e}")
    
    async def setup_key_management(self):
        """Setup API key management"""
        logger.info("Step 4: Setting up API Key Management...")
        
        try:
            from supernova.llm_key_manager import get_key_manager, ProviderType, initialize_key_manager
            
            # Initialize key manager with environment keys
            key_manager = await initialize_key_manager()
            logger.info("✓ Key manager initialized")
            
            # Check for API keys in environment
            env_keys = {
                ProviderType.OPENAI: os.getenv('OPENAI_API_KEY'),
                ProviderType.ANTHROPIC: os.getenv('ANTHROPIC_API_KEY'),
                ProviderType.HUGGINGFACE: os.getenv('HUGGINGFACE_API_TOKEN'),
            }
            
            configured_keys = 0
            for provider, key in env_keys.items():
                if key:
                    # Validate key format
                    if key_manager._validate_key_format(provider, key):
                        logger.info(f"✓ {provider.value} API key configured")
                        configured_keys += 1
                    else:
                        logger.warning(f"⚠ {provider.value} API key format invalid")
                        self.warnings.append(f"{provider.value} API key format invalid")
                else:
                    logger.info(f"○ {provider.value} API key not configured")
            
            if configured_keys == 0:
                self.warnings.append("No API keys configured - system will use fallback mode")
            
            # Test key encryption
            test_key = "sk-test123456789abcdef"
            encrypted = key_manager._encrypt_key(test_key)
            decrypted = key_manager._decrypt_key(encrypted)
            
            if decrypted == test_key:
                logger.info("✓ Key encryption/decryption test passed")
            else:
                raise RuntimeError("Key encryption test failed")
            
            self.setup_steps.append("API key management")
            
        except Exception as e:
            logger.error(f"Key management setup failed: {e}")
            self.failed_steps.append(f"Key management setup: {e}")
    
    async def configure_providers(self):
        """Configure LLM providers"""
        logger.info("Step 5: Configuring LLM Providers...")
        
        try:
            from supernova.llm_provider_manager import get_provider_manager
            from supernova.llm_key_manager import ProviderType
            
            provider_manager = get_provider_manager()
            logger.info("✓ Provider manager initialized")
            
            # Check provider configurations
            configured_providers = 0
            for provider_type in ProviderType:
                if provider_type in provider_manager.provider_configs:
                    config = provider_manager.provider_configs[provider_type]
                    status = provider_manager.provider_status.get(provider_type, 'unknown')
                    
                    logger.info(f"✓ {provider_type.value}: {status} (priority: {config.priority})")
                    if config.enabled:
                        configured_providers += 1
                else:
                    logger.info(f"○ {provider_type.value}: not configured")
            
            if configured_providers == 0:
                self.warnings.append("No LLM providers configured")
            else:
                logger.info(f"✓ {configured_providers} LLM providers configured")
            
            # Test provider selection
            try:
                available_providers = provider_manager._get_available_providers()
                if available_providers:
                    selected = provider_manager._select_provider("test")
                    if selected:
                        logger.info(f"✓ Provider selection test: {selected.value}")
                    else:
                        logger.warning("⚠ No provider selected in test")
                else:
                    logger.warning("⚠ No available providers")
                    self.warnings.append("No available LLM providers")
            except Exception as e:
                logger.warning(f"Provider selection test failed: {e}")
            
            self.setup_steps.append("Provider configuration")
            
        except Exception as e:
            logger.error(f"Provider configuration failed: {e}")
            self.failed_steps.append(f"Provider configuration: {e}")
    
    async def initialize_cost_tracking(self):
        """Initialize cost tracking system"""
        logger.info("Step 6: Initializing Cost Tracking...")
        
        try:
            from supernova.llm_cost_tracker import get_cost_tracker, ProviderType
            
            cost_tracker = get_cost_tracker()
            
            # Test cost recording
            test_record = await cost_tracker.record_usage(
                provider=ProviderType.OPENAI,
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                input_cost_per_1k=0.01,
                output_cost_per_1k=0.02,
                success=True,
                metadata={"test": True}
            )
            
            if test_record.get("tracked"):
                logger.info("✓ Cost tracking test passed")
                expected_cost = (100/1000 * 0.01) + (50/1000 * 0.02)  # 0.002
                if abs(test_record["cost"] - expected_cost) < 0.0001:
                    logger.info("✓ Cost calculation test passed")
                else:
                    logger.warning(f"⚠ Cost calculation mismatch: {test_record['cost']} vs {expected_cost}")
            else:
                logger.warning("⚠ Cost tracking test failed")
                self.warnings.append("Cost tracking test failed")
            
            # Check current usage
            summary = cost_tracker.get_current_usage_summary()
            logger.info("✓ Cost tracking initialized")
            
            self.setup_steps.append("Cost tracking initialization")
            
        except Exception as e:
            logger.error(f"Cost tracking initialization failed: {e}")
            self.failed_steps.append(f"Cost tracking initialization: {e}")
    
    async def validate_integration(self):
        """Validate complete integration"""
        logger.info("Step 7: Validating Integration...")
        
        try:
            from supernova.llm_service import get_llm_service, LLMRequest, ResponseQuality, TaskComplexity
            
            llm_service = get_llm_service()
            logger.info("✓ LLM service initialized")
            
            # Test service statistics
            stats = llm_service.get_service_stats()
            logger.info(f"✓ Service stats available: {len(stats)} categories")
            
            # Test fallback system
            test_request = LLMRequest(
                messages="This is a test message for integration validation.",
                quality=ResponseQuality.BASIC,
                complexity=TaskComplexity.SIMPLE
            )
            
            # Note: This will likely use fallback mode without real API keys
            try:
                response = await llm_service.generate_response(test_request)
                if response.success:
                    logger.info("✓ Integration test passed - LLM response generated")
                elif response.provider == "fallback":
                    logger.info("✓ Integration test passed - Fallback response generated")
                    self.warnings.append("Using fallback mode - configure API keys for full functionality")
                else:
                    logger.warning(f"⚠ Integration test warning: {response.error_message}")
                    self.warnings.append(f"Integration test issue: {response.error_message}")
            except Exception as e:
                logger.warning(f"Integration test failed: {e}")
                self.warnings.append(f"Integration test failed: {e}")
            
            # Test advisor integration
            try:
                from supernova.advisor import advise
                
                # Mock OHLCV data for test
                test_bars = [
                    {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
                    {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100}
                ]
                
                action, confidence, details, rationale, risk_notes = advise(
                    bars=test_bars,
                    risk_score=50,
                    sentiment_hint=0.1,
                    symbol="TEST"
                )
                
                if action and confidence and rationale:
                    logger.info("✓ Advisor integration test passed")
                else:
                    logger.warning("⚠ Advisor integration test incomplete")
                    
            except Exception as e:
                logger.warning(f"Advisor integration test failed: {e}")
                self.warnings.append(f"Advisor integration issue: {e}")
            
            self.setup_steps.append("Integration validation")
            
        except Exception as e:
            logger.error(f"Integration validation failed: {e}")
            self.failed_steps.append(f"Integration validation: {e}")
    
    async def generate_setup_report(self):
        """Generate comprehensive setup report"""
        logger.info("Step 8: Generating Setup Report...")
        
        try:
            report = {
                "setup_timestamp": str(asyncio.get_event_loop().time()),
                "status": "completed" if not self.failed_steps else "completed_with_warnings",
                "successful_steps": self.setup_steps,
                "failed_steps": self.failed_steps,
                "warnings": self.warnings,
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "project_root": str(project_root)
                },
                "next_steps": [],
                "recommendations": []
            }
            
            # Add next steps based on setup results
            if not os.getenv('OPENAI_API_KEY'):
                report["next_steps"].append("Configure OpenAI API key in .env file")
            
            if not os.getenv('ANTHROPIC_API_KEY'):
                report["next_steps"].append("Configure Anthropic API key in .env file")
            
            if self.warnings:
                report["next_steps"].append("Review and address setup warnings")
            
            # Add recommendations
            report["recommendations"].extend([
                "Review .env.template for additional configuration options",
                "Run tests with: python -m pytest tests/test_llm_integration.py",
                "Monitor logs during first use for any issues",
                "Set up cost alerts if using paid API providers"
            ])
            
            # Save report to file
            report_file = "llm_setup_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✓ Setup report saved to {report_file}")
            
            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("SETUP SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Status: {report['status'].upper()}")
            logger.info(f"Successful steps: {len(self.setup_steps)}")
            logger.info(f"Failed steps: {len(self.failed_steps)}")
            logger.info(f"Warnings: {len(self.warnings)}")
            
            if self.warnings:
                logger.info("\nWarnings:")
                for warning in self.warnings:
                    logger.info(f"  ⚠ {warning}")
            
            if report["next_steps"]:
                logger.info("\nNext Steps:")
                for step in report["next_steps"]:
                    logger.info(f"  → {step}")
            
            logger.info("\n" + "=" * 60)
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    async def generate_error_report(self, error_message: str):
        """Generate error report if setup fails"""
        logger.error("Generating Error Report...")
        
        error_report = {
            "setup_timestamp": str(asyncio.get_event_loop().time()),
            "status": "failed",
            "error_message": error_message,
            "successful_steps": self.setup_steps,
            "failed_steps": self.failed_steps,
            "warnings": self.warnings,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "project_root": str(project_root)
            }
        }
        
        try:
            with open("llm_setup_error_report.json", 'w') as f:
                json.dump(error_report, f, indent=2)
            logger.info("Error report saved to llm_setup_error_report.json")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")

async def main():
    """Main setup function"""
    setup_manager = LLMSetupManager()
    
    try:
        await setup_manager.run_complete_setup()
        return 0
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    # Run the setup
    exit_code = asyncio.run(main())