#!/usr/bin/env python3
"""
SuperNova AI Configuration Management CLI

Command-line interface for managing and validating configurations.
Provides tools for:
- Configuration validation and testing
- Environment-specific configuration management
- Secrets management operations
- Configuration migration and export/import
- Health checks and monitoring
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

# Import configuration modules
try:
    from .config_management import (
        ConfigurationManager, Environment, ConfigurationLevel, get_config_sync
    )
    from .config_schemas import get_config_schema, validate_environment_config
    from .config_validation import (
        ConfigurationValidationFramework, validate_config, quick_validate
    )
    from .secrets_management import (
        get_secrets_manager, SecretType, create_secret, get_secret
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from supernova.config_management import (
        ConfigurationManager, Environment, ConfigurationLevel, get_config_sync
    )
    from supernova.config_schemas import get_config_schema, validate_environment_config
    from supernova.config_validation import (
        ConfigurationValidationFramework, validate_config, quick_validate
    )
    from supernova.secrets_management import (
        get_secrets_manager, SecretType, create_secret, get_secret
    )

console = Console()


class ConfigCLI:
    """Main CLI class for configuration management."""
    
    def __init__(self):
        self.console = Console()
        self.validation_framework = ConfigurationValidationFramework()
        self.config_manager = None
        self.secrets_manager = None
    
    def init_managers(self, environment: Environment):
        """Initialize configuration and secrets managers."""
        try:
            self.config_manager = ConfigurationManager(environment=environment)
            self.secrets_manager = get_secrets_manager()
        except Exception as e:
            self.console.print(f"[red]Error initializing managers: {e}[/red]")
            sys.exit(1)
    
    async def validate_configuration(
        self,
        environment: Environment,
        config_file: Optional[str] = None,
        strict: bool = False,
        output_format: str = "text",
        save_report: Optional[str] = None
    ):
        """Validate configuration for specified environment."""
        
        self.console.print(f"[blue]Validating configuration for {environment.value} environment...[/blue]")
        
        try:
            # Load configuration
            if config_file:
                config = self._load_config_file(config_file)
            else:
                config = self._load_current_config(environment)
            
            # Run validation
            with Progress() as progress:
                task = progress.add_task("Running validation checks...", total=100)
                
                summary = await self.validation_framework.validate_configuration(
                    config, environment
                )
                
                progress.update(task, completed=100)
            
            # Display results
            self._display_validation_results(summary, output_format, strict)
            
            # Save report if requested
            if save_report:
                report = self.validation_framework.get_validation_report(
                    summary, format="json" if save_report.endswith('.json') else "text"
                )
                
                with open(save_report, 'w') as f:
                    f.write(report)
                
                self.console.print(f"[green]Report saved to {save_report}[/green]")
            
            # Return exit code based on validation results
            if strict and summary.errors > 0:
                sys.exit(1)
            elif summary.critical > 0:
                sys.exit(2)
            else:
                sys.exit(0)
                
        except Exception as e:
            self.console.print(f"[red]Validation failed: {e}[/red]")
            sys.exit(1)
    
    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    return json.load(f)
                else:
                    # Try to parse as .env file
                    config = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config[key.lower()] = value
                    return config
        except Exception as e:
            self.console.print(f"[red]Error loading config file {config_file}: {e}[/red]")
            sys.exit(1)
    
    def _load_current_config(self, environment: Environment) -> Dict[str, Any]:
        """Load current configuration from environment."""
        try:
            # Load from environment variables and config files
            config = {}
            
            # Load environment variables
            for key, value in os.environ.items():
                if key.startswith('SUPERNOVA_') or key in [
                    'DEBUG', 'LOG_LEVEL', 'DATABASE_URL', 'REDIS_URL',
                    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'
                ]:
                    config[key.lower()] = value
            
            # Add environment
            config['supernova_env'] = environment.value
            
            return config
        except Exception as e:
            self.console.print(f"[red]Error loading current config: {e}[/red]")
            sys.exit(1)
    
    def _display_validation_results(
        self,
        summary,
        output_format: str,
        strict: bool
    ):
        """Display validation results."""
        
        if output_format == "json":
            report = self.validation_framework.get_validation_report(summary, format="json")
            self.console.print(report)
            return
        
        # Rich text display
        self.console.print("\n")
        
        # Summary panel
        summary_table = Table(show_header=False)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value")
        
        summary_table.add_row("Environment", summary.environment.value)
        summary_table.add_row("Total Checks", str(summary.total_checks))
        summary_table.add_row("Passed", f"[green]{summary.passed}[/green]")
        summary_table.add_row("Failed", f"[red]{summary.failed}[/red]")
        summary_table.add_row("Warnings", f"[yellow]{summary.warnings}[/yellow]")
        summary_table.add_row("Errors", f"[red]{summary.errors}[/red]")
        summary_table.add_row("Critical", f"[bold red]{summary.critical}[/bold red]")
        summary_table.add_row("Execution Time", f"{summary.execution_time:.2f}s")
        
        self.console.print(Panel(summary_table, title="Validation Summary"))
        
        # Issues details
        if summary.results:
            # Group by severity
            critical_issues = [r for r in summary.results if r.severity.value == "critical" and not r.passed]
            error_issues = [r for r in summary.results if r.severity.value == "error" and not r.passed]
            warning_issues = [r for r in summary.results if r.severity.value == "warning" and not r.passed]
            
            # Display critical issues
            if critical_issues:
                self.console.print("\n[bold red]CRITICAL ISSUES[/bold red]")
                for issue in critical_issues:
                    self._display_issue(issue, "red")
            
            # Display errors
            if error_issues:
                self.console.print("\n[bold red]ERRORS[/bold red]")
                for issue in error_issues:
                    self._display_issue(issue, "red")
            
            # Display warnings
            if warning_issues:
                self.console.print("\n[bold yellow]WARNINGS[/bold yellow]")
                for issue in warning_issues:
                    self._display_issue(issue, "yellow")
        
        # Overall status
        if summary.critical > 0:
            status = "[bold red]CRITICAL - Immediate action required[/bold red]"
        elif summary.errors > 0:
            if strict:
                status = "[red]FAILED - Configuration has errors[/red]"
            else:
                status = "[yellow]WARNING - Configuration has errors[/yellow]"
        elif summary.warnings > 0:
            status = "[yellow]WARNING - Configuration has warnings[/yellow]"
        else:
            status = "[green]PASSED - Configuration is valid[/green]"
        
        self.console.print(f"\n[bold]Status: {status}[/bold]")
    
    def _display_issue(self, issue, color: str):
        """Display a single validation issue."""
        panel_content = f"[{color}]{issue.message}[/{color}]\n"
        
        if issue.affected_keys:
            panel_content += f"\nAffected: {', '.join(issue.affected_keys)}"
        
        if issue.suggestions:
            panel_content += "\n\nSuggestions:"
            for suggestion in issue.suggestions:
                panel_content += f"\n• {suggestion}"
        
        self.console.print(Panel(
            panel_content,
            title=f"{issue.check_id} ({issue.validation_type.value})",
            border_style=color
        ))
    
    async def check_health(self, environment: Environment):
        """Check configuration system health."""
        
        self.init_managers(environment)
        
        self.console.print(f"[blue]Checking configuration health for {environment.value}...[/blue]")
        
        try:
            # Configuration health
            config_health = await self.config_manager.get_configuration_health()
            
            # Secrets health
            secrets_health = await self.secrets_manager.get_secret_health()
            
            # Display health status
            self._display_health_status(config_health, secrets_health)
            
        except Exception as e:
            self.console.print(f"[red]Health check failed: {e}[/red]")
            sys.exit(1)
    
    def _display_health_status(self, config_health: Dict, secrets_health: Dict):
        """Display health status information."""
        
        # Configuration health
        config_table = Table(title="Configuration Health")
        config_table.add_column("Metric", style="bold")
        config_table.add_column("Status")
        
        config_table.add_row("Environment", config_health.get("environment", "Unknown"))
        config_table.add_row("Total Configurations", str(config_health.get("total_configurations", 0)))
        config_table.add_row("Cache Size", str(config_health.get("cache_size", 0)))
        config_table.add_row("Hot Reload", "✓" if config_health.get("hot_reload_enabled") else "✗")
        config_table.add_row("Encryption", "✓" if config_health.get("encryption_enabled") else "✗")
        config_table.add_row("Cloud Secrets", "✓" if config_health.get("cloud_secrets_enabled") else "✗")
        
        self.console.print(config_table)
        
        # Configuration issues
        issues = config_health.get("issues", [])
        if issues:
            self.console.print("\n[bold red]Configuration Issues:[/bold red]")
            for issue in issues:
                self.console.print(f"• {issue}")
        
        # Secrets health
        if secrets_health and not secrets_health.get("error"):
            secrets_table = Table(title="Secrets Health")
            secrets_table.add_column("Metric", style="bold")
            secrets_table.add_column("Status")
            
            secrets_table.add_row("Total Secrets", str(secrets_health.get("total_secrets", 0)))
            secrets_table.add_row("Active Secrets", str(secrets_health.get("active_secrets", 0)))
            secrets_table.add_row("Expired Secrets", str(secrets_health.get("expired_secrets", 0)))
            secrets_table.add_row("Need Rotation", str(secrets_health.get("rotation_needed", 0)))
            secrets_table.add_row("Backup Stores", str(secrets_health.get("backup_stores", 0)))
            
            self.console.print(secrets_table)
            
            # Secrets issues
            secret_issues = secrets_health.get("issues", [])
            if secret_issues:
                self.console.print("\n[bold red]Secrets Issues:[/bold red]")
                for issue in secret_issues:
                    self.console.print(f"• {issue}")
    
    async def manage_secrets(self, action: str, **kwargs):
        """Manage secrets operations."""
        
        self.secrets_manager = get_secrets_manager()
        
        if action == "create":
            await self._create_secret(**kwargs)
        elif action == "get":
            await self._get_secret(**kwargs)
        elif action == "list":
            await self._list_secrets(**kwargs)
        elif action == "delete":
            await self._delete_secret(**kwargs)
        elif action == "rotate":
            await self._rotate_secret(**kwargs)
        else:
            self.console.print(f"[red]Unknown secrets action: {action}[/red]")
            sys.exit(1)
    
    async def _create_secret(self, name: str, secret_type: str, value: Optional[str] = None, **kwargs):
        """Create a new secret."""
        
        try:
            secret_type_enum = SecretType(secret_type)
        except ValueError:
            self.console.print(f"[red]Invalid secret type: {secret_type}[/red]")
            self.console.print(f"Valid types: {[t.value for t in SecretType]}")
            sys.exit(1)
        
        if not value:
            value = Prompt.ask(f"Enter value for {name}", password=True)
        
        try:
            secret_id = await create_secret(name, secret_type_enum, value=value, **kwargs)
            if secret_id:
                self.console.print(f"[green]Secret created with ID: {secret_id}[/green]")
            else:
                self.console.print("[red]Failed to create secret[/red]")
                sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error creating secret: {e}[/red]")
            sys.exit(1)
    
    async def _get_secret(self, secret_id: str, **kwargs):
        """Retrieve a secret value."""
        
        try:
            value = await get_secret(secret_id, **kwargs)
            if value:
                if Confirm.ask("Display secret value?", default=False):
                    self.console.print(f"Secret value: [bold]{value}[/bold]")
                else:
                    self.console.print(f"Secret found (length: {len(value)})")
            else:
                self.console.print("[red]Secret not found[/red]")
                sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error retrieving secret: {e}[/red]")
            sys.exit(1)
    
    async def _list_secrets(self, **kwargs):
        """List all secrets."""
        
        try:
            secrets = await self.secrets_manager.list_secrets(**kwargs)
            
            if not secrets:
                self.console.print("No secrets found")
                return
            
            # Display secrets table
            table = Table(title="Secrets")
            table.add_column("ID", style="bold")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Status")
            table.add_column("Created")
            table.add_column("Last Accessed")
            
            for secret in secrets:
                created_at = datetime.fromisoformat(secret['created_at']).strftime('%Y-%m-%d %H:%M')
                last_accessed = "Never"
                if secret.get('last_accessed'):
                    last_accessed = datetime.fromisoformat(secret['last_accessed']).strftime('%Y-%m-%d %H:%M')
                
                status_color = "green" if secret['status'] == "active" else "yellow"
                
                table.add_row(
                    secret['secret_id'][:8] + "...",
                    secret['name'],
                    secret['secret_type'],
                    f"[{status_color}]{secret['status']}[/{status_color}]",
                    created_at,
                    last_accessed
                )
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error listing secrets: {e}[/red]")
            sys.exit(1)
    
    async def _delete_secret(self, secret_id: str, **kwargs):
        """Delete a secret."""
        
        if not Confirm.ask(f"Delete secret {secret_id}?", default=False):
            return
        
        try:
            from supernova.secrets_management import delete_secret
            success = await delete_secret(secret_id, **kwargs)
            if success:
                self.console.print(f"[green]Secret {secret_id} deleted[/green]")
            else:
                self.console.print("[red]Failed to delete secret[/red]")
                sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error deleting secret: {e}[/red]")
            sys.exit(1)
    
    async def _rotate_secret(self, secret_id: str, **kwargs):
        """Rotate a secret."""
        
        try:
            success = await self.secrets_manager.rotate_secret(secret_id, **kwargs)
            if success:
                self.console.print(f"[green]Secret {secret_id} rotated successfully[/green]")
            else:
                self.console.print("[red]Failed to rotate secret[/red]")
                sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error rotating secret: {e}[/red]")
            sys.exit(1)
    
    def generate_templates(self, environment: Environment, output_dir: str):
        """Generate configuration templates."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.console.print(f"[blue]Generating configuration templates for {environment.value}...[/blue]")
        
        try:
            # Generate .env template
            env_template = self._generate_env_template(environment)
            env_file = output_path / f".env.{environment.value}"
            
            with open(env_file, 'w') as f:
                f.write(env_template)
            
            # Generate YAML config template
            yaml_template = self._generate_yaml_template(environment)
            yaml_file = output_path / f"config.{environment.value}.yaml"
            
            with open(yaml_file, 'w') as f:
                f.write(yaml_template)
            
            self.console.print(f"[green]Templates generated in {output_path}[/green]")
            self.console.print(f"• Environment file: {env_file}")
            self.console.print(f"• Configuration file: {yaml_file}")
            
        except Exception as e:
            self.console.print(f"[red]Error generating templates: {e}[/red]")
            sys.exit(1)
    
    def _generate_env_template(self, environment: Environment) -> str:
        """Generate .env template for environment."""
        
        template = f"""# SuperNova AI - {environment.value.title()} Environment Configuration
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Core Settings
SUPERNOVA_ENV={environment.value}
DEBUG={'true' if environment == Environment.DEVELOPMENT else 'false'}
LOG_LEVEL={'DEBUG' if environment == Environment.DEVELOPMENT else 'INFO'}

# Security
SECRET_KEY=your-secret-key-here-minimum-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-here-minimum-32-chars
ENCRYPTION_KEY=your-encryption-key-here-minimum-32-chars

# Database
DATABASE_URL={'sqlite:///./supernova.db' if environment == Environment.DEVELOPMENT else 'postgresql://user:pass@host:5432/dbname'}

# LLM Configuration
LLM_ENABLED=true
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Cache
CACHE_BACKEND={'memory' if environment == Environment.DEVELOPMENT else 'redis'}
REDIS_URL={'redis://localhost:6379/0' if environment != Environment.DEVELOPMENT else ''}

# Security Settings
SSL_REQUIRED={'false' if environment == Environment.DEVELOPMENT else 'true'}
RATE_LIMIT_ENABLED={'false' if environment == Environment.DEVELOPMENT else 'true'}

# Monitoring
METRICS_ENABLED={'false' if environment == Environment.DEVELOPMENT else 'true'}
ALERTING_ENABLED={'false' if environment == Environment.DEVELOPMENT else 'true'}
"""
        
        return template
    
    def _generate_yaml_template(self, environment: Environment) -> str:
        """Generate YAML config template for environment."""
        
        config = {
            'application': {
                'name': 'SuperNova AI',
                'environment': environment.value,
                'version': '1.0.0'
            },
            'database': {
                'pool_size': 20 if environment == Environment.PRODUCTION else 5,
                'timeout': 30
            },
            'llm': {
                'temperature': 0.2,
                'max_tokens': 4000,
                'timeout': 60
            },
            'security': {
                'rate_limiting': {
                    'enabled': environment != Environment.DEVELOPMENT,
                    'per_minute': 100,
                    'per_hour': 1000
                }
            },
            'monitoring': {
                'metrics': {
                    'enabled': environment != Environment.DEVELOPMENT
                },
                'logging': {
                    'level': 'DEBUG' if environment == Environment.DEVELOPMENT else 'INFO',
                    'format': 'text' if environment == Environment.DEVELOPMENT else 'json'
                }
            }
        }
        
        return yaml.dump(config, default_flow_style=False, indent=2)


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="SuperNova AI Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate production configuration
  python -m supernova.config_cli validate --environment production --strict

  # Check system health
  python -m supernova.config_cli health --environment production

  # Create a new secret
  python -m supernova.config_cli secrets create --name api_key --type api_key

  # Generate configuration templates
  python -m supernova.config_cli generate --environment staging --output ./config
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument(
        '--environment', '-e',
        type=str,
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment to validate'
    )
    validate_parser.add_argument(
        '--config-file', '-f',
        type=str,
        help='Configuration file to validate'
    )
    validate_parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict mode - fail on any errors'
    )
    validate_parser.add_argument(
        '--output-format',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    validate_parser.add_argument(
        '--save-report',
        type=str,
        help='Save validation report to file'
    )
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    health_parser.add_argument(
        '--environment', '-e',
        type=str,
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment to check'
    )
    
    # Secrets management
    secrets_parser = subparsers.add_parser('secrets', help='Manage secrets')
    secrets_subparsers = secrets_parser.add_subparsers(dest='secrets_action')
    
    # Create secret
    create_parser = secrets_subparsers.add_parser('create', help='Create a new secret')
    create_parser.add_argument('--name', required=True, help='Secret name')
    create_parser.add_argument('--type', required=True, choices=[t.value for t in SecretType], help='Secret type')
    create_parser.add_argument('--value', help='Secret value (will prompt if not provided)')
    
    # Get secret
    get_parser = secrets_subparsers.add_parser('get', help='Retrieve a secret')
    get_parser.add_argument('secret_id', help='Secret ID')
    
    # List secrets
    list_parser = secrets_subparsers.add_parser('list', help='List all secrets')
    
    # Delete secret
    delete_parser = secrets_subparsers.add_parser('delete', help='Delete a secret')
    delete_parser.add_argument('secret_id', help='Secret ID')
    
    # Rotate secret
    rotate_parser = secrets_subparsers.add_parser('rotate', help='Rotate a secret')
    rotate_parser.add_argument('secret_id', help='Secret ID')
    
    # Generate templates
    generate_parser = subparsers.add_parser('generate', help='Generate configuration templates')
    generate_parser.add_argument(
        '--environment', '-e',
        type=str,
        choices=['development', 'staging', 'production'],
        required=True,
        help='Environment to generate templates for'
    )
    generate_parser.add_argument(
        '--output', '-o',
        type=str,
        default='./config',
        help='Output directory for templates'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create CLI instance
    cli = ConfigCLI()
    
    try:
        if args.command == 'validate':
            asyncio.run(cli.validate_configuration(
                Environment(args.environment),
                args.config_file,
                args.strict,
                args.output_format,
                args.save_report
            ))
        
        elif args.command == 'health':
            asyncio.run(cli.check_health(Environment(args.environment)))
        
        elif args.command == 'secrets':
            if args.secrets_action == 'create':
                asyncio.run(cli.manage_secrets(
                    'create',
                    name=args.name,
                    secret_type=args.type,
                    value=args.value
                ))
            elif args.secrets_action == 'get':
                asyncio.run(cli.manage_secrets('get', secret_id=args.secret_id))
            elif args.secrets_action == 'list':
                asyncio.run(cli.manage_secrets('list'))
            elif args.secrets_action == 'delete':
                asyncio.run(cli.manage_secrets('delete', secret_id=args.secret_id))
            elif args.secrets_action == 'rotate':
                asyncio.run(cli.manage_secrets('rotate', secret_id=args.secret_id))
            else:
                secrets_parser.print_help()
        
        elif args.command == 'generate':
            cli.generate_templates(Environment(args.environment), args.output)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()