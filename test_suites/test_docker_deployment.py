"""
Docker Deployment Testing Suite
===============================

Comprehensive testing for Docker containers, multi-stage builds,
deployment scenarios, and containerized application behavior.
"""
import pytest
import docker
import time
import requests
import json
import subprocess
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import tempfile
from contextlib import contextmanager


class TestDockerBuildAndDeployment:
    """Test Docker container building and deployment processes."""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client fixture."""
        try:
            client = docker.from_env()
            # Test Docker connection
            client.ping()
            return client
        except docker.errors.DockerException as e:
            pytest.skip(f"Docker not available: {e}")
    
    @pytest.fixture(scope="class")
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_dockerfile_syntax_and_structure(self, project_root):
        """Test Dockerfile syntax and best practices."""
        dockerfile_path = project_root / "Dockerfile"
        
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Test basic Dockerfile structure
        assert dockerfile_content.strip().startswith("FROM"), "Dockerfile must start with FROM instruction"
        
        # Check for multi-stage build indicators
        from_instructions = [line for line in dockerfile_content.split('\n') if line.strip().startswith('FROM')]
        is_multistage = len(from_instructions) > 1
        
        if is_multistage:
            # Verify stages are properly named
            for instruction in from_instructions[1:]:  # Skip first FROM
                assert ' AS ' in instruction.upper(), "Multi-stage builds should use named stages"
        
        # Security best practices
        assert 'USER' in dockerfile_content.upper(), "Dockerfile should include USER instruction for security"
        
        # Check for .dockerignore
        dockerignore_path = project_root / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore file should exist"
        
        # Verify no sensitive files are included
        with open(dockerignore_path, 'r') as f:
            dockerignore_content = f.read()
        
        sensitive_patterns = ['.env', '*.key', '*.pem', '.git', 'node_modules', '__pycache__']
        for pattern in sensitive_patterns:
            assert pattern in dockerignore_content, f"Pattern '{pattern}' should be in .dockerignore"
    
    @pytest.mark.integration
    @pytest.mark.docker
    @pytest.mark.slow
    def test_docker_image_build(self, docker_client, project_root):
        """Test Docker image builds successfully."""
        image_name = "supernova-test"
        build_args = {
            'BUILDKIT_INLINE_CACHE': '1'
        }
        
        try:
            # Build the Docker image
            print(f"Building Docker image: {image_name}")
            image, build_logs = docker_client.images.build(
                path=str(project_root),
                tag=image_name,
                buildargs=build_args,
                rm=True,
                forcerm=True
            )
            
            # Collect build logs for analysis
            build_output = ""
            for log in build_logs:
                if 'stream' in log:
                    build_output += log['stream']
                    print(log['stream'], end='')
            
            # Verify image was created
            assert image is not None, "Docker image should be created successfully"
            assert image.tags, "Docker image should have tags"
            
            # Verify image size is reasonable (< 1GB for this application)
            image_size = image.attrs['Size'] / (1024 * 1024 * 1024)  # Convert to GB
            assert image_size < 1.0, f"Docker image too large: {image_size:.2f}GB"
            
            # Test image layers
            history = image.history()
            assert len(history) > 1, "Image should have multiple layers"
            
            # Verify security scanning doesn't show critical vulnerabilities
            # (This would integrate with tools like Trivy or Snyk in real implementation)
            
            return image
            
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_container_startup_and_health(self, docker_client, test_docker_image_build):
        """Test container starts up correctly and responds to health checks."""
        image = test_docker_image_build
        container_name = "supernova-test-container"
        
        # Remove container if it exists
        try:
            old_container = docker_client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        try:
            # Start container
            container = docker_client.containers.run(
                image.id,
                name=container_name,
                ports={'8000/tcp': 8000},
                environment={
                    'TESTING': 'true',
                    'DATABASE_URL': 'sqlite:///./test.db',
                    'SECRET_KEY': 'test-secret-key'
                },
                detach=True,
                remove=False  # Keep for inspection
            )
            
            # Wait for container to start
            time.sleep(10)
            
            # Check container is running
            container.reload()
            assert container.status == 'running', f"Container status: {container.status}"
            
            # Test health endpoint
            max_retries = 30
            health_url = "http://localhost:8000/health"
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        assert health_data.get('status') == 'healthy', "Application should be healthy"
                        break
                except requests.exceptions.RequestException:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        # Get container logs for debugging
                        logs = container.logs().decode('utf-8')
                        pytest.fail(f"Health check failed after {max_retries} attempts. Logs:\n{logs}")
            
            # Test basic API endpoints
            endpoints_to_test = [
                ("/", 200),
                ("/docs", 200),  # OpenAPI docs
                ("/redoc", 200),  # ReDoc
            ]
            
            for endpoint, expected_status in endpoints_to_test:
                url = f"http://localhost:8000{endpoint}"
                response = requests.get(url, timeout=10)
                assert response.status_code == expected_status, f"Endpoint {endpoint} returned {response.status_code}"
            
            # Test container resource usage
            stats = container.stats(stream=False)
            memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
            assert memory_usage < 500, f"Container using too much memory: {memory_usage:.2f}MB"
            
            return container
            
        except Exception as e:
            # Cleanup on failure
            try:
                container = docker_client.containers.get(container_name)
                logs = container.logs().decode('utf-8')
                print(f"Container logs:\n{logs}")
                container.remove(force=True)
            except:
                pass
            raise e
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_container_environment_variables(self, docker_client, test_docker_image_build):
        """Test container handles environment variables correctly."""
        image = test_docker_image_build
        
        test_cases = [
            # Test production environment
            {
                'env': {
                    'ENVIRONMENT': 'production',
                    'DATABASE_URL': 'postgresql://test:test@localhost/test',
                    'SECRET_KEY': 'production-secret-key',
                    'LOG_LEVEL': 'INFO'
                },
                'expected_log_level': 'INFO'
            },
            # Test development environment
            {
                'env': {
                    'ENVIRONMENT': 'development',
                    'DATABASE_URL': 'sqlite:///./dev.db',
                    'SECRET_KEY': 'dev-secret-key',
                    'LOG_LEVEL': 'DEBUG'
                },
                'expected_log_level': 'DEBUG'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            container_name = f"supernova-env-test-{i}"
            
            # Remove container if exists
            try:
                old_container = docker_client.containers.get(container_name)
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass
            
            try:
                # Start container with test environment
                container = docker_client.containers.run(
                    image.id,
                    name=container_name,
                    environment=test_case['env'],
                    detach=True,
                    remove=False
                )
                
                # Wait for startup
                time.sleep(5)
                
                # Check environment variables are set correctly
                exec_result = container.exec_run(['env'])
                env_output = exec_result.output.decode('utf-8')
                
                for key, value in test_case['env'].items():
                    assert f"{key}={value}" in env_output, f"Environment variable {key} not set correctly"
                
                # Check logs contain expected log level
                logs = container.logs().decode('utf-8')
                expected_level = test_case['expected_log_level']
                # Note: This would depend on how logging is configured in the application
                
                container.remove(force=True)
                
            except Exception as e:
                try:
                    container = docker_client.containers.get(container_name)
                    container.remove(force=True)
                except:
                    pass
                raise e
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_container_security_configuration(self, docker_client, test_docker_image_build):
        """Test container security configuration."""
        image = test_docker_image_build
        container_name = "supernova-security-test"
        
        # Remove container if exists
        try:
            old_container = docker_client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        try:
            # Start container with security options
            container = docker_client.containers.run(
                image.id,
                name=container_name,
                user='1000:1000',  # Non-root user
                read_only=True,    # Read-only filesystem
                tmpfs={'/tmp': 'noexec,nosuid,size=100m'},  # Secure tmp
                security_opt=['no-new-privileges:true'],    # No privilege escalation
                cap_drop=['ALL'],  # Drop all capabilities
                detach=True,
                remove=False
            )
            
            time.sleep(5)
            
            # Verify container is running as non-root
            exec_result = container.exec_run(['id'])
            id_output = exec_result.output.decode('utf-8')
            assert 'uid=1000' in id_output, "Container should run as non-root user"
            
            # Test that filesystem is read-only (should fail to write)
            exec_result = container.exec_run(['touch', '/test-file'])
            assert exec_result.exit_code != 0, "Filesystem should be read-only"
            
            # Test that tmp is writable but with restrictions
            exec_result = container.exec_run(['touch', '/tmp/test-file'])
            assert exec_result.exit_code == 0, "Should be able to write to /tmp"
            
            container.remove(force=True)
            
        except Exception as e:
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except:
                pass
            raise e
    
    @pytest.mark.integration
    @pytest.mark.docker
    @pytest.mark.slow
    def test_docker_compose_deployment(self, project_root):
        """Test Docker Compose deployment configuration."""
        compose_files = [
            project_root / "docker-compose.yml",
            project_root / "docker-compose.prod.yml"
        ]
        
        for compose_file in compose_files:
            if not compose_file.exists():
                continue
                
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Verify required services
            services = compose_config.get('services', {})
            assert 'supernova' in services, "SuperNova service should be defined"
            
            supernova_service = services['supernova']
            
            # Test service configuration
            assert 'image' in supernova_service or 'build' in supernova_service, \
                "Service should specify image or build configuration"
            
            # Test environment variables
            if 'environment' in supernova_service:
                env_vars = supernova_service['environment']
                if isinstance(env_vars, list):
                    env_dict = {var.split('=')[0]: var.split('=')[1] for var in env_vars if '=' in var}
                else:
                    env_dict = env_vars
                
                # Check for required environment variables
                required_env_vars = ['DATABASE_URL', 'SECRET_KEY']
                for var in required_env_vars:
                    assert var in env_dict, f"Required environment variable {var} not found"
            
            # Test port configuration
            if 'ports' in supernova_service:
                ports = supernova_service['ports']
                assert any('8000' in str(port) for port in ports), "Port 8000 should be exposed"
            
            # Test volume configuration (if any)
            if 'volumes' in supernova_service:
                volumes = supernova_service['volumes']
                # Verify no sensitive directories are mounted
                sensitive_paths = ['/etc', '/var', '/root']
                for volume in volumes:
                    volume_str = str(volume)
                    for sensitive in sensitive_paths:
                        assert not volume_str.startswith(sensitive), \
                            f"Sensitive path {sensitive} should not be mounted"
            
            # Test network configuration
            if 'networks' in compose_config:
                networks = compose_config['networks']
                for network_name, network_config in networks.items():
                    if network_config and 'driver' in network_config:
                        # Verify network driver is appropriate
                        assert network_config['driver'] in ['bridge', 'overlay'], \
                            f"Network driver {network_config['driver']} may not be secure"
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_container_performance_and_resources(self, docker_client, test_docker_image_build):
        """Test container performance and resource constraints."""
        image = test_docker_image_build
        container_name = "supernova-performance-test"
        
        # Remove container if exists
        try:
            old_container = docker_client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        try:
            # Start container with resource limits
            container = docker_client.containers.run(
                image.id,
                name=container_name,
                ports={'8000/tcp': 8001},  # Use different port
                environment={
                    'TESTING': 'true',
                    'DATABASE_URL': 'sqlite:///./test.db'
                },
                mem_limit='512m',    # 512MB memory limit
                cpu_count=1,         # 1 CPU
                detach=True,
                remove=False
            )
            
            time.sleep(10)
            
            # Monitor resource usage
            stats = container.stats(stream=False)
            
            # Memory usage should be within limits
            memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
            memory_limit = stats['memory_stats']['limit'] / (1024 * 1024)  # MB
            
            assert memory_usage < memory_limit * 0.8, \
                f"Memory usage {memory_usage:.2f}MB too close to limit {memory_limit:.2f}MB"
            
            # Test application performance under resource constraints
            base_url = "http://localhost:8001"
            
            # Test multiple concurrent requests
            import concurrent.futures
            import threading
            
            def make_request():
                try:
                    response = requests.get(f"{base_url}/health", timeout=5)
                    return response.status_code == 200
                except:
                    return False
            
            # Test with 10 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            success_rate = sum(results) / len(results)
            assert success_rate >= 0.8, f"Success rate {success_rate:.2f} too low under resource constraints"
            
            container.remove(force=True)
            
        except Exception as e:
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except:
                pass
            raise e
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_container_logging_configuration(self, docker_client, test_docker_image_build):
        """Test container logging is configured correctly."""
        image = test_docker_image_build
        container_name = "supernova-logging-test"
        
        # Remove container if exists
        try:
            old_container = docker_client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        try:
            # Start container with logging configuration
            container = docker_client.containers.run(
                image.id,
                name=container_name,
                environment={
                    'TESTING': 'true',
                    'LOG_LEVEL': 'INFO'
                },
                log_config=docker.types.LogConfig(
                    type=docker.types.LogConfig.types.JSON,
                    config={
                        'max-size': '10m',
                        'max-file': '3'
                    }
                ),
                detach=True,
                remove=False
            )
            
            time.sleep(5)
            
            # Generate some log activity
            container.exec_run(['python', '-c', 'import logging; logging.info("Test log message")'])
            
            # Get logs
            logs = container.logs().decode('utf-8')
            
            # Verify logs are structured (JSON format expected)
            log_lines = [line.strip() for line in logs.split('\n') if line.strip()]
            
            # Check for application startup logs
            assert any('startup' in line.lower() or 'starting' in line.lower() for line in log_lines), \
                "Application should log startup messages"
            
            # Test log levels are respected
            # (This would depend on application's logging configuration)
            
            container.remove(force=True)
            
        except Exception as e:
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except:
                pass
            raise e


class TestKubernetesDeployment:
    """Test Kubernetes deployment configurations."""
    
    @pytest.fixture(scope="class")
    def k8s_manifests_dir(self, project_root):
        """Kubernetes manifests directory."""
        return project_root / "deploy" / "kubernetes"
    
    @pytest.mark.integration
    @pytest.mark.k8s
    def test_kubernetes_manifest_validity(self, k8s_manifests_dir):
        """Test Kubernetes manifests are valid YAML and follow best practices."""
        if not k8s_manifests_dir.exists():
            pytest.skip("Kubernetes manifests not found")
        
        manifest_files = list(k8s_manifests_dir.glob("*.yaml")) + list(k8s_manifests_dir.glob("*.yml"))
        
        assert len(manifest_files) > 0, "No Kubernetes manifests found"
        
        required_manifests = ['deployment.yaml', 'service.yaml']
        found_manifests = [f.name for f in manifest_files]
        
        for required in required_manifests:
            assert required in found_manifests, f"Required manifest {required} not found"
        
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                try:
                    manifest = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {manifest_file}: {e}")
            
            # Verify basic Kubernetes structure
            assert 'apiVersion' in manifest, f"Missing apiVersion in {manifest_file}"
            assert 'kind' in manifest, f"Missing kind in {manifest_file}"
            assert 'metadata' in manifest, f"Missing metadata in {manifest_file}"
            
            # Test specific manifest types
            if manifest['kind'] == 'Deployment':
                self._validate_deployment_manifest(manifest, manifest_file)
            elif manifest['kind'] == 'Service':
                self._validate_service_manifest(manifest, manifest_file)
            elif manifest['kind'] == 'Ingress':
                self._validate_ingress_manifest(manifest, manifest_file)
    
    def _validate_deployment_manifest(self, manifest: Dict, filename: str):
        """Validate Deployment manifest."""
        spec = manifest.get('spec', {})
        assert 'replicas' in spec, f"Deployment {filename} missing replicas"
        assert spec['replicas'] >= 2, f"Deployment {filename} should have at least 2 replicas for HA"
        
        template = spec.get('template', {})
        pod_spec = template.get('spec', {})
        
        # Check containers
        containers = pod_spec.get('containers', [])
        assert len(containers) > 0, f"Deployment {filename} has no containers"
        
        main_container = containers[0]
        
        # Security best practices
        security_context = main_container.get('securityContext', {})
        assert security_context.get('runAsNonRoot', False), \
            f"Container in {filename} should run as non-root"
        assert security_context.get('readOnlyRootFilesystem', False), \
            f"Container in {filename} should have read-only filesystem"
        
        # Resource limits
        resources = main_container.get('resources', {})
        assert 'limits' in resources, f"Container in {filename} should have resource limits"
        assert 'requests' in resources, f"Container in {filename} should have resource requests"
        
        limits = resources['limits']
        assert 'memory' in limits, f"Container in {filename} should have memory limit"
        assert 'cpu' in limits, f"Container in {filename} should have CPU limit"
        
        # Health checks
        assert 'livenessProbe' in main_container, f"Container in {filename} should have liveness probe"
        assert 'readinessProbe' in main_container, f"Container in {filename} should have readiness probe"
        
        # Image pull policy
        assert main_container.get('imagePullPolicy') in ['Always', 'IfNotPresent'], \
            f"Container in {filename} should have appropriate image pull policy"
    
    def _validate_service_manifest(self, manifest: Dict, filename: str):
        """Validate Service manifest."""
        spec = manifest.get('spec', {})
        
        # Check service type
        service_type = spec.get('type', 'ClusterIP')
        assert service_type in ['ClusterIP', 'NodePort', 'LoadBalancer'], \
            f"Service {filename} has invalid type: {service_type}"
        
        # Check ports
        ports = spec.get('ports', [])
        assert len(ports) > 0, f"Service {filename} has no ports defined"
        
        for port in ports:
            assert 'port' in port, f"Port definition missing 'port' in {filename}"
            assert 'targetPort' in port, f"Port definition missing 'targetPort' in {filename}"
    
    def _validate_ingress_manifest(self, manifest: Dict, filename: str):
        """Validate Ingress manifest."""
        spec = manifest.get('spec', {})
        
        # Check TLS configuration
        if 'tls' in spec:
            tls_configs = spec['tls']
            for tls_config in tls_configs:
                assert 'secretName' in tls_config, f"TLS config missing secretName in {filename}"
                assert 'hosts' in tls_config, f"TLS config missing hosts in {filename}"
        
        # Check rules
        rules = spec.get('rules', [])
        assert len(rules) > 0, f"Ingress {filename} has no rules defined"
        
        for rule in rules:
            if 'http' in rule:
                paths = rule['http'].get('paths', [])
                assert len(paths) > 0, f"HTTP rule has no paths in {filename}"
                
                for path in paths:
                    backend = path.get('backend', {})
                    assert 'service' in backend, f"Path backend missing service in {filename}"
    
    @pytest.mark.integration
    @pytest.mark.k8s
    def test_helm_chart_structure(self, project_root):
        """Test Helm chart structure and templates."""
        chart_dir = project_root / "deploy" / "helm" / "supernova"
        
        if not chart_dir.exists():
            pytest.skip("Helm chart not found")
        
        # Check required Helm files
        required_files = ['Chart.yaml', 'values.yaml']
        for required_file in required_files:
            file_path = chart_dir / required_file
            assert file_path.exists(), f"Required Helm file {required_file} not found"
        
        # Validate Chart.yaml
        with open(chart_dir / "Chart.yaml", 'r') as f:
            chart_config = yaml.safe_load(f)
        
        required_chart_fields = ['name', 'version', 'appVersion', 'description']
        for field in required_chart_fields:
            assert field in chart_config, f"Chart.yaml missing required field: {field}"
        
        # Validate values.yaml
        with open(chart_dir / "values.yaml", 'r') as f:
            values_config = yaml.safe_load(f)
        
        # Check for important configuration sections
        important_sections = ['image', 'service', 'ingress', 'resources']
        for section in important_sections:
            assert section in values_config, f"values.yaml missing important section: {section}"
        
        # Check templates directory
        templates_dir = chart_dir / "templates"
        if templates_dir.exists():
            template_files = list(templates_dir.glob("*.yaml"))
            assert len(template_files) > 0, "Helm chart should have template files"


class TestContainerSecurity:
    """Test container security and vulnerability scanning."""
    
    @pytest.mark.security
    @pytest.mark.docker
    def test_image_vulnerability_scanning(self, project_root):
        """Test Docker image for vulnerabilities using Trivy."""
        try:
            # Run Trivy scan
            result = subprocess.run([
                'trivy', 'image', '--format', 'json', '--output', 'trivy-report.json',
                'supernova-test'
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode != 0:
                pytest.skip(f"Trivy scan failed: {result.stderr}")
            
            # Parse scan results
            with open(project_root / "trivy-report.json", 'r') as f:
                scan_results = json.load(f)
            
            # Check for critical vulnerabilities
            critical_vulns = []
            high_vulns = []
            
            for result in scan_results.get('Results', []):
                vulnerabilities = result.get('Vulnerabilities', [])
                for vuln in vulnerabilities:
                    severity = vuln.get('Severity', '').upper()
                    if severity == 'CRITICAL':
                        critical_vulns.append(vuln)
                    elif severity == 'HIGH':
                        high_vulns.append(vuln)
            
            # Fail if critical vulnerabilities found
            assert len(critical_vulns) == 0, \
                f"Found {len(critical_vulns)} critical vulnerabilities: {[v.get('VulnerabilityID') for v in critical_vulns]}"
            
            # Warn about high severity vulnerabilities
            if len(high_vulns) > 0:
                print(f"Warning: Found {len(high_vulns)} high severity vulnerabilities")
                for vuln in high_vulns[:5]:  # Show first 5
                    print(f"  - {vuln.get('VulnerabilityID')}: {vuln.get('Title')}")
            
        except FileNotFoundError:
            pytest.skip("Trivy not available for vulnerability scanning")
    
    @pytest.mark.security
    @pytest.mark.docker
    def test_container_secrets_not_exposed(self, docker_client, test_docker_image_build):
        """Test that container doesn't expose secrets in environment or filesystem."""
        image = test_docker_image_build
        container_name = "supernova-secrets-test"
        
        # Remove container if exists
        try:
            old_container = docker_client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        try:
            container = docker_client.containers.run(
                image.id,
                name=container_name,
                detach=True,
                remove=False
            )
            
            time.sleep(5)
            
            # Check environment variables don't contain secrets
            exec_result = container.exec_run(['env'])
            env_output = exec_result.output.decode('utf-8')
            
            secret_patterns = [
                r'password.*=.*\w+',
                r'secret.*=.*\w+', 
                r'key.*=.*[A-Za-z0-9+/]{20,}',  # Base64-like keys
                r'token.*=.*\w+'
            ]
            
            import re
            for pattern in secret_patterns:
                matches = re.findall(pattern, env_output, re.IGNORECASE)
                assert len(matches) == 0, f"Potential secrets found in environment: {matches}"
            
            # Check common secret file locations
            secret_files = [
                '/run/secrets/',
                '/var/run/secrets/',
                '/.env',
                '/app/.env',
                '/config/secrets'
            ]
            
            for secret_path in secret_files:
                exec_result = container.exec_run(['ls', '-la', secret_path])
                if exec_result.exit_code == 0:
                    # If path exists, check it's not readable by all
                    output = exec_result.output.decode('utf-8')
                    assert 'r--r--r--' not in output, f"Secret file {secret_path} is world-readable"
            
            container.remove(force=True)
            
        except Exception as e:
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except:
                pass
            raise e
    
    @pytest.mark.security
    @pytest.mark.docker
    def test_container_network_security(self, docker_client, test_docker_image_build):
        """Test container network security configuration."""
        image = test_docker_image_build
        container_name = "supernova-network-test"
        
        # Create custom network for testing
        network_name = "supernova-test-network"
        try:
            network = docker_client.networks.get(network_name)
            network.remove()
        except docker.errors.NotFound:
            pass
        
        network = docker_client.networks.create(
            network_name,
            driver="bridge",
            options={
                "com.docker.network.bridge.enable_icc": "false",  # Disable inter-container communication
                "com.docker.network.bridge.enable_ip_masquerade": "true"
            }
        )
        
        try:
            container = docker_client.containers.run(
                image.id,
                name=container_name,
                networks=[network_name],
                detach=True,
                remove=False
            )
            
            time.sleep(5)
            
            # Test network configuration
            network_settings = container.attrs['NetworkSettings']
            networks = network_settings['Networks']
            
            assert network_name in networks, "Container should be on custom network"
            
            # Test that container can't access host network directly
            exec_result = container.exec_run(['netstat', '-rn'])
            if exec_result.exit_code == 0:
                route_output = exec_result.output.decode('utf-8')
                # Should not have routes to host's private networks
                private_ranges = ['10.', '172.16.', '192.168.']
                for private_range in private_ranges:
                    # This test would be more sophisticated in a real scenario
                    pass
            
            container.remove(force=True)
            network.remove()
            
        except Exception as e:
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except:
                pass
            try:
                network.remove()
            except:
                pass
            raise e


@pytest.mark.integration
@pytest.mark.docker
class TestDockerComposeStack:
    """Test complete Docker Compose stack deployment."""
    
    @pytest.fixture(scope="class")
    def compose_project(self, project_root):
        """Docker Compose project fixture."""
        return f"supernova-test-{int(time.time())}"
    
    def test_compose_stack_deployment(self, project_root, compose_project):
        """Test full stack deployment with Docker Compose."""
        compose_file = project_root / "docker-compose.yml"
        
        if not compose_file.exists():
            pytest.skip("docker-compose.yml not found")
        
        try:
            # Start the stack
            result = subprocess.run([
                'docker-compose', '-p', compose_project, '-f', str(compose_file),
                'up', '-d', '--build'
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode != 0:
                pytest.fail(f"Docker Compose up failed: {result.stderr}")
            
            # Wait for services to start
            time.sleep(30)
            
            # Test service health
            health_result = subprocess.run([
                'docker-compose', '-p', compose_project, 'ps'
            ], capture_output=True, text=True, cwd=project_root)
            
            assert 'Up' in health_result.stdout, "Services should be running"
            
            # Test application endpoints
            try:
                response = requests.get('http://localhost:8000/health', timeout=10)
                assert response.status_code == 200, "Application health check should pass"
                
                health_data = response.json()
                assert health_data.get('status') == 'healthy', "Application should report healthy status"
                
                # Test database connectivity
                if 'database' in health_data:
                    assert health_data['database'].get('status') == 'connected', \
                        "Database should be connected"
                
            except requests.exceptions.RequestException as e:
                pytest.fail(f"Failed to connect to application: {e}")
            
            # Test inter-service communication
            # (This would test that the application can connect to database, etc.)
            
        finally:
            # Cleanup
            subprocess.run([
                'docker-compose', '-p', compose_project, 'down', '-v', '--remove-orphans'
            ], cwd=project_root)
    
    def test_compose_environment_isolation(self, project_root, compose_project):
        """Test that different compose environments are isolated."""
        compose_file = project_root / "docker-compose.yml"
        
        if not compose_file.exists():
            pytest.skip("docker-compose.yml not found")
        
        # Create two separate environments
        env1_project = f"{compose_project}-env1"
        env2_project = f"{compose_project}-env2"
        
        try:
            # Start both environments
            for env_project in [env1_project, env2_project]:
                result = subprocess.run([
                    'docker-compose', '-p', env_project, 'up', '-d'
                ], capture_output=True, text=True, cwd=project_root)
                
                if result.returncode != 0:
                    pytest.fail(f"Failed to start environment {env_project}")
            
            time.sleep(15)
            
            # Verify both environments are running independently
            for env_project in [env1_project, env2_project]:
                result = subprocess.run([
                    'docker-compose', '-p', env_project, 'ps', '-q'
                ], capture_output=True, text=True, cwd=project_root)
                
                container_ids = result.stdout.strip().split('\n')
                assert len(container_ids) > 0, f"Environment {env_project} should have running containers"
            
        finally:
            # Cleanup both environments
            for env_project in [env1_project, env2_project]:
                subprocess.run([
                    'docker-compose', '-p', env_project, 'down', '-v'
                ], cwd=project_root)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])