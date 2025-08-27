"""
SuperNova AI Advanced Injection Prevention System
Enhanced protection against SQL, NoSQL, XSS, and other injection attacks
"""

import re
import html
import json
import base64
import hashlib
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from datetime import datetime
import logging
from collections import deque
from functools import wraps
import time

from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES
from .security_config import security_settings

logger = logging.getLogger(__name__)


class InjectionAttemptException(Exception):
    """Exception raised when injection attempt is detected"""
    def __init__(self, attack_type: str, pattern: str, input_data: str):
        self.attack_type = attack_type
        self.pattern = pattern
        self.input_data = input_data[:100]  # Truncate for security
        super().__init__(f"{attack_type} injection attempt detected")


class AdvancedInjectionPrevention:
    """
    Advanced injection prevention system with machine learning-like pattern detection,
    context-aware validation, and behavioral analysis
    """
    
    def __init__(self):
        # Enhanced SQL injection patterns with context awareness
        self.sql_injection_signatures = {
            'union_based': [
                r'(\bunion\b\s+\bselect\b)|(\bselect\b.*\bunion\b)',
                r'\bunion\b\s+\ball\s+\bselect\b',
                r'\bunion\b.*\bfrom\b.*\btables\b',
                r'\bunion\b.*\binformation_schema\b'
            ],
            'boolean_based': [
                r"(\bor\b\s+\d+\s*=\s*\d+)|(\band\b\s+\d+\s*=\s*\d+)",
                r"(\bor\b\s+'\d+'\s*=\s*'\d+')|\band\b\s+'\d+'\s*=\s*'\d+')",
                r"\bor\b\s+\btrue\b|\band\b\s+\bfalse\b",
                r"(\bor\b\s+\d+>0)|(\band\b\s+\d+<0)"
            ],
            'time_based': [
                r'\bwaitfor\b\s+\bdelay\b',
                r'\bsleep\b\s*\(\s*\d+\s*\)',
                r'\bbenchmark\b\s*\(',
                r'\bpg_sleep\b\s*\('
            ],
            'stacked_queries': [
                r';\s*\b(select|insert|update|delete|drop|create|alter)\b',
                r';\s*\bexec\b|\bexecute\b',
                r';\s*\bsp_\w+',
                r';\s*\bxp_\w+'
            ],
            'function_based': [
                r'\b(cast|convert|char|nchar|ascii|substring)\b\s*\(',
                r'\b(concat|group_concat|string_agg)\b\s*\(',
                r'\b(load_file|outfile|dumpfile)\b',
                r'\b(user|database|version|@@)\w*'
            ]
        }
        
        # Enhanced NoSQL injection patterns
        self.nosql_injection_signatures = {
            'mongodb': [
                r'\$where\s*:\s*["\'].*["\']',
                r'\$regex\s*:\s*["\'].*["\']',
                r'\$ne\s*:\s*(?:null|1)',
                r'\$exists\s*:\s*true',
                r'\$nin\s*:\s*\[\]',
                r'this\.\w+\.match\(',
                r'function\s*\(\s*\)\s*\{.*\}'
            ],
            'couchdb': [
                r'startkey.*endkey',
                r'emit\s*\(',
                r'function\s*\(doc\)',
                r'_design\/.*\/_view\/'
            ],
            'redis': [
                r'\beval\b\s*["\'].*["\']',
                r'\bflushdb\b|\bflushall\b',
                r'\bconfig\b\s+\bset\b',
                r'\bshutdown\b|\brestart\b'
            ]
        }
        
        # Advanced XSS patterns with context awareness
        self.xss_signatures = {
            'script_based': [
                r'<script[^>]*>.*?<\/script>',
                r'<script[^>]*src\s*=',
                r'javascript\s*:[^"\']*',
                r'vbscript\s*:[^"\']*'
            ],
            'event_handler': [
                r'on\w+\s*=\s*["\'][^"\']*["\']',
                r'on(load|error|click|mouseover|focus|blur)\s*=',
                r'on(keydown|keyup|keypress)\s*=',
                r'on(submit|change|select)\s*='
            ],
            'iframe_object': [
                r'<iframe[^>]*src\s*=',
                r'<object[^>]*data\s*=',
                r'<embed[^>]*src\s*=',
                r'<link[^>]*href\s*=.*javascript'
            ],
            'encoded_payloads': [
                r'%3C%73%63%72%69%70%74',  # <script encoded
                r'&lt;script',
                r'&#x3C;script',
                r'\\u003cscript'
            ],
            'attribute_breaking': [
                r'["\'][^"\']*["\']\s*>',
                r'["\'][^"\']*javascript:',
                r'["\'][^"\']*on\w+\s*=',
                r'style\s*=.*expression\s*\('
            ]
        }
        
        # Command injection patterns
        self.command_injection_signatures = {
            'shell_metacharacters': [
                r'[;&|`$(){}\\]',
                r'\|\||\&\&',
                r'>>\s*[/\\]|\<\<',
                r'\$\{.*\}|\$\(.*\)'
            ],
            'common_commands': [
                r'\b(cat|ls|dir|pwd|whoami|id|uname|ps|netstat|ifconfig)\b',
                r'\b(rm|del|mv|move|cp|copy|chmod|chown|kill|killall)\b',
                r'\b(wget|curl|nc|ncat|telnet|ssh|ftp)\b',
                r'\b(python|perl|ruby|bash|sh|cmd|powershell)\b'
            ],
            'path_traversal': [
                r'\.\.[\\/]',
                r'[\\/]etc[\\/]passwd',
                r'[\\/]windows[\\/]system32',
                r'[\\/]proc[\\/].*'
            ]
        }
        
        # LDAP injection patterns
        self.ldap_injection_patterns = [
            r'\*\)\(.*=\*',
            r'\)\(.*=',
            r'\(\|.*\)',
            r'\(\&.*\)',
            r'\(\!.*\)'
        ]
        
        # XPath injection patterns
        self.xpath_injection_patterns = [
            r'or\s+1=1',
            r"'\s*or\s*'1'='1",
            r'and\s+1=2',
            r"'\s*and\s*'1'='2"
        ]
        
        # Template injection patterns
        self.template_injection_patterns = [
            r'\{\{.*\}\}',
            r'\{\%.*\%\}',
            r'\$\{.*\}',
            r'<%.*%>',
            r'<%=.*%>'
        ]
        
        # Initialize behavioral analysis
        self.attack_patterns_cache = {}
        self.client_behavior = {}
        self.suspicious_ips = set()
        self.pattern_frequency = {}
        
        # Initialize ML-like scoring system
        self.threat_scores = {
            'sql_injection': {'base_score': 95, 'weight': 1.0},
            'nosql_injection': {'base_score': 90, 'weight': 0.9},
            'xss': {'base_score': 85, 'weight': 0.85},
            'command_injection': {'base_score': 98, 'weight': 1.0},
            'ldap_injection': {'base_score': 80, 'weight': 0.8},
            'xpath_injection': {'base_score': 75, 'weight': 0.75},
            'template_injection': {'base_score': 88, 'weight': 0.88}
        }
    
    def comprehensive_injection_detection(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive injection detection with context awareness and behavioral analysis
        """
        detection_result = {
            'safe': True,
            'threat_score': 0,
            'attack_types': [],
            'patterns_detected': [],
            'behavioral_flags': [],
            'sanitized_data': None,
            'recommendations': []
        }
        
        if not isinstance(input_data, (str, dict, list)):
            detection_result['sanitized_data'] = input_data
            return detection_result
        
        try:
            # Convert input to string for pattern matching
            if isinstance(input_data, (dict, list)):
                input_string = json.dumps(input_data, default=str)
            else:
                input_string = str(input_data)
            
            # Normalize input for better detection
            normalized_input = self._normalize_input(input_string)
            
            # Perform multi-layer detection
            sql_result = self._detect_sql_injection(normalized_input, context)
            nosql_result = self._detect_nosql_injection(normalized_input, context)
            xss_result = self._detect_xss_injection(normalized_input, context)
            command_result = self._detect_command_injection(normalized_input, context)
            other_result = self._detect_other_injections(normalized_input)
            
            # Combine results
            all_results = [sql_result, nosql_result, xss_result, command_result, other_result]
            
            for result in all_results:
                if not result['safe']:
                    detection_result['safe'] = False
                    detection_result['attack_types'].extend(result['attack_types'])
                    detection_result['patterns_detected'].extend(result['patterns'])
                    detection_result['threat_score'] = max(detection_result['threat_score'], result['threat_score'])
            
            # Behavioral analysis
            if client_ip or user_id:
                behavioral_analysis = self._analyze_behavior(
                    input_string, client_ip, user_id, detection_result['attack_types']
                )
                detection_result['behavioral_flags'] = behavioral_analysis['flags']
                detection_result['threat_score'] += behavioral_analysis['score_adjustment']
            
            # Context-aware adjustments
            if context:
                context_adjustment = self._apply_context_adjustments(detection_result, context)
                detection_result['threat_score'] += context_adjustment
            
            # Generate recommendations
            detection_result['recommendations'] = self._generate_recommendations(detection_result)
            
            # Sanitize data if safe enough
            if detection_result['threat_score'] < 70:  # Configurable threshold
                detection_result['sanitized_data'] = self._sanitize_input(input_data, detection_result)
            
            # Log security event if threat detected
            if not detection_result['safe']:
                self._log_injection_attempt(detection_result, client_ip, user_id)
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error in injection detection: {str(e)}")
            detection_result['safe'] = False
            detection_result['threat_score'] = 100
            detection_result['attack_types'] = ['unknown']
            return detection_result
    
    def _normalize_input(self, input_string: str) -> str:
        """Normalize input for better pattern detection"""
        # Decode common encodings
        normalized = input_string
        
        try:
            # URL decode
            normalized = urllib.parse.unquote(normalized)
            normalized = urllib.parse.unquote_plus(normalized)
            
            # HTML decode
            normalized = html.unescape(normalized)
            
            # Base64 decode if it looks like base64
            if re.match(r'^[A-Za-z0-9+/]*={0,2}$', normalized) and len(normalized) % 4 == 0:
                try:
                    decoded = base64.b64decode(normalized).decode('utf-8', errors='ignore')
                    if len(decoded) > 0:
                        normalized += ' ' + decoded
                except:
                    pass
            
            # Handle Unicode escapes
            try:
                normalized = normalized.encode().decode('unicode_escape', errors='ignore')
            except:
                pass
            
            # Convert to lowercase for case-insensitive matching
            normalized_lower = normalized.lower()
            
            return normalized_lower
            
        except Exception:
            return input_string.lower()
    
    def _detect_sql_injection(
        self,
        input_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Advanced SQL injection detection with context awareness"""
        
        result = {
            'safe': True,
            'threat_score': 0,
            'attack_types': [],
            'patterns': []
        }
        
        max_score = 0
        
        for injection_type, patterns in self.sql_injection_signatures.items():
            for pattern in patterns:
                if re.search(pattern, input_string, re.IGNORECASE | re.DOTALL):
                    result['safe'] = False
                    attack_type = f'sql_injection_{injection_type}'
                    result['attack_types'].append(attack_type)
                    result['patterns'].append(pattern)
                    
                    # Calculate threat score based on injection type
                    type_scores = {
                        'union_based': 95,
                        'boolean_based': 85,
                        'time_based': 90,
                        'stacked_queries': 98,
                        'function_based': 80
                    }
                    
                    score = type_scores.get(injection_type, 75)
                    max_score = max(max_score, score)
        
        # Context-aware adjustments
        if context:
            # If input is expected to contain SQL-like syntax (e.g., query builder)
            if context.get('expects_sql_syntax'):
                max_score = max(0, max_score - 30)
            
            # If input is in a restricted context (e.g., username field)
            if context.get('restricted_field'):
                max_score += 20
        
        result['threat_score'] = max_score
        return result
    
    def _detect_nosql_injection(
        self,
        input_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Advanced NoSQL injection detection"""
        
        result = {
            'safe': True,
            'threat_score': 0,
            'attack_types': [],
            'patterns': []
        }
        
        max_score = 0
        
        for db_type, patterns in self.nosql_injection_signatures.items():
            for pattern in patterns:
                if re.search(pattern, input_string, re.IGNORECASE | re.DOTALL):
                    result['safe'] = False
                    result['attack_types'].append(f'nosql_injection_{db_type}')
                    result['patterns'].append(pattern)
                    
                    # Score based on database type and pattern
                    scores = {'mongodb': 90, 'couchdb': 80, 'redis': 95}
                    score = scores.get(db_type, 75)
                    max_score = max(max_score, score)
        
        result['threat_score'] = max_score
        return result
    
    def _detect_xss_injection(
        self,
        input_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Advanced XSS detection with context awareness"""
        
        result = {
            'safe': True,
            'threat_score': 0,
            'attack_types': [],
            'patterns': []
        }
        
        max_score = 0
        
        for xss_type, patterns in self.xss_signatures.items():
            for pattern in patterns:
                if re.search(pattern, input_string, re.IGNORECASE | re.DOTALL):
                    result['safe'] = False
                    result['attack_types'].append(f'xss_{xss_type}')
                    result['patterns'].append(pattern)
                    
                    # Score based on XSS type
                    type_scores = {
                        'script_based': 95,
                        'event_handler': 85,
                        'iframe_object': 90,
                        'encoded_payloads': 88,
                        'attribute_breaking': 80
                    }
                    
                    score = type_scores.get(xss_type, 75)
                    max_score = max(max_score, score)
        
        # Context adjustments for XSS
        if context:
            # If input will be displayed in HTML context
            if context.get('html_output'):
                max_score += 15
            
            # If input is for rich text editor
            if context.get('rich_text_editor'):
                max_score = max(0, max_score - 25)
        
        result['threat_score'] = max_score
        return result
    
    def _detect_command_injection(
        self,
        input_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Advanced command injection detection"""
        
        result = {
            'safe': True,
            'threat_score': 0,
            'attack_types': [],
            'patterns': []
        }
        
        max_score = 0
        
        for cmd_type, patterns in self.command_injection_signatures.items():
            for pattern in patterns:
                if re.search(pattern, input_string, re.IGNORECASE):
                    result['safe'] = False
                    result['attack_types'].append(f'command_injection_{cmd_type}')
                    result['patterns'].append(pattern)
                    
                    # High scores for command injection
                    type_scores = {
                        'shell_metacharacters': 95,
                        'common_commands': 90,
                        'path_traversal': 98
                    }
                    
                    score = type_scores.get(cmd_type, 85)
                    max_score = max(max_score, score)
        
        result['threat_score'] = max_score
        return result
    
    def _detect_other_injections(self, input_string: str) -> Dict[str, Any]:
        """Detect other types of injections (LDAP, XPath, Template)"""
        
        result = {
            'safe': True,
            'threat_score': 0,
            'attack_types': [],
            'patterns': []
        }
        
        max_score = 0
        
        # LDAP injection detection
        for pattern in self.ldap_injection_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                result['safe'] = False
                result['attack_types'].append('ldap_injection')
                result['patterns'].append(pattern)
                max_score = max(max_score, 80)
        
        # XPath injection detection
        for pattern in self.xpath_injection_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                result['safe'] = False
                result['attack_types'].append('xpath_injection')
                result['patterns'].append(pattern)
                max_score = max(max_score, 75)
        
        # Template injection detection
        for pattern in self.template_injection_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                result['safe'] = False
                result['attack_types'].append('template_injection')
                result['patterns'].append(pattern)
                max_score = max(max_score, 88)
        
        result['threat_score'] = max_score
        return result
    
    def _analyze_behavior(
        self,
        input_string: str,
        client_ip: Optional[str],
        user_id: Optional[str],
        attack_types: List[str]
    ) -> Dict[str, Any]:
        """Behavioral analysis for pattern detection"""
        
        analysis = {
            'flags': [],
            'score_adjustment': 0
        }
        
        current_time = time.time()
        
        # Track client behavior
        if client_ip:
            if client_ip not in self.client_behavior:
                self.client_behavior[client_ip] = {
                    'requests': deque(maxlen=100),
                    'attack_attempts': deque(maxlen=50),
                    'patterns': set()
                }
            
            client_data = self.client_behavior[client_ip]
            client_data['requests'].append(current_time)
            
            if attack_types:
                client_data['attack_attempts'].append(current_time)
                client_data['patterns'].update(attack_types)
            
            # Check for rapid fire requests
            recent_requests = [t for t in client_data['requests'] if current_time - t < 60]
            if len(recent_requests) > 50:  # More than 50 requests per minute
                analysis['flags'].append('rapid_requests')
                analysis['score_adjustment'] += 10
            
            # Check for repeated attacks
            recent_attacks = [t for t in client_data['attack_attempts'] if current_time - t < 300]
            if len(recent_attacks) > 3:  # More than 3 attacks in 5 minutes
                analysis['flags'].append('repeated_attacks')
                analysis['score_adjustment'] += 20
                self.suspicious_ips.add(client_ip)
            
            # Check for pattern diversity (might indicate automated scanning)
            if len(client_data['patterns']) > 5:
                analysis['flags'].append('diverse_attack_patterns')
                analysis['score_adjustment'] += 15
        
        return analysis
    
    def _apply_context_adjustments(
        self,
        detection_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> int:
        """Apply context-aware adjustments to threat score"""
        
        adjustment = 0
        
        # Field type adjustments
        field_type = context.get('field_type')
        if field_type == 'password':
            adjustment += 20  # Higher scrutiny for password fields
        elif field_type == 'email':
            adjustment += 10
        elif field_type == 'url':
            adjustment -= 5  # URLs might contain special characters
        
        # User role adjustments
        user_role = context.get('user_role')
        if user_role == 'admin':
            adjustment -= 10  # Admins might need more flexibility
        elif user_role == 'anonymous':
            adjustment += 15  # Anonymous users get higher scrutiny
        
        # Endpoint security level
        security_level = context.get('security_level')
        if security_level == 'high':
            adjustment += 25
        elif security_level == 'low':
            adjustment -= 10
        
        return adjustment
    
    def _generate_recommendations(self, detection_result: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on detection results"""
        
        recommendations = []
        
        if 'sql_injection' in str(detection_result['attack_types']):
            recommendations.extend([
                "Use parameterized queries/prepared statements",
                "Implement input validation with whitelisting",
                "Apply least privilege principle to database connections",
                "Enable database query logging and monitoring"
            ])
        
        if 'xss' in str(detection_result['attack_types']):
            recommendations.extend([
                "Implement Content Security Policy (CSP) headers",
                "Use output encoding/escaping for all user data",
                "Validate and sanitize all input data",
                "Consider using a web application firewall (WAF)"
            ])
        
        if 'command_injection' in str(detection_result['attack_types']):
            recommendations.extend([
                "Avoid system command execution with user input",
                "Use safe APIs instead of shell commands",
                "Implement strict input validation",
                "Run applications with minimal privileges"
            ])
        
        if detection_result['threat_score'] > 80:
            recommendations.append("Consider implementing CAPTCHA for repeated violations")
        
        if 'rapid_requests' in detection_result['behavioral_flags']:
            recommendations.append("Implement rate limiting to prevent abuse")
        
        return recommendations
    
    def _sanitize_input(self, input_data: Any, detection_result: Dict[str, Any]) -> Any:
        """Sanitize input data based on detection results"""
        
        if isinstance(input_data, str):
            sanitized = input_data
            
            # Apply basic HTML sanitization
            sanitized = html.escape(sanitized)
            
            # Remove dangerous patterns (conservative approach)
            dangerous_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript\s*:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>'
            ]
            
            for pattern in dangerous_patterns:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            
            return sanitized
            
        elif isinstance(input_data, dict):
            return {k: self._sanitize_input(v, detection_result) for k, v in input_data.items()}
            
        elif isinstance(input_data, list):
            return [self._sanitize_input(item, detection_result) for item in input_data]
        
        return input_data
    
    def _log_injection_attempt(
        self,
        detection_result: Dict[str, Any],
        client_ip: Optional[str],
        user_id: Optional[str]
    ):
        """Log injection attempt for security monitoring"""
        
        security_logger.log_security_event(
            event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
            level=SecurityEventLevel.WARNING,
            client_ip=client_ip,
            user_id=user_id,
            details={
                "injection_attempt": True,
                "attack_types": detection_result['attack_types'],
                "threat_score": detection_result['threat_score'],
                "patterns_detected": len(detection_result['patterns_detected']),
                "behavioral_flags": detection_result['behavioral_flags']
            }
        )
    
    def is_suspicious_ip(self, client_ip: str) -> bool:
        """Check if IP is marked as suspicious"""
        return client_ip in self.suspicious_ips
    
    def clear_behavioral_data(self, older_than_hours: int = 24):
        """Clear old behavioral data to prevent memory bloat"""
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        # Clean up client behavior data
        ips_to_remove = []
        for ip, data in self.client_behavior.items():
            # Filter out old requests
            data['requests'] = deque([t for t in data['requests'] if t > cutoff_time], maxlen=100)
            data['attack_attempts'] = deque([t for t in data['attack_attempts'] if t > cutoff_time], maxlen=50)
            
            # Remove IP if no recent activity
            if not data['requests']:
                ips_to_remove.append(ip)
        
        for ip in ips_to_remove:
            del self.client_behavior[ip]
            self.suspicious_ips.discard(ip)


# Global instance
injection_prevention = AdvancedInjectionPrevention()


# Decorator for automatic injection prevention
def prevent_injection(
    context: Optional[Dict[str, Any]] = None,
    threat_threshold: int = 70
):
    """Decorator to automatically prevent injection attacks on function inputs"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get client info from request context if available
            client_ip = None
            user_id = None
            
            # Check Flask or FastAPI request context
            try:
                from flask import request as flask_request, g
                client_ip = flask_request.remote_addr
                user_id = getattr(g, 'user_id', None)
            except ImportError:
                try:
                    # FastAPI context would need to be passed explicitly
                    pass
                except ImportError:
                    pass
            
            # Analyze all string arguments and keyword arguments
            for arg in args:
                if isinstance(arg, (str, dict, list)):
                    result = injection_prevention.comprehensive_injection_detection(
                        arg, context, client_ip, user_id
                    )
                    if result['threat_score'] >= threat_threshold:
                        raise InjectionAttemptException(
                            result['attack_types'][0] if result['attack_types'] else 'unknown',
                            result['patterns_detected'][0] if result['patterns_detected'] else '',
                            str(arg)
                        )
            
            for key, value in kwargs.items():
                if isinstance(value, (str, dict, list)):
                    result = injection_prevention.comprehensive_injection_detection(
                        value, context, client_ip, user_id
                    )
                    if result['threat_score'] >= threat_threshold:
                        raise InjectionAttemptException(
                            result['attack_types'][0] if result['attack_types'] else 'unknown',
                            result['patterns_detected'][0] if result['patterns_detected'] else '',
                            str(value)
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator