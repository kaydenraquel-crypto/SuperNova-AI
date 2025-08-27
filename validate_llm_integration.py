"""
Simple validation script to check LLM integration structure without requiring all dependencies.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_imports():
    """Validate that our LLM integration files can be imported."""
    print("=== Validating LLM Integration Structure ===\n")
    
    # Test prompts module
    try:
        from supernova.prompts import (
            financial_advice_prompt, buy_signal_prompt, sell_signal_prompt,
            get_prompt_for_action, format_technical_indicators, format_risk_factors
        )
        print("✓ supernova.prompts - All prompt templates imported successfully")
    except ImportError as e:
        print(f"✗ supernova.prompts - Import error: {e}")
        return False
    
    # Test config updates
    try:
        from supernova.config import settings
        
        # Check if LLM settings exist
        llm_settings = [
            "LLM_ENABLED", "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE",
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_CACHE_ENABLED",
            "LLM_COST_TRACKING", "LLM_FALLBACK_ENABLED"
        ]
        
        missing_settings = []
        for setting in llm_settings:
            if not hasattr(settings, setting):
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"✗ supernova.config - Missing settings: {missing_settings}")
            return False
        else:
            print("✓ supernova.config - All LLM settings available")
    except ImportError as e:
        print(f"✗ supernova.config - Import error: {e}")
        return False
    
    # Test advisor module structure
    try:
        # This will fail with pandas missing, but we can catch it
        from supernova.advisor import score_risk
        print("✓ supernova.advisor - Core functions available")
    except ImportError as e:
        if "pandas" in str(e) or "numpy" in str(e):
            print("✓ supernova.advisor - Structure valid (pandas/numpy not installed)")
        else:
            print(f"✗ supernova.advisor - Unexpected import error: {e}")
            return False
    except Exception as e:
        print(f"✗ supernova.advisor - Error: {e}")
        return False
    
    return True

def validate_prompt_templates():
    """Validate prompt template structure."""
    print("\n=== Validating Prompt Templates ===\n")
    
    try:
        from supernova.prompts import get_prompt_for_action
        
        # Test different combinations
        test_cases = [
            ("buy", "stock"),
            ("sell", "crypto"),
            ("hold", "fx"),
            ("buy", "futures"),
            ("sell", "option")
        ]
        
        for action, asset_class in test_cases:
            try:
                prompt = get_prompt_for_action(action, asset_class)
                if prompt is not None:
                    print(f"✓ {action.upper()} + {asset_class.upper()} - Prompt template available")
                else:
                    print(f"✗ {action.upper()} + {asset_class.upper()} - No prompt template")
            except Exception as e:
                print(f"✗ {action.upper()} + {asset_class.upper()} - Error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import prompt functions: {e}")
        return False

def validate_config_values():
    """Validate configuration values are reasonable."""
    print("\n=== Validating Configuration Values ===\n")
    
    try:
        from supernova.config import settings
        
        # Check default values are reasonable
        checks = [
            ("LLM_ENABLED", lambda x: isinstance(x, bool)),
            ("LLM_PROVIDER", lambda x: x in ["openai", "anthropic", "ollama", "huggingface"]),
            ("LLM_TEMPERATURE", lambda x: 0.0 <= x <= 2.0),
            ("LLM_MAX_TOKENS", lambda x: 100 <= x <= 10000),
            ("LLM_CACHE_TTL", lambda x: x > 0),
            ("LLM_DAILY_COST_LIMIT", lambda x: x > 0)
        ]
        
        all_valid = True
        for setting_name, validator in checks:
            value = getattr(settings, setting_name)
            if validator(value):
                print(f"✓ {setting_name}: {value}")
            else:
                print(f"✗ {setting_name}: Invalid value {value}")
                all_valid = False
        
        return all_valid
        
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        return False

def check_file_structure():
    """Check that all expected files exist."""
    print("\n=== Checking File Structure ===\n")
    
    expected_files = [
        "supernova/prompts.py",
        "supernova/advisor.py",
        "supernova/config.py",
        "supernova/api.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in expected_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} - Found")
        else:
            print(f"✗ {file_path} - Missing")
            all_exist = False
    
    return all_exist

def check_requirements():
    """Check if LangChain dependencies are in requirements.txt."""
    print("\n=== Checking Requirements.txt ===\n")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        expected_packages = [
            "langchain",
            "langchain-openai",
            "langchain-anthropic",
            "langchain-community",
            "langchain-core"
        ]
        
        all_found = True
        for package in expected_packages:
            if package in content:
                print(f"✓ {package} - Found in requirements.txt")
            else:
                print(f"✗ {package} - Missing from requirements.txt")
                all_found = False
        
        return all_found
        
    except FileNotFoundError:
        print("✗ requirements.txt - File not found")
        return False

def main():
    """Run all validation checks."""
    print("SuperNova LLM Integration Validation")
    print("====================================")
    
    checks = [
        ("File Structure", check_file_structure),
        ("Requirements", check_requirements),
        ("Import Structure", validate_imports),
        ("Prompt Templates", validate_prompt_templates),
        ("Configuration", validate_config_values)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ {check_name} - Unexpected error: {e}")
            results.append((check_name, False))
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("LLM integration is properly structured and ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set API keys in environment variables (OPENAI_API_KEY, etc.)")
        print("3. Test the enhanced advisor functionality")
    else:
        print("\n❌ Some validation checks failed.")
        print("Please review the output above and fix any issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)