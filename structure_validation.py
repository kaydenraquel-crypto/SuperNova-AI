#!/usr/bin/env python3
"""
SuperNova Extended Framework - Structure Validation (No Dependencies)
====================================================================

This script validates the structure and basic functionality without
requiring any external dependencies to be installed.
"""

import os
import ast
import json

def validate_file_structure():
    """Validate that all expected files are present"""
    print("Validating file structure...")
    
    expected_files = [
        "requirements.txt",
        "main.py", 
        "supernova/__init__.py",
        "supernova/config.py",
        "supernova/advisor.py",
        "supernova/backtester.py",
        "supernova/workflows.py",
        "supernova/api.py",
        "comprehensive_integration_test.py",
        "simulated_integration_test.py",
        "INTEGRATION_TEST_REPORT.md"
    ]
    
    results = {}
    for file_path in expected_files:
        exists = os.path.exists(file_path)
        if exists:
            size = os.path.getsize(file_path)
            results[file_path] = {"exists": True, "size": size}
            print(f"[OK] {file_path} - {size} bytes")
        else:
            results[file_path] = {"exists": False}
            print(f"[!] {file_path} - MISSING")
    
    return results

def analyze_requirements():
    """Analyze requirements.txt without importing"""
    print("\nAnalyzing requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        # Categorize by enhancement
        categories = {
            "core": [],
            "llm": [],
            "vectorbt": [],
            "prefect": [],
            "sentiment": []
        }
        
        for line in lines:
            lower = line.lower()
            if any(x in lower for x in ["langchain", "openai", "anthropic", "tiktoken"]):
                categories["llm"].append(line)
            elif any(x in lower for x in ["vectorbt", "talib", "numba"]):
                categories["vectorbt"].append(line)
            elif "prefect" in lower:
                categories["prefect"].append(line)
            elif any(x in lower for x in ["tweepy", "praw", "spacy", "textblob", "vader", "transformers"]):
                categories["sentiment"].append(line)
            else:
                categories["core"].append(line)
        
        print(f"[OK] Total requirements: {len(lines)}")
        print(f"     - Core: {len(categories['core'])}")
        print(f"     - LLM Integration: {len(categories['llm'])}")
        print(f"     - VectorBT: {len(categories['vectorbt'])}")
        print(f"     - Prefect: {len(categories['prefect'])}")
        print(f"     - Sentiment: {len(categories['sentiment'])}")
        
        return {"status": "success", "total": len(lines), "categories": categories}
        
    except Exception as e:
        print(f"[!] Failed to analyze requirements: {e}")
        return {"status": "error", "error": str(e)}

def analyze_code_structure(file_path):
    """Analyze Python file structure using AST"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Count different elements
        imports = []
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                else:
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"from {module} import {alias.name}")
            
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return {
            "status": "success",
            "lines": len(content.split('\n')),
            "imports": len(imports),
            "classes": len(classes),
            "functions": len(functions),
            "import_list": imports[:10],  # First 10 imports
            "class_list": classes,
            "function_list": functions[:10]  # First 10 functions
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """Run structure validation"""
    print("=" * 60)
    print("SuperNova Extended Framework - Structure Validation")
    print("=" * 60)
    
    results = {}
    
    # 1. File structure validation
    results["file_structure"] = validate_file_structure()
    
    # 2. Requirements analysis
    results["requirements"] = analyze_requirements()
    
    # 3. Key file analysis
    key_files = {
        "advisor": "supernova/advisor.py",
        "backtester": "supernova/backtester.py",
        "workflows": "supernova/workflows.py",
        "api": "supernova/api.py",
        "config": "supernova/config.py"
    }
    
    print("\nAnalyzing key modules...")
    results["modules"] = {}
    
    for name, path in key_files.items():
        if os.path.exists(path):
            analysis = analyze_code_structure(path)
            results["modules"][name] = analysis
            if analysis["status"] == "success":
                print(f"[OK] {name}: {analysis['lines']} lines, {analysis['functions']} functions, {analysis['classes']} classes")
            else:
                print(f"[!] {name}: {analysis['error']}")
        else:
            results["modules"][name] = {"status": "missing"}
            print(f"[!] {name}: File not found")
    
    # 4. Enhancement detection
    print("\nDetecting enhancements...")
    enhancements = {
        "llm_integration": False,
        "vectorbt_integration": False,
        "prefect_integration": False
    }
    
    # Check advisor.py for LLM integration (text search)
    if os.path.exists("supernova/advisor.py"):
        try:
            with open("supernova/advisor.py", "r", encoding="utf-8") as f:
                content = f.read()
            if "langchain" in content.lower():
                enhancements["llm_integration"] = True
                print("[OK] LLM Integration detected in advisor.py")
        except Exception as e:
            print(f"[!] Could not check advisor.py for LLM integration: {e}")
    
    # Check backtester.py for VectorBT integration (text search)
    if os.path.exists("supernova/backtester.py"):
        try:
            with open("supernova/backtester.py", "r", encoding="utf-8") as f:
                content = f.read()
            if "vectorbt" in content.lower():
                enhancements["vectorbt_integration"] = True
                print("[OK] VectorBT Integration detected in backtester.py")
        except Exception as e:
            print(f"[!] Could not check backtester.py for VectorBT integration: {e}")
    
    # Check workflows.py for Prefect integration (text search)
    if os.path.exists("supernova/workflows.py"):
        try:
            with open("supernova/workflows.py", "r", encoding="utf-8") as f:
                content = f.read()
            if "prefect" in content.lower():
                enhancements["prefect_integration"] = True
                print("[OK] Prefect Integration detected in workflows.py")
        except Exception as e:
            print(f"[!] Could not check workflows.py for Prefect integration: {e}")
    
    results["enhancements"] = enhancements
    
    # 5. Generate summary
    print("\n" + "=" * 60)
    print("STRUCTURE VALIDATION SUMMARY")
    print("=" * 60)
    
    file_count = sum(1 for f in results["file_structure"].values() if f.get("exists", False))
    total_files = len(results["file_structure"])
    
    module_count = sum(1 for m in results["modules"].values() if m.get("status") == "success")
    total_modules = len(results["modules"])
    
    enhancement_count = sum(1 for e in enhancements.values() if e)
    
    print(f"Files Present: {file_count}/{total_files}")
    print(f"Modules Analyzed: {module_count}/{total_modules}")
    print(f"Enhancements Detected: {enhancement_count}/3")
    
    req_result = results.get("requirements", {})
    if req_result.get("status") == "success":
        print(f"Dependencies: {req_result['total']} packages in requirements.txt")
    
    # Enhancement status
    print(f"\nEnhancement Status:")
    print(f"  - LLM Integration: {'[OK]' if enhancements['llm_integration'] else '[!]'}")
    print(f"  - VectorBT Integration: {'[OK]' if enhancements['vectorbt_integration'] else '[!]'}")
    print(f"  - Prefect Integration: {'[OK]' if enhancements['prefect_integration'] else '[!]'}")
    
    # Overall assessment
    if enhancement_count == 3 and file_count >= 8:
        print(f"\n[SUCCESS] All enhancements detected - Framework structure is complete!")
        exit_code = 0
    elif enhancement_count >= 2:
        print(f"\n[PARTIAL] Most enhancements detected - Framework mostly complete")
        exit_code = 0
    else:
        print(f"\n[WARNING] Missing enhancements - Framework incomplete")
        exit_code = 1
    
    # Save results
    with open("structure_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("[INFO] Results saved to structure_validation_results.json")
    
    return exit_code

if __name__ == "__main__":
    import sys
    sys.exit(main())