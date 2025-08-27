#!/usr/bin/env python3
"""
SuperNova AI - Quick Start Launcher
Smart detection and execution of the best setup method for your system
"""

import sys
import os
import platform
import subprocess
import webbrowser
from pathlib import Path

def print_banner():
    """Print SuperNova AI banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               🚀 SuperNova AI Quick Start 🚀                ║
    ║                                                              ║
    ║           Enterprise Financial Intelligence Platform         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def detect_system():
    """Detect system type and capabilities"""
    system_info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'architecture': platform.machine(),
        'python_version': sys.version,
        'has_gui': False,
        'has_git': False,
        'has_node': False,
        'recommended_method': None
    }
    
    # Check for GUI availability
    try:
        if system_info['os'] == 'Windows':
            system_info['has_gui'] = True
        else:
            # Try to import tkinter
            import tkinter
            system_info['has_gui'] = True
    except ImportError:
        pass
    
    # Check for Git
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        system_info['has_git'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for Node.js
    try:
        subprocess.run(['node', '--version'], capture_output=True, check=True)
        system_info['has_node'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Determine recommended method
    if system_info['os'] == 'Windows':
        system_info['recommended_method'] = 'batch'
    elif system_info['os'] in ['Darwin', 'Linux']:
        system_info['recommended_method'] = 'shell'
    else:
        system_info['recommended_method'] = 'manual'
    
    return system_info

def print_system_info(system_info):
    """Print detected system information"""
    print("🔍 System Detection Results:")
    print(f"   Operating System: {system_info['os']} ({system_info['architecture']})")
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   GUI Available: {'✅ Yes' if system_info['has_gui'] else '❌ No'}")
    print(f"   Git Available: {'✅ Yes' if system_info['has_git'] else '❌ No'}")
    print(f"   Node.js Available: {'✅ Yes' if system_info['has_node'] else '❌ No'}")
    print()

def show_setup_options(system_info):
    """Show available setup options"""
    print("📋 Available Setup Methods:")
    print()
    
    if system_info['os'] == 'Windows':
        print("   1. 🚀 One-Click Windows Setup (Recommended)")
        print("      - Double-click setup.bat or run from command prompt")
        print("      - Automatic dependency checking and installation")
        print("      - Creates convenient startup shortcuts")
        print()
    
    if system_info['os'] in ['Darwin', 'Linux']:
        print("   2. 🚀 Shell Script Setup (Recommended)")
        print("      - Run: chmod +x setup.sh && ./setup.sh")
        print("      - Cross-platform compatibility")
        print("      - Colored terminal output and progress indicators")
        print()
    
    if system_info['has_gui']:
        print("   3. 🖥️ GUI Setup Wizard")
        print("      - Run: python setup_wizard.py")
        print("      - Step-by-step visual interface")
        print("      - Interactive configuration")
        print()
    
    print("   4. 🐳 Docker Setup (Advanced)")
    print("      - Run: docker-compose up -d")
    print("      - Containerized deployment")
    print("      - Production-ready configuration")
    print()
    
    print("   5. ⚙️ Manual Setup (Experts)")
    print("      - Follow SETUP_GUIDE.md instructions")
    print("      - Full control over configuration")
    print("      - Custom deployment options")
    print()

def run_recommended_setup(system_info):
    """Execute the recommended setup method"""
    if system_info['recommended_method'] == 'batch':
        print("🚀 Starting Windows batch setup...")
        if os.path.exists('setup.bat'):
            if os.name == 'nt':
                os.system('setup.bat')
            else:
                print("❌ Windows batch file detected but not on Windows system")
        else:
            print("❌ setup.bat not found")
    
    elif system_info['recommended_method'] == 'shell':
        print("🚀 Starting shell script setup...")
        if os.path.exists('setup.sh'):
            # Make executable and run
            os.chmod('setup.sh', 0o755)
            os.system('./setup.sh')
        else:
            print("❌ setup.sh not found")
    
    else:
        print("ℹ️ Please choose a setup method manually")

def check_existing_installation():
    """Check if SuperNova AI is already installed"""
    indicators = [
        '.venv',
        'frontend/node_modules',
        '.env',
        'supernova.db'
    ]
    
    existing = []
    for indicator in indicators:
        if os.path.exists(indicator):
            existing.append(indicator)
    
    if existing:
        print("🔍 Existing installation detected:")
        for item in existing:
            print(f"   ✅ {item}")
        print()
        
        response = input("Would you like to:\n"
                        "  1. Start existing installation\n"
                        "  2. Reinstall/update\n"
                        "  3. Exit\n"
                        "Choose (1-3): ").strip()
        
        if response == '1':
            start_existing_installation()
            return True
        elif response == '2':
            print("Proceeding with reinstallation...")
            return False
        else:
            print("Goodbye! 👋")
            sys.exit(0)
    
    return False

def start_existing_installation():
    """Start existing SuperNova AI installation"""
    print("🚀 Starting SuperNova AI...")
    
    system = platform.system()
    
    if system == 'Windows':
        if os.path.exists('start_supernova.bat'):
            os.system('start_supernova.bat')
        else:
            print("Starting services manually...")
            print("Backend: uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081")
            print("Frontend: cd frontend && npm start")
    else:
        if os.path.exists('start_supernova.sh'):
            os.chmod('start_supernova.sh', 0o755)
            os.system('./start_supernova.sh')
        else:
            print("Starting services manually...")
            print("Run these commands in separate terminals:")
            print("Backend: source .venv/bin/activate && uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081")
            print("Frontend: cd frontend && npm start")
    
    # Wait and open browser
    import time
    print("⏳ Waiting for services to start...")
    time.sleep(5)
    
    try:
        webbrowser.open('http://localhost:3000')
        print("🌐 Opening SuperNova AI in your browser...")
    except Exception:
        print("Please open http://localhost:3000 in your browser")

def show_help():
    """Show help information"""
    help_text = """
🆘 SuperNova AI Quick Start Help

This launcher automatically detects your system and recommends the best
installation method for SuperNova AI.

Usage:
  python quick_start.py [options]

Options:
  --gui          Force GUI setup wizard
  --batch        Force Windows batch setup (Windows only)  
  --shell        Force shell script setup (macOS/Linux only)
  --docker       Show Docker setup instructions
  --manual       Show manual setup instructions
  --help         Show this help message

System Requirements:
  - Python 3.8+ (3.11+ recommended)
  - Node.js 16+ (18+ recommended)
  - 4GB RAM (8GB recommended)
  - 2GB disk space

Troubleshooting:
  - Ensure Python and Node.js are in your PATH
  - Run as administrator/sudo if permission errors occur
  - Check SETUP_GUIDE.md for detailed instructions
  - Verify internet connection for downloading dependencies

Support:
  - Documentation: docs/ folder
  - Issues: GitHub repository
  - Setup Guide: SETUP_GUIDE.md
    """
    print(help_text)

def main():
    """Main entry point"""
    # Check command line arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return
    
    print_banner()
    
    # Check if already installed
    if check_existing_installation():
        return
    
    # Detect system
    print("🔍 Detecting your system...")
    system_info = detect_system()
    print_system_info(system_info)
    
    # Handle forced options
    if '--gui' in sys.argv:
        if system_info['has_gui']:
            print("🖥️ Starting GUI setup wizard...")
            os.system('python setup_wizard.py')
        else:
            print("❌ GUI not available on this system")
        return
    
    if '--batch' in sys.argv:
        if system_info['os'] == 'Windows':
            os.system('setup.bat')
        else:
            print("❌ Batch setup only available on Windows")
        return
    
    if '--shell' in sys.argv:
        if os.path.exists('setup.sh'):
            os.chmod('setup.sh', 0o755)
            os.system('./setup.sh')
        else:
            print("❌ setup.sh not found")
        return
    
    if '--docker' in sys.argv:
        print("🐳 Docker Setup Instructions:")
        print("1. Install Docker and Docker Compose")
        print("2. Run: docker-compose -f deploy/docker/docker-compose.development.yml up -d")
        print("3. Access: http://localhost:3000")
        return
    
    if '--manual' in sys.argv:
        print("⚙️ Manual Setup Instructions:")
        print("Please follow the detailed guide in SETUP_GUIDE.md")
        return
    
    # Show options and get user choice
    show_setup_options(system_info)
    
    choice = input("Choose setup method (1-5) or 'r' for recommended: ").strip().lower()
    
    if choice == 'r' or choice == '':
        run_recommended_setup(system_info)
    elif choice == '1' and system_info['os'] == 'Windows':
        os.system('setup.bat')
    elif choice == '2' and system_info['os'] in ['Darwin', 'Linux']:
        os.chmod('setup.sh', 0o755)
        os.system('./setup.sh')
    elif choice == '3' and system_info['has_gui']:
        os.system('python setup_wizard.py')
    elif choice == '4':
        print("🐳 Docker setup selected. Please check SETUP_GUIDE.md for Docker instructions.")
    elif choice == '5':
        print("⚙️ Manual setup selected. Please follow SETUP_GUIDE.md")
        if os.path.exists('SETUP_GUIDE.md'):
            try:
                webbrowser.open(f'file://{os.path.abspath("SETUP_GUIDE.md")}')
            except Exception:
                print("Please open SETUP_GUIDE.md manually")
    else:
        print("❌ Invalid choice. Please run again and select a valid option.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try manual setup or check SETUP_GUIDE.md")
        sys.exit(1)