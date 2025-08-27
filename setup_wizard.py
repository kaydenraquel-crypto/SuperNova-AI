#!/usr/bin/env python3
"""
SuperNova AI Setup Wizard
Cross-platform GUI installer for the SuperNova AI Financial Intelligence Platform
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import json
import webbrowser
from pathlib import Path

class SuperNovaSetupWizard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SuperNova AI - Setup Wizard")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Setup variables
        self.current_step = 0
        self.setup_progress = 0
        self.python_path = ""
        self.node_path = ""
        self.git_path = ""
        self.install_location = os.getcwd()
        
        # Environment configuration
        self.env_config = {
            'OPENAI_API_KEY': '',
            'ANTHROPIC_API_KEY': '',
            'DATABASE_URL': 'sqlite:///supernova.db',
            'REDIS_URL': 'redis://localhost:6379',
            'JWT_SECRET_KEY': '',
            'DEBUG': 'true'
        }
        
        self.setup_ui()
        self.check_prerequisites()
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="SuperNova AI Setup Wizard", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 5))
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Enterprise Financial Intelligence Platform",
                                  font=('Arial', 10))
        subtitle_label.grid(row=1, column=0)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Notebook for different setup steps
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Setup pages
        self.create_welcome_page()
        self.create_prerequisites_page()
        self.create_configuration_page()
        self.create_installation_page()
        self.create_completion_page()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.back_button = ttk.Button(button_frame, text="Back", command=self.go_back)
        self.back_button.grid(row=0, column=0, padx=(0, 5))
        
        self.next_button = ttk.Button(button_frame, text="Next", command=self.go_next)
        self.next_button.grid(row=0, column=1, padx=(0, 5))
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_setup)
        self.cancel_button.grid(row=0, column=2, padx=(0, 5))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to begin setup")
        status_label = ttk.Label(button_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=3, padx=(20, 0), sticky=tk.W)
        
        # Initially disable back button
        self.back_button.config(state='disabled')
        
    def create_welcome_page(self):
        """Create welcome page"""
        welcome_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(welcome_frame, text="Welcome")
        
        welcome_text = """
Welcome to SuperNova AI!

This setup wizard will help you install and configure the SuperNova AI
Financial Intelligence Platform on your system.

SuperNova AI Features:
• Advanced Financial Analytics (20+ metrics, VaR/CVaR analysis)
• Real-Time Collaboration (team management, live portfolio sharing)
• Enterprise API Management (rate limiting, usage analytics)
• AI-Powered Insights (multi-provider LLM integration)
• Professional Material-UI Interface
• Comprehensive Security Framework

System Requirements:
• Python 3.11+ (recommended) or 3.8+
• Node.js 18+ (recommended) or 16+
• 4GB RAM minimum, 8GB recommended
• 2GB disk space

Click Next to begin the installation process.
        """
        
        text_widget = tk.Text(welcome_frame, wrap=tk.WORD, height=20, width=70)
        text_widget.insert('1.0', welcome_text.strip())
        text_widget.config(state='disabled')
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        welcome_frame.columnconfigure(0, weight=1)
        welcome_frame.rowconfigure(0, weight=1)
        
    def create_prerequisites_page(self):
        """Create prerequisites check page"""
        self.prereq_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.prereq_frame, text="Prerequisites")
        
        ttk.Label(self.prereq_frame, text="Checking System Prerequisites...", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Python check
        ttk.Label(self.prereq_frame, text="Python 3.8+:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.python_status = ttk.Label(self.prereq_frame, text="Checking...")
        self.python_status.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Node.js check
        ttk.Label(self.prereq_frame, text="Node.js 16+:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.node_status = ttk.Label(self.prereq_frame, text="Checking...")
        self.node_status.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Git check
        ttk.Label(self.prereq_frame, text="Git:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.git_status = ttk.Label(self.prereq_frame, text="Checking...")
        self.git_status.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Install location
        ttk.Label(self.prereq_frame, text="Install Location:").grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        location_frame = ttk.Frame(self.prereq_frame)
        location_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(20, 5))
        
        self.location_var = tk.StringVar(value=self.install_location)
        location_entry = ttk.Entry(location_frame, textvariable=self.location_var, width=50)
        location_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        browse_button = ttk.Button(location_frame, text="Browse", command=self.browse_location)
        browse_button.grid(row=0, column=1, padx=(5, 0))
        
        location_frame.columnconfigure(0, weight=1)
        
        # Recheck button
        recheck_button = ttk.Button(self.prereq_frame, text="Recheck Prerequisites", 
                                   command=self.check_prerequisites)
        recheck_button.grid(row=5, column=0, columnspan=2, pady=(20, 0))
        
    def create_configuration_page(self):
        """Create configuration page"""
        config_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(config_frame, text="Configuration")
        
        ttk.Label(config_frame, text="Environment Configuration", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        ttk.Label(config_frame, text="Configure your API keys and settings. You can skip this and configure later in the .env file.", 
                 wraplength=600).grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Configuration entries
        self.config_vars = {}
        row = 2
        
        for key, default_value in self.env_config.items():
            ttk.Label(config_frame, text=f"{key}:").grid(row=row, column=0, sticky=tk.W, pady=5)
            
            var = tk.StringVar(value=default_value)
            self.config_vars[key] = var
            
            if 'KEY' in key or 'SECRET' in key:
                entry = ttk.Entry(config_frame, textvariable=var, show="*", width=60)
            else:
                entry = ttk.Entry(config_frame, textvariable=var, width=60)
            
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
            row += 1
        
        config_frame.columnconfigure(1, weight=1)
        
        # Help text
        help_text = """
Configuration Help:
• OPENAI_API_KEY: Get from https://platform.openai.com/api-keys
• ANTHROPIC_API_KEY: Get from https://console.anthropic.com/
• DATABASE_URL: Leave default for SQLite (recommended for local development)
• REDIS_URL: Leave default (requires Redis server)
• JWT_SECRET_KEY: Leave empty to generate automatically
• DEBUG: Set to 'false' for production
        """
        
        help_label = ttk.Label(config_frame, text=help_text.strip(), 
                              justify=tk.LEFT, wraplength=600)
        help_label.grid(row=row, column=0, columnspan=2, pady=(20, 0), sticky=tk.W)
        
    def create_installation_page(self):
        """Create installation progress page"""
        install_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(install_frame, text="Installation")
        
        ttk.Label(install_frame, text="Installation Progress", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 20))
        
        # Installation log
        self.install_log = scrolledtext.ScrolledText(install_frame, height=20, width=80)
        self.install_log.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        install_frame.columnconfigure(0, weight=1)
        install_frame.rowconfigure(1, weight=1)
        
        # Install button
        self.install_button = ttk.Button(install_frame, text="Start Installation", 
                                        command=self.start_installation)
        self.install_button.grid(row=2, column=0, pady=(10, 0))
        
    def create_completion_page(self):
        """Create completion page"""
        complete_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(complete_frame, text="Complete")
        
        ttk.Label(complete_frame, text="Installation Complete!", 
                 font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=(0, 20))
        
        success_text = """
SuperNova AI has been successfully installed!

Next Steps:
1. Review your configuration in the .env file
2. Start the application using the provided scripts
3. Access the web interface at http://localhost:3000
4. View API documentation at http://localhost:8081/docs

Startup Commands:
• Windows: Double-click start_supernova.bat
• macOS/Linux: Run ./start_supernova.sh

Need Help?
• Documentation: Check the docs/ folder
• Issues: Report at GitHub repository
• Community: Join our discussions

Thank you for choosing SuperNova AI!
        """
        
        success_label = ttk.Label(complete_frame, text=success_text.strip(), 
                                 justify=tk.LEFT, wraplength=600)
        success_label.grid(row=1, column=0, pady=(0, 20), sticky=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(complete_frame)
        button_frame.grid(row=2, column=0, pady=(20, 0))
        
        ttk.Button(button_frame, text="Open Documentation", 
                  command=self.open_docs).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame, text="Start SuperNova AI", 
                  command=self.start_supernova).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(button_frame, text="Open Web Interface", 
                  command=self.open_web).grid(row=0, column=2)
        
    def check_prerequisites(self):
        """Check system prerequisites"""
        def check_in_thread():
            # Check Python
            try:
                result = subprocess.run([sys.executable, '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    self.python_path = sys.executable
                    self.root.after(0, lambda: self.python_status.config(text=f"✓ {version}", foreground="green"))
                else:
                    self.root.after(0, lambda: self.python_status.config(text="✗ Not found", foreground="red"))
            except Exception as e:
                self.root.after(0, lambda: self.python_status.config(text="✗ Error checking", foreground="red"))
            
            # Check Node.js
            try:
                result = subprocess.run(['node', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    self.node_path = 'node'
                    self.root.after(0, lambda: self.node_status.config(text=f"✓ {version}", foreground="green"))
                else:
                    self.root.after(0, lambda: self.node_status.config(text="✗ Not found", foreground="red"))
            except Exception as e:
                self.root.after(0, lambda: self.node_status.config(text="✗ Error checking", foreground="red"))
            
            # Check Git
            try:
                result = subprocess.run(['git', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip().split()[-1]
                    self.git_path = 'git'
                    self.root.after(0, lambda: self.git_status.config(text=f"✓ {version}", foreground="green"))
                else:
                    self.root.after(0, lambda: self.git_status.config(text="✗ Not found", foreground="orange"))
            except Exception as e:
                self.root.after(0, lambda: self.git_status.config(text="✗ Not available", foreground="orange"))
        
        # Run checks in background thread
        threading.Thread(target=check_in_thread, daemon=True).start()
    
    def browse_location(self):
        """Browse for installation location"""
        directory = filedialog.askdirectory(initialdir=self.install_location)
        if directory:
            self.install_location = directory
            self.location_var.set(directory)
    
    def start_installation(self):
        """Start the installation process"""
        self.install_button.config(state='disabled')
        self.next_button.config(state='disabled')
        
        def install_in_thread():
            try:
                self.log_message("Starting SuperNova AI installation...")
                self.update_progress(5)
                
                # Change to install directory
                os.chdir(self.install_location)
                
                # Create .env file
                self.log_message("Creating environment configuration...")
                self.create_env_file()
                self.update_progress(15)
                
                # Create virtual environment
                self.log_message("Creating Python virtual environment...")
                result = subprocess.run([sys.executable, '-m', 'venv', '.venv'], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    self.log_message(f"Error creating virtual environment: {result.stderr}")
                    return
                self.update_progress(25)
                
                # Activate virtual environment and install Python packages
                self.log_message("Installing Python dependencies...")
                if sys.platform == 'win32':
                    python_exe = os.path.join('.venv', 'Scripts', 'python.exe')
                    pip_exe = os.path.join('.venv', 'Scripts', 'pip.exe')
                else:
                    python_exe = os.path.join('.venv', 'bin', 'python')
                    pip_exe = os.path.join('.venv', 'bin', 'pip')
                
                # Upgrade pip
                result = subprocess.run([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                      capture_output=True, text=True)
                self.update_progress(35)
                
                # Install requirements
                if os.path.exists('requirements.txt'):
                    result = subprocess.run([pip_exe, 'install', '-r', 'requirements.txt'], 
                                          capture_output=True, text=True)
                    if result.returncode != 0:
                        self.log_message(f"Error installing Python packages: {result.stderr}")
                        return
                else:
                    self.log_message("Warning: requirements.txt not found")
                
                self.update_progress(60)
                
                # Install Node.js dependencies
                self.log_message("Installing Node.js dependencies...")
                if os.path.exists('frontend'):
                    os.chdir('frontend')
                    result = subprocess.run(['npm', 'install'], 
                                          capture_output=True, text=True)
                    if result.returncode != 0:
                        self.log_message(f"Error installing Node.js packages: {result.stderr}")
                        return
                    os.chdir('..')
                else:
                    self.log_message("Warning: frontend directory not found")
                
                self.update_progress(80)
                
                # Initialize database
                self.log_message("Initializing database...")
                try:
                    result = subprocess.run([python_exe, '-c', 
                                           'from supernova.db import init_db; init_db()'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log_message("Database initialized successfully")
                    else:
                        self.log_message(f"Database initialization warning: {result.stderr}")
                except Exception as e:
                    self.log_message(f"Database initialization skipped: {str(e)}")
                
                self.update_progress(95)
                
                # Create startup scripts (handled by main setup scripts)
                self.log_message("Setup completed successfully!")
                self.update_progress(100)
                
                # Enable next button
                self.root.after(0, lambda: self.next_button.config(state='normal'))
                self.root.after(0, lambda: self.status_var.set("Installation completed!"))
                
            except Exception as e:
                self.log_message(f"Installation failed: {str(e)}")
                self.root.after(0, lambda: self.install_button.config(state='normal'))
        
        # Run installation in background thread
        threading.Thread(target=install_in_thread, daemon=True).start()
    
    def create_env_file(self):
        """Create .env file with configuration"""
        env_content = []
        for key, var in self.config_vars.items():
            value = var.get()
            if key == 'JWT_SECRET_KEY' and not value:
                # Generate a random secret key
                import secrets
                value = secrets.token_urlsafe(32)
            env_content.append(f"{key}={value}")
        
        with open('.env', 'w') as f:
            f.write('\n'.join(env_content))
    
    def log_message(self, message):
        """Add message to installation log"""
        def add_log():
            self.install_log.insert(tk.END, f"{message}\n")
            self.install_log.see(tk.END)
        
        self.root.after(0, add_log)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def go_next(self):
        """Go to next page"""
        current = self.notebook.index(self.notebook.select())
        if current < self.notebook.index('end') - 1:
            self.notebook.select(current + 1)
            self.back_button.config(state='normal')
            
            # Update button text for last page
            if current + 1 == self.notebook.index('end') - 1:
                self.next_button.config(text='Finish')
            else:
                self.next_button.config(text='Next')
        else:
            # Finish setup
            self.root.destroy()
    
    def go_back(self):
        """Go to previous page"""
        current = self.notebook.index(self.notebook.select())
        if current > 0:
            self.notebook.select(current - 1)
            self.next_button.config(text='Next')
            
            if current - 1 == 0:
                self.back_button.config(state='disabled')
    
    def cancel_setup(self):
        """Cancel setup"""
        if messagebox.askokcancel("Cancel Setup", "Are you sure you want to cancel the setup?"):
            self.root.destroy()
    
    def open_docs(self):
        """Open documentation"""
        docs_path = os.path.join(self.install_location, 'docs', 'README.md')
        if os.path.exists(docs_path):
            webbrowser.open(f'file://{os.path.abspath(docs_path)}')
        else:
            messagebox.showinfo("Documentation", "Documentation not found. Check the docs/ folder.")
    
    def start_supernova(self):
        """Start SuperNova AI"""
        if sys.platform == 'win32':
            script_path = os.path.join(self.install_location, 'start_supernova.bat')
            if os.path.exists(script_path):
                subprocess.Popen(script_path, shell=True)
            else:
                messagebox.showerror("Error", "Startup script not found")
        else:
            script_path = os.path.join(self.install_location, 'start_supernova.sh')
            if os.path.exists(script_path):
                subprocess.Popen(['bash', script_path])
            else:
                messagebox.showerror("Error", "Startup script not found")
    
    def open_web(self):
        """Open web interface"""
        webbrowser.open('http://localhost:3000')
    
    def run(self):
        """Run the setup wizard"""
        self.root.mainloop()

if __name__ == '__main__':
    # Check if tkinter is available
    try:
        import tkinter
    except ImportError:
        print("Error: tkinter not available. Please install tkinter:")
        print("  - Ubuntu/Debian: sudo apt install python3-tk")
        print("  - CentOS/RHEL: sudo yum install tkinter")
        print("  - macOS: tkinter should be included with Python")
        print("  - Windows: tkinter should be included with Python")
        sys.exit(1)
    
    wizard = SuperNovaSetupWizard()
    wizard.run()