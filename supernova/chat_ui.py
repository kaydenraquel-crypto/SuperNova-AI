"""
Chat UI Components for SuperNova Financial Advisor
Professional chat interface with real-time communication, chart embedding, and mobile support
"""

from typing import Optional, Dict, Any, List
from fastapi import Request
from fastapi.responses import HTMLResponse
import json

def get_chat_interface_html(
    session_id: Optional[str] = None,
    profile_id: Optional[int] = None,
    theme: str = "dark",
    api_base_url: str = "/api",
    websocket_url: str = "/ws/chat"
) -> str:
    """
    Generate complete HTML chat interface
    
    Args:
        session_id: Existing chat session ID
        profile_id: User profile ID for personalization
        theme: UI theme (dark/light)
        api_base_url: Base URL for API endpoints
        websocket_url: WebSocket endpoint URL
    
    Returns:
        Complete HTML string for chat interface
    """
    
    return f'''
<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperNova Financial Advisor - Chat</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-dark.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    
    <style>
        {get_chat_styles(theme)}
    </style>
</head>
<body>
    <div id="chat-container">
        <!-- Header -->
        <header class="chat-header">
            <div class="header-left">
                <div class="logo">
                    <i class="fas fa-chart-line"></i>
                    <span>SuperNova</span>
                </div>
                <div class="connection-status" id="connection-status">
                    <i class="fas fa-circle"></i>
                    <span>Connected</span>
                </div>
            </div>
            
            <div class="header-actions">
                <button class="action-btn" id="theme-toggle" title="Toggle Theme">
                    <i class="fas fa-moon"></i>
                </button>
                <button class="action-btn" id="settings-btn" title="Settings">
                    <i class="fas fa-cog"></i>
                </button>
                <button class="action-btn" id="export-btn" title="Export Chat">
                    <i class="fas fa-download"></i>
                </button>
            </div>
        </header>

        <!-- Main Chat Area -->
        <div class="chat-main">
            <!-- Sidebar -->
            <aside class="chat-sidebar" id="chat-sidebar">
                <div class="sidebar-header">
                    <h3>Chat Sessions</h3>
                    <button class="new-session-btn" id="new-session-btn">
                        <i class="fas fa-plus"></i>
                        New Chat
                    </button>
                </div>
                
                <div class="session-list" id="session-list">
                    <!-- Session items will be loaded here -->
                </div>
                
                <div class="sidebar-footer">
                    <div class="user-profile">
                        <div class="profile-avatar">
                            <i class="fas fa-user"></i>
                        </div>
                        <div class="profile-info">
                            <span class="profile-name">Demo User</span>
                            <span class="profile-status">Online</span>
                        </div>
                    </div>
                </div>
            </aside>

            <!-- Chat Content -->
            <main class="chat-content">
                <!-- Messages Area -->
                <div class="messages-container" id="messages-container">
                    <div class="welcome-message" id="welcome-message">
                        <div class="welcome-content">
                            <i class="fas fa-robot"></i>
                            <h2>Welcome to SuperNova Financial Advisor</h2>
                            <p>I'm your AI financial advisor, ready to help you with:</p>
                            <ul>
                                <li><i class="fas fa-chart-bar"></i> Market analysis and stock research</li>
                                <li><i class="fas fa-coins"></i> Investment recommendations</li>
                                <li><i class="fas fa-calculator"></i> Portfolio optimization</li>
                                <li><i class="fas fa-history"></i> Strategy backtesting</li>
                            </ul>
                            <p>How can I help you today?</p>
                        </div>
                    </div>
                    
                    <div class="messages-list" id="messages-list">
                        <!-- Messages will be loaded here -->
                    </div>
                    
                    <div class="typing-indicator" id="typing-indicator" style="display: none;">
                        <div class="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        <span class="typing-text">AI is thinking...</span>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div class="quick-actions" id="quick-actions">
                    <button class="quick-action-btn" data-action="market-overview">
                        <i class="fas fa-chart-line"></i>
                        Market Overview
                    </button>
                    <button class="quick-action-btn" data-action="portfolio-analysis">
                        <i class="fas fa-pie-chart"></i>
                        Portfolio Analysis
                    </button>
                    <button class="quick-action-btn" data-action="stock-analysis">
                        <i class="fas fa-search"></i>
                        Stock Analysis
                    </button>
                    <button class="quick-action-btn" data-action="strategy-backtest">
                        <i class="fas fa-history"></i>
                        Backtest Strategy
                    </button>
                </div>

                <!-- Input Area -->
                <div class="input-container">
                    <div class="input-wrapper">
                        <div class="file-upload" id="file-upload">
                            <input type="file" id="file-input" accept=".csv,.json,.png,.jpg,.jpeg" multiple>
                            <button class="upload-btn" id="upload-btn" title="Upload File">
                                <i class="fas fa-paperclip"></i>
                            </button>
                        </div>
                        
                        <textarea 
                            id="message-input" 
                            placeholder="Ask me anything about finance, markets, or your portfolio..." 
                            rows="1"
                        ></textarea>
                        
                        <div class="input-actions">
                            <button class="voice-btn" id="voice-btn" title="Voice Message">
                                <i class="fas fa-microphone"></i>
                            </button>
                            <button class="send-btn" id="send-btn" title="Send Message">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Suggestions -->
                    <div class="suggestions-container" id="suggestions-container" style="display: none;">
                        <div class="suggestions-list" id="suggestions-list">
                            <!-- Suggestions will be loaded here -->
                        </div>
                    </div>
                </div>
            </main>
        </div>

        <!-- Chart Modal -->
        <div class="modal" id="chart-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="chart-title">Chart</h3>
                    <button class="close-btn" id="close-chart-modal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div id="chart-container" style="width: 100%; height: 400px;"></div>
                </div>
            </div>
        </div>

        <!-- Settings Modal -->
        <div class="modal" id="settings-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Settings</h3>
                    <button class="close-btn" id="close-settings-modal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="settings-section">
                        <h4>Appearance</h4>
                        <div class="setting-item">
                            <label>Theme</label>
                            <select id="theme-select">
                                <option value="dark" {"selected" if theme == "dark" else ""}>Dark</option>
                                <option value="light" {"selected" if theme == "light" else ""}>Light</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="settings-section">
                        <h4>Notifications</h4>
                        <div class="setting-item">
                            <label>
                                <input type="checkbox" id="sound-notifications" checked>
                                Sound notifications
                            </label>
                        </div>
                        <div class="setting-item">
                            <label>
                                <input type="checkbox" id="desktop-notifications" checked>
                                Desktop notifications
                            </label>
                        </div>
                    </div>
                    
                    <div class="settings-section">
                        <h4>Chat</h4>
                        <div class="setting-item">
                            <label>
                                <input type="checkbox" id="auto-suggestions" checked>
                                Show auto-suggestions
                            </label>
                        </div>
                        <div class="setting-item">
                            <label>
                                <input type="checkbox" id="typing-indicators" checked>
                                Show typing indicators
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        {get_chat_javascript(session_id, profile_id, api_base_url, websocket_url)}
    </script>
</body>
</html>
    '''

def get_chat_styles(theme: str = "dark") -> str:
    """Generate CSS styles for chat interface"""
    
    # Color schemes
    colors = {
        "dark": {
            "bg-primary": "#1a1a1a",
            "bg-secondary": "#2d2d2d", 
            "bg-tertiary": "#404040",
            "bg-hover": "#525252",
            "text-primary": "#ffffff",
            "text-secondary": "#cccccc",
            "text-muted": "#999999",
            "border": "#404040",
            "accent": "#00d4aa",
            "accent-hover": "#00b894",
            "danger": "#e74c3c",
            "warning": "#f39c12",
            "success": "#00d4aa",
            "user-message": "#007bff",
            "ai-message": "#28a745"
        },
        "light": {
            "bg-primary": "#ffffff",
            "bg-secondary": "#f8f9fa",
            "bg-tertiary": "#e9ecef", 
            "bg-hover": "#dee2e6",
            "text-primary": "#212529",
            "text-secondary": "#495057",
            "text-muted": "#6c757d",
            "border": "#dee2e6",
            "accent": "#007bff",
            "accent-hover": "#0056b3",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "success": "#28a745",
            "user-message": "#007bff",
            "ai-message": "#28a745"
        }
    }
    
    current_colors = colors.get(theme, colors["dark"])
    
    return f'''
        :root {{
            --bg-primary: {current_colors["bg-primary"]};
            --bg-secondary: {current_colors["bg-secondary"]};
            --bg-tertiary: {current_colors["bg-tertiary"]};
            --bg-hover: {current_colors["bg-hover"]};
            --text-primary: {current_colors["text-primary"]};
            --text-secondary: {current_colors["text-secondary"]};
            --text-muted: {current_colors["text-muted"]};
            --border: {current_colors["border"]};
            --accent: {current_colors["accent"]};
            --accent-hover: {current_colors["accent-hover"]};
            --danger: {current_colors["danger"]};
            --warning: {current_colors["warning"]};
            --success: {current_colors["success"]};
            --user-message: {current_colors["user-message"]};
            --ai-message: {current_colors["ai-message"]};
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
        }}

        #chat-container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        /* Header */
        .chat-header {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 2rem;
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: bold;
            font-size: 1.2rem;
        }}

        .logo i {{
            color: var(--accent);
            font-size: 1.4rem;
        }}

        .connection-status {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}

        .connection-status.connected i {{
            color: var(--success);
        }}

        .connection-status.disconnected i {{
            color: var(--danger);
        }}

        .header-actions {{
            display: flex;
            gap: 0.5rem;
        }}

        .action-btn {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .action-btn:hover {{
            background: var(--bg-hover);
            border-color: var(--accent);
        }}

        /* Main Chat Area */
        .chat-main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        /* Sidebar */
        .chat-sidebar {{
            width: 300px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease;
        }}

        .sidebar-header {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .sidebar-header h3 {{
            margin-bottom: 1rem;
        }}

        .new-session-btn {{
            width: 100%;
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 0.5rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: background 0.2s;
        }}

        .new-session-btn:hover {{
            background: var(--accent-hover);
        }}

        .session-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }}

        .session-item {{
            padding: 0.75rem;
            border-radius: 0.5rem;
            cursor: pointer;
            margin-bottom: 0.5rem;
            transition: background 0.2s;
        }}

        .session-item:hover {{
            background: var(--bg-hover);
        }}

        .session-item.active {{
            background: var(--accent);
            color: white;
        }}

        .session-preview {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .sidebar-footer {{
            padding: 1rem;
            border-top: 1px solid var(--border);
        }}

        .user-profile {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .profile-avatar {{
            width: 40px;
            height: 40px;
            background: var(--accent);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }}

        .profile-info {{
            display: flex;
            flex-direction: column;
        }}

        .profile-name {{
            font-weight: 500;
        }}

        .profile-status {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        /* Chat Content */
        .chat-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .messages-container {{
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }}

        .welcome-message {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex: 1;
            text-align: center;
        }}

        .welcome-content {{
            max-width: 600px;
            padding: 2rem;
        }}

        .welcome-content i {{
            font-size: 4rem;
            color: var(--accent);
            margin-bottom: 1rem;
        }}

        .welcome-content h2 {{
            margin-bottom: 1rem;
        }}

        .welcome-content ul {{
            list-style: none;
            margin: 1.5rem 0;
        }}

        .welcome-content li {{
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 0.5rem;
        }}

        .welcome-content li i {{
            font-size: 1rem;
            color: var(--accent);
            width: 20px;
        }}

        .messages-list {{
            display: none;
        }}

        .messages-list.active {{
            display: block;
        }}

        .message {{
            margin-bottom: 1rem;
            display: flex;
            gap: 0.75rem;
        }}

        .message.user {{
            flex-direction: row-reverse;
        }}

        .message-avatar {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            color: white;
            flex-shrink: 0;
        }}

        .message.user .message-avatar {{
            background: var(--user-message);
        }}

        .message.ai .message-avatar {{
            background: var(--ai-message);
        }}

        .message-content {{
            max-width: 70%;
            background: var(--bg-secondary);
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            position: relative;
        }}

        .message.user .message-content {{
            background: var(--user-message);
            color: white;
        }}

        .message-text {{
            line-height: 1.5;
        }}

        .message-metadata {{
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .message-actions {{
            display: flex;
            gap: 0.5rem;
        }}

        .message-action {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 0.25rem;
            border-radius: 0.25rem;
            transition: all 0.2s;
        }}

        .message-action:hover {{
            color: var(--accent);
            background: var(--bg-hover);
        }}

        .chart-embed {{
            margin: 0.5rem 0;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            overflow: hidden;
            cursor: pointer;
            transition: border-color 0.2s;
        }}

        .chart-embed:hover {{
            border-color: var(--accent);
        }}

        .chart-placeholder {{
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .typing-indicator {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1rem 0;
        }}

        .typing-dots {{
            display: flex;
            gap: 0.25rem;
        }}

        .typing-dots span {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
            animation: typing 1.4s infinite ease-in-out;
        }}

        .typing-dots span:nth-child(1) {{ animation-delay: -0.32s; }}
        .typing-dots span:nth-child(2) {{ animation-delay: -0.16s; }}

        @keyframes typing {{
            0%, 80%, 100% {{
                transform: scale(0);
                opacity: 0.5;
            }}
            40% {{
                transform: scale(1);
                opacity: 1;
            }}
        }}

        /* Quick Actions */
        .quick-actions {{
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            border-top: 1px solid var(--border);
            overflow-x: auto;
        }}

        .quick-action-btn {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            cursor: pointer;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }}

        .quick-action-btn:hover {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}

        /* Input Area */
        .input-container {{
            border-top: 1px solid var(--border);
            background: var(--bg-secondary);
        }}

        .input-wrapper {{
            display: flex;
            align-items: flex-end;
            gap: 0.5rem;
            padding: 1rem;
        }}

        .file-upload {{
            display: flex;
            align-items: center;
        }}

        #file-input {{
            display: none;
        }}

        .upload-btn {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.75rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .upload-btn:hover {{
            color: var(--accent);
            border-color: var(--accent);
        }}

        #message-input {{
            flex: 1;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            resize: none;
            min-height: 50px;
            max-height: 120px;
            line-height: 1.5;
            font-family: inherit;
        }}

        #message-input:focus {{
            outline: none;
            border-color: var(--accent);
        }}

        .input-actions {{
            display: flex;
            gap: 0.5rem;
        }}

        .voice-btn, .send-btn {{
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}

        .voice-btn {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .voice-btn:hover {{
            background: var(--accent);
            color: white;
        }}

        .voice-btn.recording {{
            background: var(--danger);
            color: white;
            animation: pulse 1s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}

        .send-btn {{
            background: var(--accent);
            color: white;
        }}

        .send-btn:hover {{
            background: var(--accent-hover);
        }}

        .send-btn:disabled {{
            background: var(--bg-tertiary);
            color: var(--text-muted);
            cursor: not-allowed;
        }}

        /* Suggestions */
        .suggestions-container {{
            padding: 0 1rem 1rem;
        }}

        .suggestions-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .suggestion-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.5rem 0.75rem;
            border-radius: 1rem;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}

        .suggestion-btn:hover {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}

        /* Modals */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
        }}

        .modal.show {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .modal-content {{
            background: var(--bg-primary);
            border-radius: 0.5rem;
            border: 1px solid var(--border);
            width: 90%;
            max-width: 800px;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .modal-header {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .modal-body {{
            padding: 1rem;
            overflow-y: auto;
            flex: 1;
        }}

        .close-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 0.25rem;
            transition: all 0.2s;
        }}

        .close-btn:hover {{
            color: var(--danger);
            background: var(--bg-hover);
        }}

        /* Settings */
        .settings-section {{
            margin-bottom: 2rem;
        }}

        .settings-section h4 {{
            margin-bottom: 1rem;
            color: var(--accent);
        }}

        .setting-item {{
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .setting-item label {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .setting-item select {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem;
            border-radius: 0.25rem;
        }}

        .setting-item input[type="checkbox"] {{
            accent-color: var(--accent);
        }}

        /* Mobile Responsive */
        @media (max-width: 768px) {{
            .chat-sidebar {{
                position: fixed;
                left: 0;
                top: 0;
                height: 100vh;
                z-index: 200;
                transform: translateX(-100%);
            }}

            .chat-sidebar.open {{
                transform: translateX(0);
            }}

            .chat-content {{
                width: 100%;
            }}

            .header-left .logo span {{
                display: none;
            }}

            .quick-actions {{
                padding: 0.5rem;
            }}

            .quick-action-btn {{
                font-size: 0.8rem;
                padding: 0.4rem 0.8rem;
            }}

            .message-content {{
                max-width: 85%;
            }}

            .input-wrapper {{
                padding: 0.75rem;
            }}

            .modal-content {{
                width: 95%;
                margin: 1rem;
            }}
        }}

        /* Accessibility */
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}

        /* Focus styles for accessibility */
        button:focus, input:focus, textarea:focus, select:focus {{
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }}

        /* High contrast mode support */
        @media (prefers-contrast: high) {{
            :root {{
                --border: #ffffff;
                --bg-hover: #ffffff20;
            }}
        }}
    '''

def get_chat_javascript(
    session_id: Optional[str] = None,
    profile_id: Optional[int] = None,
    api_base_url: str = "/api",
    websocket_url: str = "/ws/chat"
) -> str:
    """Generate JavaScript code for chat functionality"""
    
    return f'''
        class SuperNovaChatClient {{
            constructor() {{
                this.sessionId = '{session_id or ""}';
                this.profileId = {profile_id or "null"};
                this.apiBaseUrl = '{api_base_url}';
                this.websocketUrl = '{websocket_url}';
                this.websocket = null;
                this.isConnected = false;
                this.currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
                this.messageHistory = [];
                this.suggestions = [];
                this.isRecording = false;
                this.mediaRecorder = null;
                
                this.initializeElements();
                this.setupEventListeners();
                this.initializeChat();
                this.loadSessions();
            }}
            
            initializeElements() {{
                this.elements = {{
                    // Header elements
                    connectionStatus: document.getElementById('connection-status'),
                    themeToggle: document.getElementById('theme-toggle'),
                    settingsBtn: document.getElementById('settings-btn'),
                    exportBtn: document.getElementById('export-btn'),
                    
                    // Sidebar elements
                    chatSidebar: document.getElementById('chat-sidebar'),
                    newSessionBtn: document.getElementById('new-session-btn'),
                    sessionList: document.getElementById('session-list'),
                    
                    // Chat elements
                    messagesContainer: document.getElementById('messages-container'),
                    welcomeMessage: document.getElementById('welcome-message'),
                    messagesList: document.getElementById('messages-list'),
                    typingIndicator: document.getElementById('typing-indicator'),
                    
                    // Quick actions
                    quickActions: document.getElementById('quick-actions'),
                    
                    // Input elements
                    messageInput: document.getElementById('message-input'),
                    fileInput: document.getElementById('file-input'),
                    uploadBtn: document.getElementById('upload-btn'),
                    voiceBtn: document.getElementById('voice-btn'),
                    sendBtn: document.getElementById('send-btn'),
                    suggestionsContainer: document.getElementById('suggestions-container'),
                    suggestionsList: document.getElementById('suggestions-list'),
                    
                    // Modals
                    chartModal: document.getElementById('chart-modal'),
                    settingsModal: document.getElementById('settings-modal'),
                    closeChartModal: document.getElementById('close-chart-modal'),
                    closeSettingsModal: document.getElementById('close-settings-modal'),
                    chartContainer: document.getElementById('chart-container'),
                    chartTitle: document.getElementById('chart-title'),
                    
                    // Settings
                    themeSelect: document.getElementById('theme-select')
                }};
            }}
            
            setupEventListeners() {{
                // Header actions
                this.elements.themeToggle.addEventListener('click', () => this.toggleTheme());
                this.elements.settingsBtn.addEventListener('click', () => this.showSettingsModal());
                this.elements.exportBtn.addEventListener('click', () => this.exportChat());
                
                // Sidebar
                this.elements.newSessionBtn.addEventListener('click', () => this.createNewSession());
                
                // Input handling
                this.elements.messageInput.addEventListener('keydown', (e) => {{
                    if (e.key === 'Enter' && !e.shiftKey) {{
                        e.preventDefault();
                        this.sendMessage();
                    }}
                }});
                
                this.elements.messageInput.addEventListener('input', () => {{
                    this.handleInputChange();
                    this.sendTypingIndicator(true);
                }});
                
                // Auto-resize textarea
                this.elements.messageInput.addEventListener('input', () => {{
                    this.elements.messageInput.style.height = 'auto';
                    this.elements.messageInput.style.height = Math.min(this.elements.messageInput.scrollHeight, 120) + 'px';
                }});
                
                // Buttons
                this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
                this.elements.voiceBtn.addEventListener('click', () => this.toggleVoiceRecording());
                this.elements.uploadBtn.addEventListener('click', () => this.elements.fileInput.click());
                this.elements.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
                
                // Quick actions
                this.elements.quickActions.addEventListener('click', (e) => {{
                    if (e.target.classList.contains('quick-action-btn')) {{
                        this.handleQuickAction(e.target.dataset.action);
                    }}
                }});
                
                // Modal handling
                this.elements.closeChartModal.addEventListener('click', () => this.hideChartModal());
                this.elements.closeSettingsModal.addEventListener('click', () => this.hideSettingsModal());
                this.elements.chartModal.addEventListener('click', (e) => {{
                    if (e.target === this.elements.chartModal) this.hideChartModal();
                }});
                this.elements.settingsModal.addEventListener('click', (e) => {{
                    if (e.target === this.elements.settingsModal) this.hideSettingsModal();
                }});
                
                // Settings
                this.elements.themeSelect.addEventListener('change', (e) => {{
                    this.setTheme(e.target.value);
                }});
                
                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {{
                    if (e.ctrlKey || e.metaKey) {{
                        switch(e.key) {{
                            case 'n':
                                e.preventDefault();
                                this.createNewSession();
                                break;
                            case 'e':
                                e.preventDefault();
                                this.exportChat();
                                break;
                            case ',':
                                e.preventDefault();
                                this.showSettingsModal();
                                break;
                        }}
                    }}
                }});
            }}
            
            async initializeChat() {{
                if (this.sessionId) {{
                    await this.loadChatHistory();
                    this.connectWebSocket();
                }} else {{
                    await this.createNewSession();
                }}
                
                this.loadSuggestions();
            }}
            
            async createNewSession() {{
                try {{
                    const response = await fetch(`${{this.apiBaseUrl}}/chat/session`, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer demo_token'  // Replace with actual token
                        }},
                        body: JSON.stringify({{
                            profile_id: this.profileId
                        }})
                    }});
                    
                    const data = await response.json();
                    this.sessionId = data.session_id;
                    
                    // Update URL without page reload
                    const url = new URL(window.location);
                    url.searchParams.set('session_id', this.sessionId);
                    window.history.pushState({{}}, '', url);
                    
                    // Clear chat and show welcome message
                    this.clearChat();
                    this.connectWebSocket();
                    this.loadSessions();
                    
                }} catch (error) {{
                    console.error('Error creating new session:', error);
                    this.showError('Failed to create new chat session');
                }}
            }}
            
            async loadChatHistory() {{
                try {{
                    const response = await fetch(`${{this.apiBaseUrl}}/chat/session/${{this.sessionId}}`, {{
                        headers: {{
                            'Authorization': 'Bearer demo_token'
                        }}
                    }});
                    
                    if (response.ok) {{
                        const data = await response.json();
                        this.displayMessages(data.messages);
                        this.hideWelcomeMessage();
                    }}
                }} catch (error) {{
                    console.error('Error loading chat history:', error);
                }}
            }}
            
            async loadSessions() {{
                try {{
                    const response = await fetch(`${{this.apiBaseUrl}}/chat/sessions`, {{
                        headers: {{
                            'Authorization': 'Bearer demo_token'
                        }}
                    }});
                    
                    const data = await response.json();
                    this.displaySessions(data.sessions);
                }} catch (error) {{
                    console.error('Error loading sessions:', error);
                }}
            }}
            
            displaySessions(sessions) {{
                this.elements.sessionList.innerHTML = '';
                
                sessions.forEach(session => {{
                    const sessionItem = document.createElement('div');
                    sessionItem.className = 'session-item';
                    if (session.session_id === this.sessionId) {{
                        sessionItem.classList.add('active');
                    }}
                    
                    sessionItem.innerHTML = `
                        <div class="session-title">${{new Date(session.created_at).toLocaleDateString()}}</div>
                        <div class="session-preview">${{session.preview}}</div>
                    `;
                    
                    sessionItem.addEventListener('click', () => {{
                        this.switchToSession(session.session_id);
                    }});
                    
                    this.elements.sessionList.appendChild(sessionItem);
                }});
            }}
            
            async switchToSession(sessionId) {{
                if (sessionId === this.sessionId) return;
                
                this.sessionId = sessionId;
                this.disconnectWebSocket();
                
                // Update URL
                const url = new URL(window.location);
                url.searchParams.set('session_id', sessionId);
                window.history.pushState({{}}, '', url);
                
                // Load new session
                await this.loadChatHistory();
                this.connectWebSocket();
                
                // Update active session in sidebar
                document.querySelectorAll('.session-item').forEach(item => {{
                    item.classList.remove('active');
                }});
                event.currentTarget.classList.add('active');
            }}
            
            connectWebSocket() {{
                if (this.websocket) {{
                    this.websocket.close();
                }}
                
                const wsUrl = `ws://localhost:8000${{this.websocketUrl}}/${{this.sessionId}}`;
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {{
                    this.isConnected = true;
                    this.updateConnectionStatus(true);
                    console.log('WebSocket connected');
                }};
                
                this.websocket.onclose = () => {{
                    this.isConnected = false;
                    this.updateConnectionStatus(false);
                    console.log('WebSocket disconnected');
                    
                    // Attempt to reconnect after 3 seconds
                    setTimeout(() => {{
                        if (!this.isConnected) {{
                            this.connectWebSocket();
                        }}
                    }}, 3000);
                }};
                
                this.websocket.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                }};
                
                this.websocket.onmessage = (event) => {{
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                }};
            }}
            
            disconnectWebSocket() {{
                if (this.websocket) {{
                    this.websocket.close();
                    this.websocket = null;
                }}
            }}
            
            handleWebSocketMessage(message) {{
                switch (message.type) {{
                    case 'message':
                        this.displayMessage(message.data);
                        this.hideTypingIndicator();
                        break;
                    case 'typing':
                        if (message.user_id !== 'demo_user') {{
                            this.showTypingIndicator();
                        }}
                        break;
                    case 'market_update':
                        this.handleMarketUpdate(message.data);
                        break;
                    case 'system':
                        this.handleSystemMessage(message.data);
                        break;
                }}
            }}
            
            async sendMessage(text = null) {{
                const message = text || this.elements.messageInput.value.trim();
                if (!message) return;
                
                // Clear input and hide suggestions
                this.elements.messageInput.value = '';
                this.elements.messageInput.style.height = 'auto';
                this.hideSuggestions();
                
                // Display user message immediately
                const userMessage = {{
                    id: Date.now().toString(),
                    role: 'user',
                    content: message,
                    timestamp: new Date().toISOString()
                }};
                
                this.displayMessage(userMessage);
                this.hideWelcomeMessage();
                this.showTypingIndicator();
                
                try {{
                    const response = await fetch(`${{this.apiBaseUrl}}/chat`, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer demo_token'
                        }},
                        body: JSON.stringify({{
                            message: message,
                            session_id: this.sessionId,
                            profile_id: this.profileId
                        }})
                    }});
                    
                    const data = await response.json();
                    
                    // Display AI response
                    this.displayMessage(data.message);
                    this.hideTypingIndicator();
                    
                    // Show suggestions if available
                    if (data.suggestions && data.suggestions.length > 0) {{
                        this.showSuggestions(data.suggestions);
                    }}
                    
                    // Display charts if available
                    if (data.charts && data.charts.length > 0) {{
                        data.charts.forEach(chart => this.displayChart(chart));
                    }}
                    
                }} catch (error) {{
                    console.error('Error sending message:', error);
                    this.hideTypingIndicator();
                    this.showError('Failed to send message. Please try again.');
                }}
            }}
            
            displayMessages(messages) {{
                this.elements.messagesList.innerHTML = '';
                messages.forEach(message => this.displayMessage(message));
                this.scrollToBottom();
            }}
            
            displayMessage(message) {{
                const messageEl = document.createElement('div');
                messageEl.className = `message ${{message.role}}`;
                messageEl.dataset.messageId = message.id;
                
                const avatar = message.role === 'user' ? 'U' : 'AI';
                const avatarColor = message.role === 'user' ? 'var(--user-message)' : 'var(--ai-message)';
                
                messageEl.innerHTML = `
                    <div class="message-avatar" style="background: ${{avatarColor}}">
                        ${{avatar}}
                    </div>
                    <div class="message-content">
                        <div class="message-text">${{this.formatMessageContent(message.content)}}</div>
                        <div class="message-metadata">
                            <span class="message-time">${{this.formatTime(message.timestamp)}}</span>
                            <div class="message-actions">
                                <button class="message-action" title="Copy" onclick="chatClient.copyMessage('${{message.id}}')">
                                    <i class="fas fa-copy"></i>
                                </button>
                                ${{message.role === 'assistant' ? `
                                    <button class="message-action" title="Thumbs Up" onclick="chatClient.rateMessage('${{message.id}}', true)">
                                        <i class="fas fa-thumbs-up"></i>
                                    </button>
                                    <button class="message-action" title="Thumbs Down" onclick="chatClient.rateMessage('${{message.id}}', false)">
                                        <i class="fas fa-thumbs-down"></i>
                                    </button>
                                ` : ''}}
                            </div>
                        </div>
                    </div>
                `;
                
                this.elements.messagesList.appendChild(messageEl);
                this.scrollToBottom();
            }}
            
            formatMessageContent(content) {{
                // Basic markdown-like formatting
                let formatted = content
                    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')  // Bold
                    .replace(/\\*(.*?)\\*/g, '<em>$1</em>')  // Italic
                    .replace(/`(.*?)`/g, '<code>$1</code>')  // Inline code
                    .replace(/\\n/g, '<br>');  // Line breaks
                
                // Auto-link URLs
                formatted = formatted.replace(
                    /(https?:\\/\\/[^\\s]+)/g,
                    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
                );
                
                return formatted;
            }}
            
            formatTime(timestamp) {{
                return new Date(timestamp).toLocaleTimeString([], {{
                    hour: '2-digit',
                    minute: '2-digit'
                }});
            }}
            
            clearChat() {{
                this.elements.messagesList.innerHTML = '';
                this.elements.welcomeMessage.style.display = 'flex';
                this.elements.messagesList.classList.remove('active');
            }}
            
            hideWelcomeMessage() {{
                this.elements.welcomeMessage.style.display = 'none';
                this.elements.messagesList.classList.add('active');
            }}
            
            showTypingIndicator() {{
                this.elements.typingIndicator.style.display = 'flex';
                this.scrollToBottom();
            }}
            
            hideTypingIndicator() {{
                this.elements.typingIndicator.style.display = 'none';
            }}
            
            scrollToBottom() {{
                this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
            }}
            
            handleInputChange() {{
                const hasText = this.elements.messageInput.value.trim().length > 0;
                this.elements.sendBtn.disabled = !hasText;
            }}
            
            sendTypingIndicator(isTyping) {{
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {{
                    this.websocket.send(JSON.stringify({{
                        type: 'typing',
                        data: {{ is_typing: isTyping }}
                    }}));
                }}
            }}
            
            async loadSuggestions() {{
                try {{
                    const response = await fetch(`${{this.apiBaseUrl}}/chat/suggestions?profile_id=${{this.profileId || ''}}`, {{
                        headers: {{
                            'Authorization': 'Bearer demo_token'
                        }}
                    }});
                    
                    const data = await response.json();
                    this.suggestions = data.suggestions;
                    this.showSuggestions(this.suggestions);
                }} catch (error) {{
                    console.error('Error loading suggestions:', error);
                }}
            }}
            
            showSuggestions(suggestions) {{
                if (!suggestions || suggestions.length === 0) {{
                    this.hideSuggestions();
                    return;
                }}
                
                this.elements.suggestionsList.innerHTML = '';
                suggestions.forEach(suggestion => {{
                    const btn = document.createElement('button');
                    btn.className = 'suggestion-btn';
                    btn.textContent = suggestion;
                    btn.addEventListener('click', () => {{
                        this.sendMessage(suggestion);
                    }});
                    this.elements.suggestionsList.appendChild(btn);
                }});
                
                this.elements.suggestionsContainer.style.display = 'block';
            }}
            
            hideSuggestions() {{
                this.elements.suggestionsContainer.style.display = 'none';
            }}
            
            handleQuickAction(action) {{
                const actionMessages = {{
                    'market-overview': 'Give me a market overview for today',
                    'portfolio-analysis': 'Analyze my portfolio performance',
                    'stock-analysis': 'Help me analyze a stock',
                    'strategy-backtest': 'I want to backtest a trading strategy'
                }};
                
                const message = actionMessages[action];
                if (message) {{
                    this.sendMessage(message);
                }}
            }}
            
            updateConnectionStatus(connected) {{
                const statusEl = this.elements.connectionStatus;
                const icon = statusEl.querySelector('i');
                const text = statusEl.querySelector('span');
                
                if (connected) {{
                    statusEl.className = 'connection-status connected';
                    text.textContent = 'Connected';
                }} else {{
                    statusEl.className = 'connection-status disconnected';
                    text.textContent = 'Disconnected';
                }}
            }}
            
            toggleTheme() {{
                const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
                this.setTheme(newTheme);
            }}
            
            setTheme(theme) {{
                this.currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                this.elements.themeSelect.value = theme;
                
                // Update theme toggle icon
                const icon = this.elements.themeToggle.querySelector('i');
                icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
                
                // Store preference
                localStorage.setItem('supernova-theme', theme);
            }}
            
            showSettingsModal() {{
                this.elements.settingsModal.classList.add('show');
            }}
            
            hideSettingsModal() {{
                this.elements.settingsModal.classList.remove('show');
            }}
            
            showChartModal(title, chartData) {{
                this.elements.chartTitle.textContent = title;
                this.elements.chartModal.classList.add('show');
                
                // Render chart with Plotly
                Plotly.newPlot(this.elements.chartContainer, chartData.data, chartData.layout, {{
                    responsive: true,
                    displayModeBar: true
                }});
            }}
            
            hideChartModal() {{
                this.elements.chartModal.classList.remove('show');
            }}
            
            displayChart(chartConfig) {{
                // This would integrate with your charting system
                const chartEl = document.createElement('div');
                chartEl.className = 'chart-embed';
                chartEl.innerHTML = `
                    <div class="chart-placeholder">
                        <i class="fas fa-chart-line" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                        <div>Click to view ${{chartConfig.symbol}} chart</div>
                    </div>
                `;
                
                chartEl.addEventListener('click', () => {{
                    this.showChartModal(
                        `${{chartConfig.symbol}} - ${{chartConfig.timeframe}}`,
                        this.generateSampleChartData(chartConfig.symbol)
                    );
                }});
                
                // Add to last message
                const lastMessage = this.elements.messagesList.lastElementChild;
                if (lastMessage) {{
                    const messageContent = lastMessage.querySelector('.message-content');
                    messageContent.appendChild(chartEl);
                }}
            }}
            
            generateSampleChartData(symbol) {{
                // Generate sample OHLC data for demo
                const dates = [];
                const opens = [];
                const highs = [];
                const lows = [];
                const closes = [];
                
                let price = 100;
                for (let i = 0; i < 30; i++) {{
                    const date = new Date();
                    date.setDate(date.getDate() - (30 - i));
                    dates.push(date.toISOString().split('T')[0]);
                    
                    const open = price;
                    const change = (Math.random() - 0.5) * 10;
                    const close = Math.max(0.1, open + change);
                    const high = Math.max(open, close) + Math.random() * 5;
                    const low = Math.min(open, close) - Math.random() * 5;
                    
                    opens.push(open);
                    highs.push(high);
                    lows.push(low);
                    closes.push(close);
                    
                    price = close;
                }}
                
                return {{
                    data: [{{
                        x: dates,
                        open: opens,
                        high: highs,
                        low: lows,
                        close: closes,
                        type: 'candlestick',
                        name: symbol
                    }}],
                    layout: {{
                        title: `${{symbol}} Price Chart`,
                        xaxis: {{ title: 'Date' }},
                        yaxis: {{ title: 'Price ($)' }},
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: {{ color: 'var(--text-primary)' }}
                    }}
                }};
            }}
            
            async handleFileUpload(event) {{
                const files = event.target.files;
                if (!files.length) return;
                
                // Show upload progress (simplified)
                this.showMessage('📁 Uploading file(s)...', 'system');
                
                try {{
                    // In a real implementation, you'd upload to your backend
                    // For now, just simulate the upload
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    
                    const fileNames = Array.from(files).map(f => f.name).join(', ');
                    this.sendMessage(`I've uploaded the following files: ${{fileNames}}. Please analyze them.`);
                    
                }} catch (error) {{
                    this.showError('Failed to upload files');
                }}
                
                // Reset file input
                event.target.value = '';
            }}
            
            toggleVoiceRecording() {{
                if (this.isRecording) {{
                    this.stopRecording();
                }} else {{
                    this.startRecording();
                }}
            }}
            
            async startRecording() {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    this.mediaRecorder = new MediaRecorder(stream);
                    this.isRecording = true;
                    
                    this.elements.voiceBtn.classList.add('recording');
                    this.elements.voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
                    
                    const audioChunks = [];
                    this.mediaRecorder.ondataavailable = (event) => {{
                        audioChunks.push(event.data);
                    }};
                    
                    this.mediaRecorder.onstop = async () => {{
                        const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                        await this.processVoiceMessage(audioBlob);
                    }};
                    
                    this.mediaRecorder.start();
                }} catch (error) {{
                    console.error('Error starting recording:', error);
                    this.showError('Failed to access microphone');
                }}
            }}
            
            stopRecording() {{
                if (this.mediaRecorder && this.isRecording) {{
                    this.mediaRecorder.stop();
                    this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
                    this.isRecording = false;
                    
                    this.elements.voiceBtn.classList.remove('recording');
                    this.elements.voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                }}
            }}
            
            async processVoiceMessage(audioBlob) {{
                // In a real implementation, you'd send this to a speech-to-text service
                // For now, just simulate the process
                this.showMessage('🎤 Processing voice message...', 'system');
                
                try {{
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    
                    // Simulate transcription
                    const transcription = "What's the latest news about Apple stock?";
                    this.sendMessage(transcription);
                    
                }} catch (error) {{
                    this.showError('Failed to process voice message');
                }}
            }}
            
            copyMessage(messageId) {{
                const messageEl = document.querySelector(`[data-message-id="${{messageId}}"]`);
                const messageText = messageEl.querySelector('.message-text').textContent;
                
                navigator.clipboard.writeText(messageText).then(() => {{
                    this.showMessage('Message copied to clipboard', 'system');
                }});
            }}
            
            async rateMessage(messageId, helpful) {{
                try {{
                    await fetch(`${{this.apiBaseUrl}}/chat/feedback`, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer demo_token'
                        }},
                        body: JSON.stringify({{
                            message_id: messageId,
                            helpful: helpful,
                            rating: helpful ? 5 : 2
                        }})
                    }});
                    
                    this.showMessage(`Thank you for your feedback!`, 'system');
                }} catch (error) {{
                    console.error('Error submitting feedback:', error);
                }}
            }}
            
            exportChat() {{
                const messages = Array.from(this.elements.messagesList.children).map(msgEl => ({{
                    role: msgEl.classList.contains('user') ? 'user' : 'assistant',
                    content: msgEl.querySelector('.message-text').textContent,
                    timestamp: msgEl.querySelector('.message-time').textContent
                }}));
                
                const chatData = {{
                    session_id: this.sessionId,
                    exported_at: new Date().toISOString(),
                    messages: messages
                }};
                
                const blob = new Blob([JSON.stringify(chatData, null, 2)], {{ type: 'application/json' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `supernova-chat-${{this.sessionId.slice(0, 8)}}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
            
            showMessage(text, type = 'system') {{
                const message = {{
                    id: Date.now().toString(),
                    role: type,
                    content: text,
                    timestamp: new Date().toISOString()
                }};
                
                this.displayMessage(message);
                
                // Auto-hide system messages after 3 seconds
                if (type === 'system') {{
                    setTimeout(() => {{
                        const msgEl = document.querySelector(`[data-message-id="${{message.id}}"]`);
                        if (msgEl) {{
                            msgEl.style.opacity = '0.5';
                        }}
                    }}, 3000);
                }}
            }}
            
            showError(message) {{
                this.showMessage(`❌ ${{message}}`, 'system');
            }}
            
            handleMarketUpdate(data) {{
                // Handle real-time market data updates
                console.log('Market update:', data);
            }}
            
            handleSystemMessage(data) {{
                this.showMessage(data.message, 'system');
            }}
        }}
        
        // Initialize chat client when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {{
            window.chatClient = new SuperNovaChatClient();
            
            // Load saved theme preference
            const savedTheme = localStorage.getItem('supernova-theme');
            if (savedTheme) {{
                window.chatClient.setTheme(savedTheme);
            }}
        }});
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {{
            if (document.hidden) {{
                // Page is hidden, reduce activity
                if (window.chatClient) {{
                    window.chatClient.sendTypingIndicator(false);
                }}
            }}
        }});
        
        // Handle beforeunload to cleanup WebSocket
        window.addEventListener('beforeunload', () => {{
            if (window.chatClient) {{
                window.chatClient.disconnectWebSocket();
            }}
        }});
    '''

async def render_chat_page(
    request: Request,
    session_id: Optional[str] = None,
    profile_id: Optional[int] = None
) -> HTMLResponse:
    """
    FastAPI endpoint to render chat interface
    """
    theme = request.cookies.get("theme", "dark")
    html_content = get_chat_interface_html(
        session_id=session_id,
        profile_id=profile_id,
        theme=theme
    )
    
    return HTMLResponse(content=html_content)

def create_mobile_optimized_chat() -> str:
    """
    Create a mobile-optimized version of the chat interface
    """
    return '''
    <!-- Mobile-specific optimizations -->
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="mobile-web-app-capable" content="yes">
    
    <!-- Progressive Web App configuration -->
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#00d4aa">
    
    <!-- Additional mobile styles -->
    <style>
        @media (max-width: 480px) {
            .input-wrapper {
                padding: 0.5rem;
            }
            
            .message-content {
                max-width: 90%;
            }
            
            .quick-actions {
                flex-direction: column;
                gap: 0.25rem;
            }
            
            .quick-action-btn {
                width: 100%;
                justify-content: center;
            }
            
            .settings-section {
                margin-bottom: 1rem;
            }
            
            .modal-content {
                width: 100%;
                height: 100%;
                border-radius: 0;
                margin: 0;
            }
        }
        
        /* Touch-friendly button sizes */
        @media (pointer: coarse) {
            .action-btn, .message-action, .upload-btn, .voice-btn, .send-btn {
                min-width: 44px;
                min-height: 44px;
            }
        }
    </style>
    '''

# Export main functions for use in FastAPI app
__all__ = [
    'get_chat_interface_html',
    'get_chat_styles', 
    'get_chat_javascript',
    'render_chat_page',
    'create_mobile_optimized_chat'
]