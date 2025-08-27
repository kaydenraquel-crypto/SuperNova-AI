-- SuperNova AI Collaboration System Database Migration
-- Adds team management, sharing, and communication tables
-- Migration: 002_collaboration_setup.sql

-- ================================
-- TEAM MANAGEMENT TABLES
-- ================================

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_private BOOLEAN DEFAULT 1 NOT NULL,
    max_members INTEGER DEFAULT 50 NOT NULL,
    invite_code VARCHAR(20) UNIQUE,
    created_by INTEGER NOT NULL,
    parent_team_id INTEGER,
    settings TEXT, -- JSON configuration
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1 NOT NULL,
    FOREIGN KEY (created_by) REFERENCES users(id),
    FOREIGN KEY (parent_team_id) REFERENCES teams(id)
);

-- Team members association table
CREATE TABLE IF NOT EXISTS team_members (
    team_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role VARCHAR(20) DEFAULT 'member' NOT NULL,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    invited_by INTEGER,
    is_active BOOLEAN DEFAULT 1 NOT NULL,
    permissions TEXT, -- JSON string for custom permissions
    PRIMARY KEY (team_id, user_id),
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (invited_by) REFERENCES users(id)
);

-- Team invitations table
CREATE TABLE IF NOT EXISTS team_invitations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    email VARCHAR(255) NOT NULL,
    invited_by INTEGER NOT NULL,
    role VARCHAR(20) DEFAULT 'member' NOT NULL,
    invitation_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_accepted BOOLEAN DEFAULT 0 NOT NULL,
    accepted_at TIMESTAMP,
    accepted_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    message TEXT,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (invited_by) REFERENCES users(id),
    FOREIGN KEY (accepted_by) REFERENCES users(id)
);

-- ================================
-- SHARING & PERMISSIONS TABLES
-- ================================

-- Shared portfolios table
CREATE TABLE IF NOT EXISTS shared_portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    team_id INTEGER,
    user_id INTEGER,
    permission_level VARCHAR(20) DEFAULT 'view' NOT NULL,
    is_public BOOLEAN DEFAULT 0 NOT NULL,
    public_share_token VARCHAR(255) UNIQUE,
    password_protected BOOLEAN DEFAULT 0 NOT NULL,
    access_password VARCHAR(255),
    shared_by INTEGER NOT NULL,
    expires_at TIMESTAMP,
    view_count INTEGER DEFAULT 0 NOT NULL,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1 NOT NULL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (shared_by) REFERENCES users(id),
    CHECK ((team_id IS NOT NULL AND user_id IS NULL) OR (team_id IS NULL AND user_id IS NOT NULL) OR (team_id IS NULL AND user_id IS NULL AND is_public = 1))
);

-- Portfolio shares association table
CREATE TABLE IF NOT EXISTS portfolio_shares (
    portfolio_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    permission_level VARCHAR(20) DEFAULT 'view' NOT NULL,
    shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    shared_by INTEGER NOT NULL,
    expires_at TIMESTAMP,
    PRIMARY KEY (portfolio_id, user_id),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (shared_by) REFERENCES users(id)
);

-- Strategy shares association table
CREATE TABLE IF NOT EXISTS strategy_shares (
    strategy_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    permission_level VARCHAR(20) DEFAULT 'view' NOT NULL,
    shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    shared_by INTEGER NOT NULL,
    expires_at TIMESTAMP,
    PRIMARY KEY (strategy_id, user_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (shared_by) REFERENCES users(id)
);

-- Public share links table
CREATE TABLE IF NOT EXISTS share_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    share_token VARCHAR(255) UNIQUE NOT NULL,
    resource_type VARCHAR(50) NOT NULL, -- portfolio, strategy, analysis
    resource_id INTEGER NOT NULL,
    permission_level VARCHAR(20) DEFAULT 'view' NOT NULL,
    password_protected BOOLEAN DEFAULT 0 NOT NULL,
    access_password VARCHAR(255),
    max_views INTEGER,
    current_views INTEGER DEFAULT 0 NOT NULL,
    expires_at TIMESTAMP,
    created_by INTEGER NOT NULL,
    title VARCHAR(255),
    description TEXT,
    unique_visitors INTEGER DEFAULT 0 NOT NULL,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1 NOT NULL,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- ================================
-- COMMUNICATION TABLES
-- ================================

-- Team chat channels table
CREATE TABLE IF NOT EXISTS team_chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_private BOOLEAN DEFAULT 0 NOT NULL,
    channel_type VARCHAR(20) DEFAULT 'general' NOT NULL, -- general, portfolio, strategy
    resource_id INTEGER, -- linked portfolio/strategy
    created_by INTEGER NOT NULL,
    last_message_at TIMESTAMP,
    message_count INTEGER DEFAULT 0 NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1 NOT NULL,
    is_archived BOOLEAN DEFAULT 0 NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'text' NOT NULL, -- text, file, image, system
    thread_id INTEGER,
    mentions TEXT, -- JSON array of mentioned user IDs
    attachments TEXT, -- JSON array of file info
    edited_at TIMESTAMP,
    reactions TEXT, -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    is_deleted BOOLEAN DEFAULT 0 NOT NULL,
    deleted_at TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES team_chats(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (thread_id) REFERENCES chat_messages(id)
);

-- Direct messages table
CREATE TABLE IF NOT EXISTS direct_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL,
    recipient_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'text' NOT NULL,
    is_read BOOLEAN DEFAULT 0 NOT NULL,
    read_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    is_deleted BOOLEAN DEFAULT 0 NOT NULL,
    deleted_by INTEGER,
    FOREIGN KEY (sender_id) REFERENCES users(id),
    FOREIGN KEY (recipient_id) REFERENCES users(id),
    FOREIGN KEY (deleted_by) REFERENCES users(id)
);

-- ================================
-- NOTIFICATION TABLES
-- ================================

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data TEXT, -- JSON with additional context
    action_url VARCHAR(500),
    team_id INTEGER,
    related_user_id INTEGER,
    is_read BOOLEAN DEFAULT 0 NOT NULL,
    read_at TIMESTAMP,
    email_sent BOOLEAN DEFAULT 0 NOT NULL,
    push_sent BOOLEAN DEFAULT 0 NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (related_user_id) REFERENCES users(id)
);

-- ================================
-- ACTIVITY & AUDIT TABLES
-- ================================

-- Team activities table
CREATE TABLE IF NOT EXISTS team_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    metadata TEXT, -- JSON with activity data
    resource_type VARCHAR(50), -- portfolio, strategy, etc.
    resource_id INTEGER,
    is_public BOOLEAN DEFAULT 1 NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Collaboration audit trail table
CREATE TABLE IF NOT EXISTS collaboration_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id INTEGER NOT NULL,
    details TEXT, -- JSON with action details
    ip_address VARCHAR(45),
    user_agent TEXT,
    team_id INTEGER,
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (team_id) REFERENCES teams(id)
);

-- ================================
-- COMMENT TABLES
-- ================================

-- Portfolio/strategy comments table
CREATE TABLE IF NOT EXISTS portfolio_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_type VARCHAR(20) NOT NULL, -- portfolio, strategy, backtest
    resource_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    parent_comment_id INTEGER,
    mentions TEXT, -- JSON array of mentioned users
    attachments TEXT, -- JSON array of attachments
    is_resolved BOOLEAN DEFAULT 0 NOT NULL,
    resolved_by INTEGER,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (parent_comment_id) REFERENCES portfolio_comments(id),
    FOREIGN KEY (resolved_by) REFERENCES users(id)
);

-- ================================
-- INDEXES FOR PERFORMANCE
-- ================================

-- Team indexes
CREATE INDEX IF NOT EXISTS idx_teams_created_by ON teams(created_by);
CREATE INDEX IF NOT EXISTS idx_teams_active ON teams(is_active);
CREATE INDEX IF NOT EXISTS idx_teams_invite_code ON teams(invite_code);
CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_team_members_active ON team_members(is_active);
CREATE INDEX IF NOT EXISTS idx_team_invitations_email ON team_invitations(email);
CREATE INDEX IF NOT EXISTS idx_team_invitations_token ON team_invitations(invitation_token);
CREATE INDEX IF NOT EXISTS idx_team_invitations_expires ON team_invitations(expires_at);

-- Sharing indexes
CREATE INDEX IF NOT EXISTS idx_shared_portfolios_portfolio ON shared_portfolios(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_shared_portfolios_team ON shared_portfolios(team_id);
CREATE INDEX IF NOT EXISTS idx_shared_portfolios_user ON shared_portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_shared_portfolios_token ON shared_portfolios(public_share_token);
CREATE INDEX IF NOT EXISTS idx_shared_portfolios_active ON shared_portfolios(is_active);
CREATE INDEX IF NOT EXISTS idx_share_links_token ON share_links(share_token);
CREATE INDEX IF NOT EXISTS idx_share_links_resource ON share_links(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_share_links_created_by ON share_links(created_by);

-- Communication indexes
CREATE INDEX IF NOT EXISTS idx_team_chats_team ON team_chats(team_id);
CREATE INDEX IF NOT EXISTS idx_team_chats_active ON team_chats(is_active);
CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user ON chat_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_direct_messages_sender ON direct_messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_direct_messages_recipient ON direct_messages(recipient_id);
CREATE INDEX IF NOT EXISTS idx_direct_messages_created ON direct_messages(created_at);

-- Notification indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_type ON notifications(type);
CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at);

-- Activity indexes
CREATE INDEX IF NOT EXISTS idx_team_activities_team ON team_activities(team_id);
CREATE INDEX IF NOT EXISTS idx_team_activities_user ON team_activities(user_id);
CREATE INDEX IF NOT EXISTS idx_team_activities_type ON team_activities(activity_type);
CREATE INDEX IF NOT EXISTS idx_team_activities_created ON team_activities(created_at);
CREATE INDEX IF NOT EXISTS idx_collaboration_audits_user ON collaboration_audits(user_id);
CREATE INDEX IF NOT EXISTS idx_collaboration_audits_action ON collaboration_audits(action);
CREATE INDEX IF NOT EXISTS idx_collaboration_audits_resource ON collaboration_audits(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_collaboration_audits_created ON collaboration_audits(created_at);

-- Comment indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_comments_resource ON portfolio_comments(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_comments_user ON portfolio_comments(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_comments_parent ON portfolio_comments(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_comments_created ON portfolio_comments(created_at);

-- ================================
-- INITIAL DATA
-- ================================

-- Insert default team chat channel types
INSERT OR IGNORE INTO team_chats (id, team_id, name, description, channel_type, created_by) 
VALUES (0, 0, 'General', 'Default general discussion channel', 'general', 1);

-- Insert default notification types (for reference)
-- These would be handled by the application layer, but documented here

-- ================================
-- VIEWS FOR COMMON QUERIES
-- ================================

-- Team member summary view
CREATE VIEW IF NOT EXISTS team_member_summary AS
SELECT 
    t.id as team_id,
    t.name as team_name,
    t.is_private,
    COUNT(tm.user_id) as member_count,
    t.max_members,
    t.created_at as team_created_at
FROM teams t
LEFT JOIN team_members tm ON t.id = tm.team_id AND tm.is_active = 1
WHERE t.is_active = 1
GROUP BY t.id, t.name, t.is_private, t.max_members, t.created_at;

-- Recent team activity view
CREATE VIEW IF NOT EXISTS recent_team_activity AS
SELECT 
    ta.team_id,
    t.name as team_name,
    ta.user_id,
    u.name as user_name,
    ta.activity_type,
    ta.title,
    ta.description,
    ta.created_at
FROM team_activities ta
JOIN teams t ON ta.team_id = t.id
JOIN users u ON ta.user_id = u.id
WHERE ta.is_public = 1 AND t.is_active = 1
ORDER BY ta.created_at DESC;

-- User notification summary view
CREATE VIEW IF NOT EXISTS user_notification_summary AS
SELECT 
    n.user_id,
    COUNT(CASE WHEN n.is_read = 0 THEN 1 END) as unread_count,
    COUNT(*) as total_notifications,
    MAX(n.created_at) as latest_notification
FROM notifications n
GROUP BY n.user_id;

-- Team sharing summary view
CREATE VIEW IF NOT EXISTS team_sharing_summary AS
SELECT 
    t.id as team_id,
    t.name as team_name,
    COUNT(DISTINCT sp.portfolio_id) as shared_portfolios,
    COUNT(DISTINCT sl.id) as public_links,
    t.created_at
FROM teams t
LEFT JOIN shared_portfolios sp ON t.id = sp.team_id AND sp.is_active = 1
LEFT JOIN share_links sl ON sl.created_by IN (
    SELECT tm.user_id FROM team_members tm WHERE tm.team_id = t.id AND tm.is_active = 1
)
WHERE t.is_active = 1
GROUP BY t.id, t.name, t.created_at;

-- ================================
-- TRIGGERS FOR DATA INTEGRITY
-- ================================

-- Update team member count on team_members changes
CREATE TRIGGER IF NOT EXISTS update_team_activity_on_member_change
AFTER INSERT ON team_members
BEGIN
    INSERT INTO team_activities (team_id, user_id, activity_type, title, metadata)
    VALUES (
        NEW.team_id,
        NEW.user_id,
        'user_joined',
        'User joined the team',
        json_object('role', NEW.role, 'invited_by', NEW.invited_by)
    );
END;

-- Update last_message_at when new chat message is added
CREATE TRIGGER IF NOT EXISTS update_chat_last_message
AFTER INSERT ON chat_messages
BEGIN
    UPDATE team_chats 
    SET last_message_at = NEW.created_at,
        message_count = message_count + 1
    WHERE id = NEW.chat_id;
END;

-- Auto-expire old invitations
-- Note: This would typically be handled by a scheduled job
-- CREATE TRIGGER IF NOT EXISTS expire_old_invitations
-- AFTER INSERT ON team_invitations
-- BEGIN
--     UPDATE team_invitations 
--     SET is_accepted = 0 
--     WHERE expires_at < datetime('now') AND is_accepted = 0;
-- END;

-- ================================
-- COMPLETION MESSAGE
-- ================================

-- Insert migration record
INSERT OR REPLACE INTO schema_migrations (version, description, applied_at) 
VALUES ('002', 'Collaboration system tables and indexes', datetime('now'));

-- Success message
SELECT 'SuperNova AI Collaboration System migration completed successfully!' as message;