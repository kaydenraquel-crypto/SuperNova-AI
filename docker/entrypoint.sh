#!/bin/bash
set -e

# SuperNova AI Production Entrypoint Script
echo "🚀 Starting SuperNova AI Production Deployment..."

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for Redis
until nc -z redis 6379; do
  echo "Waiting for Redis..."
  sleep 2
done
echo "✅ Redis is ready"

# Wait for TimescaleDB
until nc -z timescaledb 5432; do
  echo "Waiting for TimescaleDB..."
  sleep 2
done
echo "✅ TimescaleDB is ready"

# Initialize database
echo "🗄️ Initializing database..."
python -c "
from supernova.db import init_db
from supernova.timescale_setup import initialize_timescale
try:
    init_db()
    print('✅ SQLite database initialized')
    initialize_timescale()
    print('✅ TimescaleDB initialized')
except Exception as e:
    print(f'⚠️ Database initialization warning: {e}')
"

# Run database migrations if needed
echo "🔄 Running database migrations..."
python -c "
from supernova.db import run_migrations
try:
    run_migrations()
    print('✅ Database migrations completed')
except Exception as e:
    print(f'⚠️ Migration warning: {e}')
"

# Create necessary directories
mkdir -p /app/logs /app/data /app/uploads
touch /app/logs/supernova.log

# Set proper permissions
chmod 755 /app/logs /app/data /app/uploads
chmod 644 /app/logs/supernova.log

# Health check endpoint for the application
echo "🏥 Setting up health monitoring..."

# Start application based on arguments
if [ "$1" = "web" ]; then
    echo "🌐 Starting web server..."
    exec uvicorn main:app --host 0.0.0.0 --port 8081 --workers 4 --log-level info
elif [ "$1" = "worker" ]; then
    echo "👷 Starting background worker..."
    exec python -m supernova.worker
elif [ "$1" = "scheduler" ]; then
    echo "📅 Starting scheduler..."
    exec python -m supernova.scheduler
elif [ "$1" = "migrate" ]; then
    echo "🔄 Running migrations only..."
    python -c "from supernova.db import run_migrations; run_migrations()"
    echo "✅ Migrations completed"
    exit 0
elif [ "$1" = "test" ]; then
    echo "🧪 Running tests..."
    exec pytest -v
else
    echo "🎛️ Starting supervisord..."
    exec "$@"
fi