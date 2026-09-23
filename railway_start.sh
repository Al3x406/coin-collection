#!/bin/sh
set -eu
PORT="${PORT:-8080}"
if [ -d /data ]; then
  mkdir -p /data/uploads/coins /data/uploads/artifacts
  if [ ! -f /data/coins.db ]; then
    cp /app/instance/coins.db /data/coins.db
  fi
  if [ ! -f /data/.uploads_initialized ]; then
    cp -a /app/static/uploads/. /data/uploads/
    touch /data/.uploads_initialized
  fi
  rm -rf /app/static/uploads
  ln -s /data/uploads /app/static/uploads
  export DATABASE_URL="sqlite:////data/coins.db"
fi
exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 120 app:app
