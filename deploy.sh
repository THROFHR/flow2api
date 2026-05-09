#!/bin/bash

echo "📥 Pull latest code..."
git pull origin main

echo "🏗️ Stop server..."
docker-compose down

echo "🔁 Reload server..."
docker-compose -f docker-compose.local.yml up -d --build

echo "✅ Done"
