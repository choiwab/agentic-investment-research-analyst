#!/bin/bash

# React Frontend Setup Script
# This script sets up the React frontend for local development

set -e

echo "================================"
echo "React Frontend Setup"
echo "================================"
echo ""

# Check Node.js version
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version is too old (found v$NODE_VERSION, need v18+)"
    echo "Please upgrade Node.js from https://nodejs.org"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  Please edit .env and set your API URL:"
    echo "   VITE_API_URL=http://localhost:8000"
    echo ""
else
    echo "✅ .env file already exists"
    echo ""
fi

# Check if backend is running
echo "🔍 Checking if backend is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running on http://localhost:8000"
else
    echo "⚠️  Backend is not running"
    echo ""
    echo "To start the backend, run from project root:"
    echo "  uvicorn backend.app.api_controller:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
fi

echo "================================"
echo "Setup Complete! 🎉"
echo "================================"
echo ""
echo "To start the development server, run:"
echo "  npm run dev"
echo ""
echo "The app will be available at:"
echo "  http://localhost:3000"
echo ""
