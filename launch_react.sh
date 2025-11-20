#!/bin/bash
# Launch React Frontend

echo "=================================================="
echo "Launching Equity Research AI - React Frontend"
echo "=================================================="
echo ""

# Check if backend is running
if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "⚠️  Backend not running!"
    echo ""
    echo "Please start the backend first in another terminal:"
    echo "  cd backend"
    echo "  python -m uvicorn app.api_controller:app --host 127.0.0.1 --port 8000 --reload"
    echo ""
    exit 1
fi

echo "✓ Backend is running"
echo ""

cd frontend-react

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
    echo ""
fi

echo "Starting React frontend with Vite..."
echo "------------------------------------------------"
echo ""
echo "Frontend will be available at: http://localhost:5173"
echo ""

npm run dev
