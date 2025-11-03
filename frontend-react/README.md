# Equity Research AI - React Frontend

A modern, professional React frontend for the AI-powered equity research chatbot.

## Features

- 🎨 **Modern UI**: Clean, responsive design with dark theme
- 💬 **Real-time Chat**: Interactive chat interface with markdown support
- 📊 **Multi-Intent Support**: Company analysis, market insights, and financial education
- ⚡ **Fast**: Built with Vite for lightning-fast development and builds
- 🎯 **Type-Safe API**: Axios-based API client with error handling
- 📱 **Responsive**: Works on desktop, tablet, and mobile

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000` (or configured in `.env`)

## Quick Start

### 1. Install Dependencies

```bash
cd frontend-react
npm install
```

### 2. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set your API URL:

```
VITE_API_URL=http://localhost:8000
```

### 3. Start Development Server

```bash
npm run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000)

## Available Scripts

- `npm run dev` - Start development server (port 3000)
- `npm run build` - Build for production (outputs to `dist/`)
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint

## Project Structure

```
frontend-react/
├── public/              # Static assets
├── src/
│   ├── components/      # React components
│   │   ├── ChatInterface.jsx
│   │   ├── ChatInput.jsx
│   │   ├── Message.jsx
│   │   ├── MessageList.jsx
│   │   ├── LoadingIndicator.jsx
│   │   └── Sidebar.jsx
│   ├── services/        # API services
│   │   └── api.js
│   ├── styles/          # CSS files
│   │   ├── index.css
│   │   ├── App.css
│   │   └── *.css
│   ├── App.jsx          # Main app component
│   └── main.jsx         # Entry point
├── index.html
├── vite.config.js
└── package.json
```

## API Integration

The frontend connects to the FastAPI backend via:

- **POST /api/research** - Main research endpoint
- **GET /api/tickers** - Get available tickers
- **GET /api/company/{ticker}** - Get company info
- **GET /health** - Health check

API calls are handled in `src/services/api.js`.

## Deployment

### Option 1: Vercel (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd frontend-react
vercel
```

Set environment variable in Vercel dashboard:
- `VITE_API_URL` = Your backend API URL

### Option 2: Netlify

```bash
# Build
npm run build

# Deploy dist/ folder to Netlify
```

Set environment variable in Netlify:
- `VITE_API_URL` = Your backend API URL

### Option 3: Static Hosting

Build and upload the `dist/` folder to any static host:

```bash
npm run build
# Upload dist/ folder
```

## Customization

### Styling

All styles are in `src/styles/`. Modify CSS variables in `index.css`:

```css
:root {
  --primary-color: #4CAF50;
  --background-dark: #0E1117;
  /* ... more variables ... */
}
```

### API URL

Change the API URL in `.env`:

```
VITE_API_URL=https://your-api-domain.com
```

### Example Queries

Edit example queries in `src/components/ChatInput.jsx`:

```javascript
const exampleQueries = [
  "Your custom query 1",
  "Your custom query 2",
  "Your custom query 3",
];
```

## Troubleshooting

### CORS Errors

Make sure your backend has CORS configured for your frontend URL. Check `backend/app/api_controller.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    ...
)
```

### API Connection Failed

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check `.env` has correct `VITE_API_URL`
3. Restart dev server after changing `.env`

### Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules dist
npm install
npm run build
```

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **React Syntax Highlighter** - Code highlighting

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.
