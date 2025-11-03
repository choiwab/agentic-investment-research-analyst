import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for LLM processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Send a query to the equity research pipeline
 * @param {string} query - User query
 * @returns {Promise} - API response with analysis results
 */
export const runEquityResearch = async (query) => {
  try {
    const response = await api.post('/api/research', { query });
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.message ||
      'Failed to process query. Please try again.'
    );
  }
};

/**
 * Get list of available tickers
 * @returns {Promise<Array>} - List of ticker symbols
 */
export const getTickers = async () => {
  try {
    const response = await api.get('/api/tickers');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch tickers:', error);
    return [];
  }
};

/**
 * Get company information by ticker
 * @param {string} ticker - Stock ticker symbol
 * @returns {Promise<Object>} - Company information
 */
export const getCompanyInfo = async (ticker) => {
  try {
    const response = await api.get(`/api/company/${ticker}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch info for ${ticker}:`, error);
    return null;
  }
};

/**
 * Health check endpoint
 * @returns {Promise<boolean>} - Backend health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    return false;
  }
};

export default api;
