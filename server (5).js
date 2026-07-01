/**
 * AI Chatbot Backend Server
 * Node.js Express server with Groq API integration
 * Features: Chat completions, streaming responses, rate limiting, error handling
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const axios = require('axios');

// ===========================
// Configuration
// ===========================

const app = express();
const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1/chat/completions';

// System prompt for the AI assistant
const SYSTEM_PROMPT = `You are a helpful AI assistant like Qwen. You can help with coding, writing, analysis, math, and general questions. Provide clear, accurate, and helpful responses. When writing code, use proper formatting with syntax highlighting hints like \`\`\`language at the start of code blocks.`;

// Model configurations
const AVAILABLE_MODELS = {
  'llama-3.3-70b-versatile': { name: 'Llama 3.3 70B', maxTokens: 8192 },
  'mixtral-8x7b-32768': { name: 'Mixtral 8x7B', maxTokens: 32768 },
  'gemma-7b-it': { name: 'Gemma 7B', maxTokens: 8192 },
};

const DEFAULT_MODEL = 'llama-3.3-70b-versatile';

// ===========================
// Middleware Configuration
// ===========================

// Security headers
app.use(helmet());

// CORS
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  maxAge: 86400,
}));

// Body parser
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

// Rate limiting: 100 requests per hour per IP
const limiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
  skip: (req) => {
    // Skip rate limiting for health check
    return req.path === '/';
  },
  keyGenerator: (req) => {
    // Use IP address as the key
    return req.ip || req.connection.remoteAddress;
  },
});

app.use(limiter);

// ===========================
// Validation Middleware
// ===========================

/**
 * Validate that API key is configured
 */
function validateApiKey(req, res, next) {
  if (!GROQ_API_KEY) {
    return res.status(500).json({
      error: 'API key not configured. Please set GROQ_API_KEY environment variable.',
      timestamp: new Date().toISOString(),
    });
  }
  next();
}

/**
 * Validate chat request body
 */
function validateChatRequest(req, res, next) {
  const { message, history = [], model = DEFAULT_MODEL } = req.body;

  if (!message || typeof message !== 'string' || message.trim().length === 0) {
    return res.status(400).json({
      error: 'Invalid request: message is required and must be a non-empty string',
    });
  }

  if (!Array.isArray(history)) {
    return res.status(400).json({
      error: 'Invalid request: history must be an array',
    });
  }

  if (!AVAILABLE_MODELS[model]) {
    const availableModels = Object.keys(AVAILABLE_MODELS).join(', ');
    return res.status(400).json({
      error: `Invalid model. Available models: ${availableModels}`,
    });
  }

  // Validate history format
  for (let i = 0; i < history.length; i++) {
    if (!history[i].role || !history[i].content) {
      return res.status(400).json({
        error: 'Invalid history format. Each message must have "role" and "content" fields.',
      });
    }
  }

  req.validatedData = { message, history, model };
  next();
}

// ===========================
// Helper Functions
// ===========================

/**
 * Build messages array for Groq API
 * @param {string} userMessage - Current user message
 * @param {array} history - Previous messages
 * @returns {array} Formatted messages for API
 */
function buildMessages(userMessage, history = []) {
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...history,
    { role: 'user', content: userMessage },
  ];
  return messages;
}

/**
 * Create Groq API request config
 * @param {string} model - Model name
 * @param {array} messages - Messages array
 * @param {boolean} stream - Enable streaming
 * @returns {object} Request config
 */
function createGroqRequest(model, messages, stream = false) {
  return {
    model,
    messages,
    temperature: 0.7,
    max_tokens: AVAILABLE_MODELS[model].maxTokens,
    top_p: 0.95,
    stream,
  };
}

/**
 * Create Groq API headers
 * @returns {object} Request headers
 */
function createGroqHeaders() {
  return {
    'Authorization': `Bearer ${GROQ_API_KEY}`,
    'Content-Type': 'application/json',
  };
}

/**
 * Handle Groq API errors
 * @param {Error} error - Error object
 * @returns {object} Formatted error response
 */
function handleGroqError(error) {
  console.error('Groq API Error:', error.message);

  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;

    if (status === 429) {
      return {
        statusCode: 429,
        error: 'Rate limit exceeded. Please try again later.',
        retryAfter: error.response.headers['retry-after'] || 60,
      };
    }

    if (status === 401) {
      return {
        statusCode: 401,
        error: 'Authentication failed. Invalid API key.',
      };
    }

    if (status === 400) {
      return {
        statusCode: 400,
        error: data.error?.message || 'Invalid request to Groq API',
      };
    }

    if (status === 503) {
      return {
        statusCode: 503,
        error: 'Groq API is temporarily unavailable. Please try again later.',
      };
    }

    return {
      statusCode: status,
      error: data.error?.message || 'Groq API error',
    };
  }

  if (error.code === 'ECONNREFUSED') {
    return {
      statusCode: 503,
      error: 'Cannot connect to Groq API. Please check your internet connection.',
    };
  }

  if (error.code === 'ETIMEDOUT') {
    return {
      statusCode: 504,
      error: 'Request to Groq API timed out. Please try again.',
    };
  }

  return {
    statusCode: 500,
    error: 'An unexpected error occurred',
  };
}

// ===========================
// Routes
// ===========================

/**
 * Health Check Endpoint
 * GET /
 */
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    service: 'AI Chatbot Backend',
  });
});

/**
 * Chat Completion Endpoint (Non-streaming)
 * POST /api/chat
 * Body: { message, history?, model? }
 */
app.post('/api/chat', validateApiKey, validateChatRequest, async (req, res) => {
  try {
    const { message, history, model } = req.validatedData;

    console.log(`Processing chat request with model: ${model}`);

    // Build messages for API
    const messages = buildMessages(message, history);

    // Create Groq API request
    const groqRequest = createGroqRequest(model, messages, false);
    const headers = createGroqHeaders();

    // Call Groq API
    const response = await axios.post(GROQ_BASE_URL, groqRequest, { headers });

    // Extract response data
    const aiMessage = response.data.choices[0].message.content;
    const usage = response.data.usage;

    console.log(`Chat response generated. Tokens - Input: ${usage.prompt_tokens}, Output: ${usage.completion_tokens}`);

    // Return response
    res.json({
      response: aiMessage,
      model,
      usage: {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    const errorResponse = handleGroqError(error);
    const statusCode = errorResponse.statusCode || 500;

    res.status(statusCode).json({
      error: errorResponse.error,
      ...(errorResponse.retryAfter && { 'retry-after': errorResponse.retryAfter }),
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * Streaming Chat Endpoint (Server-Sent Events)
 * POST /api/chat/stream
 * Body: { message, history?, model? }
 */
app.post('/api/chat/stream', validateApiKey, validateChatRequest, async (req, res) => {
  try {
    const { message, history, model } = req.validatedData;

    console.log(`Processing streaming chat request with model: ${model}`);

    // Build messages for API
    const messages = buildMessages(message, history);

    // Create Groq API request
    const groqRequest = createGroqRequest(model, messages, true);
    const headers = createGroqHeaders();

    // Set SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');

    // Call Groq API with streaming
    const response = await axios.post(GROQ_BASE_URL, groqRequest, {
      headers,
      responseType: 'stream',
    });

    let tokenCount = 0;
    let fullResponse = '';

    // Handle streaming response
    response.data.on('data', (chunk) => {
      const lines = chunk.toString().split('\n').filter((line) => line.trim());

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();

          if (data === '[DONE]') {
            // Stream complete
            res.write(`data: [DONE]\n\n`);
            console.log(`Stream completed. Total tokens: ${tokenCount}`);
            res.end();
            return;
          }

          try {
            const json = JSON.parse(data);
            const token = json.choices[0]?.delta?.content || '';

            if (token) {
              tokenCount++;
              fullResponse += token;
              // Send token as SSE
              res.write(`data: ${JSON.stringify({ token, index: tokenCount })}\n\n`);
            }
          } catch (e) {
            // Skip invalid JSON
            console.error('Error parsing stream data:', e.message);
          }
        }
      }
    });

    // Handle stream errors
    response.data.on('error', (error) => {
      console.error('Stream error:', error.message);
      res.write(`data: ${JSON.stringify({ error: 'Stream error occurred' })}\n\n`);
      res.end();
    });

    // Handle client disconnect
    req.on('close', () => {
      console.log('Client disconnected from stream');
      response.data.destroy();
    });
  } catch (error) {
    const errorResponse = handleGroqError(error);
    const statusCode = errorResponse.statusCode || 500;

    res.status(statusCode).json({
      error: errorResponse.error,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * Models List Endpoint
 * GET /api/models
 */
app.get('/api/models', (req, res) => {
  const models = Object.entries(AVAILABLE_MODELS).map(([id, config]) => ({
    id,
    name: config.name,
    maxTokens: config.maxTokens,
  }));

  res.json({
    models,
    default: DEFAULT_MODEL,
    timestamp: new Date().toISOString(),
  });
});

// ===========================
// Error Handling
// ===========================

/**
 * 404 Handler
 */
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.path,
    method: req.method,
    timestamp: new Date().toISOString(),
  });
});

/**
 * Global Error Handler
 */
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);

  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined,
    timestamp: new Date().toISOString(),
  });
});

// ===========================
// Server Startup
// ===========================

app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════╗
║   AI Chatbot Backend Server Started  ║
╚══════════════════════════════════════╝

Server: http://localhost:${PORT}
Environment: ${process.env.NODE_ENV || 'development'}
Groq API: ${GROQ_API_KEY ? '✓ Configured' : '✗ Not configured'}

Available Endpoints:
  GET  /                    - Health check
  GET  /api/models          - List available models
  POST /api/chat            - Chat completion
  POST /api/chat/stream     - Streaming chat

Available Models:
${Object.entries(AVAILABLE_MODELS)
  .map(([id, config]) => `  - ${config.name} (${id})`)
  .join('\n')}

Rate Limiting: 100 requests/hour per IP
CORS: Enabled for all origins
Security: Helmet enabled

Ready to receive requests...
  `);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  process.exit(0);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

module.exports = app;