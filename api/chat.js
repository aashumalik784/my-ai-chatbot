/**
 * Vercel Serverless AI Chatbot API Endpoint
 * 
 * Handles AI chat completions using Groq API
 * Features: Rate limiting, CORS, error handling, request validation
 * 
 * @file /api/chat.js
 * @author Your Name
 * @version 1.0.0
 */

const https = require('https');
const querystring = require('querystring');

// ===========================
// Configuration & Constants
// ===========================

const GROQ_BASE_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const NODE_ENV = process.env.NODE_ENV || 'development';

// System prompt for AI assistant
const SYSTEM_PROMPT = `You are a helpful AI assistant like Qwen. You can help with coding, writing, analysis, math, and general questions. Provide clear, accurate, and helpful responses. When writing code, use proper formatting and explain your solutions.`;

// Available AI models
const AVAILABLE_MODELS = {
  'llama-3.3-70b-versatile': {
    name: 'Llama 3.3 70B Versatile',
    maxTokens: 8192,
  },
  'mixtral-8x7b-32768': {
    name: 'Mixtral 8x7B 32K',
    maxTokens: 32768,
  },
  'gemma-7b-it': {
    name: 'Gemma 7B IT',
    maxTokens: 8192,
  },
  'llama3-8b-8192': {
    name: 'Llama 3 8B 8K',
    maxTokens: 8192,
  },
  'llama3-70b-8192': {
    name: 'Llama 3 70B 8K',
    maxTokens: 8192,
  },
};

const DEFAULT_MODEL = 'llama-3.3-70b-versatile';

// Rate limiting store (in-memory)
const rateLimitStore = {};
const RATE_LIMIT_WINDOW = 60 * 60 * 1000; // 1 hour in milliseconds
const RATE_LIMIT_MAX = 100; // max requests per hour

// ===========================
// Utility Functions
// ===========================

/**
 * Get current timestamp in ISO format
 * @returns {string} ISO timestamp
 */
function getTimestamp() {
  return new Date().toISOString();
}

/**
 * Get client IP address from request
 * @param {object} req - Node request object
 * @returns {string} Client IP address
 */
function getClientIP(req) {
  return (
    req.headers['x-forwarded-for']?.split(',')[0] ||
    req.headers['x-real-ip'] ||
    req.connection.remoteAddress ||
    'unknown'
  ).trim();
}

/**
 * Set CORS headers on response
 * @param {object} res - Node response object
 */
function setCORSHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Content-Type', 'application/json');
}

/**
 * Check if request is HTTPS (security requirement)
 * @param {object} req - Node request object
 * @returns {boolean} True if HTTPS or development environment
 */
function isSecureRequest(req) {
  if (NODE_ENV === 'development') return true;
  const proto = req.headers['x-forwarded-proto'] || 'https';
  return proto === 'https';
}

/**
 * Log request information
 * @param {string} message - Log message
 * @param {object} meta - Additional metadata
 */
function log(message, meta = {}) {
  const logEntry = {
    timestamp: getTimestamp(),
    message,
    ...meta,
  };
  console.log(JSON.stringify(logEntry));
}

/**
 * Sanitize user input (basic XSS prevention)
 * @param {string} input - Raw user input
 * @returns {string} Sanitized input
 */
function sanitizeInput(input) {
  if (typeof input !== 'string') return '';
  return input
    .trim()
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .slice(0, 10000); // Max 10k characters
}

/**
 * Check and update rate limit for IP
 * @param {string} ip - Client IP address
 * @returns {object} { allowed: boolean, remaining: number, resetTime: number }
 */
function checkRateLimit(ip) {
  const now = Date.now();
  const key = `rl_${ip}`;

  // Clean up old entries
  if (!rateLimitStore[key]) {
    rateLimitStore[key] = {
      count: 0,
      resetTime: now + RATE_LIMIT_WINDOW,
    };
  }

  // Check if window has expired
  if (now > rateLimitStore[key].resetTime) {
    rateLimitStore[key] = {
      count: 0,
      resetTime: now + RATE_LIMIT_WINDOW,
    };
  }

  const current = rateLimitStore[key];
  const allowed = current.count < RATE_LIMIT_MAX;

  if (allowed) {
    current.count++;
  }

  const remaining = Math.max(0, RATE_LIMIT_MAX - current.count);
  const retryAfter = Math.ceil((current.resetTime - now) / 1000);

  return { allowed, remaining, retryAfter };
}

/**
 * Send error response
 * @param {object} res - Node response object
 * @param {number} statusCode - HTTP status code
 * @param {string} error - Error message
 * @param {object} options - Additional options
 */
function sendError(res, statusCode, error, options = {}) {
  const { details, retryAfter, ...rest } = options;

  res.statusCode = statusCode;

  if (retryAfter) {
    res.setHeader('Retry-After', retryAfter);
  }

  const response = {
    error,
    ...(NODE_ENV === 'development' && details && { details }),
    ...rest,
  };

  res.end(JSON.stringify(response));
}

/**
 * Send success response
 * @param {object} res - Node response object
 * @param {object} data - Response data
 */
function sendSuccess(res, data) {
  res.statusCode = 200;
  res.end(JSON.stringify(data));
}

/**
 * Validate request body
 * @param {object} body - Request body object
 * @returns {object} { valid: boolean, error?: string, data?: object }
 */
function validateRequest(body) {
  if (!body) {
    return { valid: false, error: 'Request body is required' };
  }

  const { message, history = [], model = DEFAULT_MODEL } = body;

  // Validate message
  if (!message || typeof message !== 'string') {
    return { valid: false, error: 'Message is required and must be a string' };
  }

  if (message.trim().length === 0) {
    return { valid: false, error: 'Message cannot be empty' };
  }

  if (message.length > 10000) {
    return { valid: false, error: 'Message exceeds maximum length of 10000 characters' };
  }

  // Validate history
  if (!Array.isArray(history)) {
    return { valid: false, error: 'History must be an array' };
  }

  for (let i = 0; i < history.length; i++) {
    const msg = history[i];
    if (!msg.role || !msg.content || typeof msg.role !== 'string' || typeof msg.content !== 'string') {
      return { valid: false, error: `Invalid message format at index ${i}` };
    }
  }

  // Validate model
  if (!AVAILABLE_MODELS[model]) {
    const models = Object.keys(AVAILABLE_MODELS).join(', ');
    return { valid: false, error: `Invalid model. Available: ${models}` };
  }

  return {
    valid: true,
    data: {
      message: sanitizeInput(message),
      history: history.map(msg => ({
        role: msg.role,
        content: sanitizeInput(msg.content),
      })),
      model,
    },
  };
}

/**
 * Make HTTPS request to Groq API
 * @param {object} payload - Request payload
 * @param {string} apiKey - Groq API key
 * @returns {Promise<object>} API response
 */
async function callGroqAPI(payload, apiKey) {
  return new Promise((resolve, reject) => {
    const postData = JSON.stringify(payload);

    const options = {
      hostname: 'api.groq.com',
      path: '/openai/v1/chat/completions',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
        'User-Agent': 'Vercel-AI-Chatbot/1.0',
      },
      timeout: 30000, // 30 second timeout
    };

    const req = https.request(options, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ status: res.statusCode, data: parsed, headers: res.headers });
        } catch (e) {
          reject(new Error(`Failed to parse response: ${e.message}`));
        }
      });
    });

    req.on('error', (error) => {
      reject(error);
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request to Groq API timed out'));
    });

    req.write(postData);
    req.end();
  });
}

/**
 * Handle /api/models endpoint
 * @param {object} res - Node response object
 */
function handleModelsEndpoint(res) {
  const models = Object.entries(AVAILABLE_MODELS).map(([id, config]) => ({
    id,
    name: config.name,
    maxTokens: config.maxTokens,
  }));

  sendSuccess(res, {
    models,
    default: DEFAULT_MODEL,
    timestamp: getTimestamp(),
  });
}

/**
 * Handle preflight OPTIONS request
 * @param {object} res - Node response object
 */
function handlePreflight(res) {
  res.statusCode = 200;
  res.end();
}

// ===========================
// Main Handler
// ===========================

/**
 * Main Vercel serverless handler function
 * @param {object} req - Node request object
 * @param {object} res - Node response object
 * @returns {Promise<void>}
 */
async function handler(req, res) {
  try {
    // Set CORS headers
    setCORSHeaders(res);

    // Get client IP
    const clientIP = getClientIP(req);
    log('Incoming request', {
      method: req.method,
      url: req.url,
      ip: clientIP,
    });

    // Handle preflight requests
    if (req.method === 'OPTIONS') {
      handlePreflight(res);
      return;
    }

    // Handle GET /api/models
    if (req.method === 'GET' && req.url === '/api/models') {
      log('Models endpoint accessed', { ip: clientIP });
      handleModelsEndpoint(res);
      return;
    }

    // Handle GET / (health check)
    if (req.method === 'GET' && (req.url === '/' || req.url === '/api/chat')) {
      res.statusCode = 200;
      sendSuccess(res, {
        status: 'ok',
        service: 'AI Chatbot API',
        timestamp: getTimestamp(),
      });
      return;
    }

    // Only allow POST for chat endpoint
    if (req.method !== 'POST') {
      return sendError(res, 405, 'Method not allowed', {
        allowed: ['POST', 'OPTIONS', 'GET'],
      });
    }

    // Check if it's a chat request
    if (req.url !== '/api/chat') {
      return sendError(res, 404, 'Endpoint not found');
    }

    // Check HTTPS
    if (!isSecureRequest(req)) {
      log('Insecure request attempted', { ip: clientIP });
      return sendError(res, 400, 'HTTPS required');
    }

    // Check rate limit
    const rateLimit = checkRateLimit(clientIP);
    if (!rateLimit.allowed) {
      log('Rate limit exceeded', { ip: clientIP });
      return sendError(res, 429, 'Too many requests', {
        retryAfter: rateLimit.retryAfter,
      });
    }

    // Check API key
    if (!GROQ_API_KEY) {
      log('GROQ_API_KEY not configured', { ip: clientIP });
      return sendError(res, 500, 'API key not configured');
    }

    // Parse request body
    let body = '';
    await new Promise((resolve, reject) => {
      req.on('data', (chunk) => {
        body += chunk.toString();
      });
      req.on('end', resolve);
      req.on('error', reject);

      // Timeout after 10 seconds
      setTimeout(() => reject(new Error('Request timeout')), 10000);
    });

    let parsedBody;
    try {
      parsedBody = JSON.parse(body);
    } catch (e) {
      log('Invalid JSON in request body', { ip: clientIP });
      return sendError(res, 400, 'Invalid request body', {
        details: 'Request body must be valid JSON',
      });
    }

    // Validate request
    const validation = validateRequest(parsedBody);
    if (!validation.valid) {
      log('Request validation failed', { ip: clientIP, error: validation.error });
      return sendError(res, 400, validation.error);
    }

    const { message, history, model } = validation.data;

    log('Processing chat request', {
      ip: clientIP,
      model,
      messageLength: message.length,
      historyLength: history.length,
    });

    // Build messages array with system prompt
    const messages = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...history,
      { role: 'user', content: message },
    ];

    // Create Groq API request payload
    const groqPayload = {
      model,
      messages,
      temperature: 0.7,
      max_tokens: AVAILABLE_MODELS[model].maxTokens,
      top_p: 0.95,
      stream: false,
    };

    // Call Groq API
    const groqResponse = await callGroqAPI(groqPayload, GROQ_API_KEY);

    // Handle API errors
    if (groqResponse.status !== 200) {
      const errorData = groqResponse.data;

      log('Groq API error', {
        ip: clientIP,
        status: groqResponse.status,
        error: errorData.error?.message || 'Unknown error',
      });

      if (groqResponse.status === 429) {
        const retryAfter = groqResponse.headers['retry-after'] || 60;
        return sendError(res, 429, 'Groq API rate limit exceeded', {
          retryAfter,
        });
      }

      if (groqResponse.status === 401) {
        return sendError(res, 500, 'API authentication failed');
      }

      return sendError(res, 500, 'Groq API error', {
        details: errorData.error?.message || 'Unknown error',
      });
    }

    // Extract response data
    const { choices, usage, model: usedModel } = groqResponse.data;

    if (!choices || choices.length === 0) {
      log('Empty response from Groq API', { ip: clientIP });
      return sendError(res, 500, 'Invalid response from AI API');
    }

    const aiMessage = choices[0].message.content;

    log('Chat response generated', {
      ip: clientIP,
      model: usedModel,
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
    });

    // Return success response
    sendSuccess(res, {
      response: aiMessage,
      model: usedModel,
      usage: {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
      },
      timestamp: getTimestamp(),
    });
  } catch (error) {
    const clientIP = getClientIP(req);
    log('Unhandled error', {
      ip: clientIP,
      error: error.message,
      stack: NODE_ENV === 'development' ? error.stack : undefined,
    });

    return sendError(res, 500, 'Internal server error', {
      details: NODE_ENV === 'development' ? error.message : undefined,
    });
  }
}

// ===========================
// Export Handler
// ===========================

module.exports = handler;
