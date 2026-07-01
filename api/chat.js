/**
 * Vercel Serverless AI Chatbot API
 * Simple and efficient implementation
 */

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get data from request body (Vercel automatically parses JSON)
    const { message, history = [], model = 'llama-3.3-70b-versatile' } = req.body;

    // Validate input
    if (!message || message.trim() === '') {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Check API key
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      console.error('GROQ_API_KEY not configured');
      return res.status(500).json({ 
        error: 'API key not configured',
        message: 'Please add GROQ_API_KEY in Vercel Environment Variables'
      });
    }

    // Build messages array
    const messages = [
      {
        role: 'system',
        content: 'You are a helpful AI assistant. Provide clear, accurate, and helpful responses. When writing code, use proper formatting.'
      },
      ...history,
      { role: 'user', content: message }
    ];

    // Call Groq API using native fetch
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: model,
        messages: messages,
        temperature: 0.7,
        max_tokens: 4096,
        top_p: 0.95,
        stream: false
      })
    });

    // Handle API errors
    if (!response.ok) {
      const errorData = await response.json();
      console.error('Groq API Error:', errorData);
      
      if (response.status === 429) {
        return res.status(429).json({ 
          error: 'Rate limit exceeded',
          retryAfter: errorData.error?.message || 'Please wait before trying again'
        });
      }
      
      return res.status(response.status).json({ 
        error: 'Groq API error',
        details: errorData.error?.message || 'Unknown error'
      });
    }

    const data = await response.json();

    // Validate response
    if (!data.choices || !data.choices[0] || !data.choices[0].message) {
      return res.status(500).json({ error: 'Invalid response from Groq API' });
    }

    // Return success
    return res.status(200).json({
      response: data.choices[0].message.content,
      model: data.model,
      usage: data.usage
    });

  } catch (error) {
    console.error('Server error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error.message
    });
  }
}
