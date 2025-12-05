export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Parse body
  let body = req.body;
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch (e) {
      return res.status(400).json({ error: 'Invalid JSON' });
    }
  }
  
  const { accessCode } = body;
  
 // Valid access codes
  const VALID_CODES = new Set([
  'demo-banana-potato',
  'pokemon-spensa-banana',
  ]);
  
  // Check access code
  if (!accessCode || !VALID_CODES.has(accessCode)) {
    return res.status(403).json({ error: 'Invalid access code' });
  }
  
  // Check if API key exists
  if (!process.env.OPENAI_API_KEY) {
    console.error('OPENAI_API_KEY environment variable is not set!');
    return res.status(500).json({ error: 'Server configuration error: API key missing' });
  }
  
  try {
    console.log('Calling OpenAI API...');
    
    const response = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        session: { type: 'realtime', model: 'gpt-realtime' }
      })
    });
    
    console.log('OpenAI response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('OpenAI API error:', errorText);
      return res.status(500).json({ 
        error: 'Failed to generate token from OpenAI',
        details: errorText 
      });
    }
    
    const data = await response.json();
    console.log('OpenAI response structure:', JSON.stringify(data).substring(0, 200));
    
    // NEW: Handle the updated response format
    // The token is now directly in data.value, not data.client_secret.value
    if (!data.value) {
      console.error('Unexpected OpenAI response - missing value field:', data);
      return res.status(500).json({ 
        error: 'Unexpected response from OpenAI',
        details: 'Missing value field in response'
      });
    }
    
    // Return in the format the frontend expects (for backward compatibility)
    res.status(200).json({
      client_secret: {
        value: data.value,
        expires_at: data.expires_at
      }
    });
    
  } catch (error) {
    console.error('Error calling OpenAI:', error);
    return res.status(500).json({ 
      error: 'Failed to generate token',
      details: error.message 
    });
  }
}