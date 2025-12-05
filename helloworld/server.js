import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('dist'));

// Valid access codes (add as many as you want)
const VALID_CODES = new Set([
  'demo-banana-potato',
  'pokemon-spensa-banana',
  // Add more codes for other people
]);

// Endpoint to generate tokens - requires valid access code
app.post('/api/token', async (req, res) => {
  const { accessCode } = req.body;
  
  // Check if access code is valid
  if (!accessCode || !VALID_CODES.has(accessCode)) {
    return res.status(403).json({ error: 'Invalid access code' });
  }
  
  try {
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
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate token' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});