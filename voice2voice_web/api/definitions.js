export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  let body = req.body;
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch (e) {
      return res.status(400).json({ error: 'Invalid JSON' });
    }
  }
  
  const { words } = body;
  
  if (!words || !Array.isArray(words) || words.length === 0) {
    return res.status(400).json({ error: 'Invalid word list' });
  }
  
  if (!process.env.MW_API_KEY) {
    return res.status(500).json({ error: 'Dictionary API key not configured' });
  }
  
  try {
    const vocabDict = {};
    
    // Look up each word
    for (const word of words) {
      const cleanWord = word.trim().toLowerCase();
      if (!cleanWord) continue;
      
      try {
        const response = await fetch(
          `https://dictionaryapi.com/api/v3/references/collegiate/json/${cleanWord}?key=${process.env.MW_API_KEY}`,
          { timeout: 5000 }
        );
        
        if (!response.ok) {
          vocabDict[cleanWord] = "Definition not available";
          continue;
        }
        
        const data = await response.json();
        
        // Extract definition
        if (data && data[0] && data[0].shortdef && data[0].shortdef[0]) {
          vocabDict[cleanWord] = data[0].shortdef[0];
        } else {
          vocabDict[cleanWord] = "Definition not found";
        }
      } catch (error) {
        console.error(`Error looking up ${cleanWord}:`, error);
        vocabDict[cleanWord] = "Definition not available";
      }
    }
    
    res.status(200).json({ vocabDict });
    
  } catch (error) {
    console.error('Error processing words:', error);
    res.status(500).json({ error: 'Failed to process word list' });
  }
}