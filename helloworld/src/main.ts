// src/main.ts
import { RealtimeAgent, RealtimeSession } from '@openai/agents-realtime';

console.log('=== MAIN.TS LOADED ===');

// Get UI elements
const loginForm = document.getElementById('login-form');
const agentInterface = document.getElementById('agent-interface');
const accessCodeInput = document.getElementById('access-code-input') as HTMLInputElement;
const connectButton = document.getElementById('connect-button') as HTMLButtonElement;
const errorMessage = document.getElementById('error-message');
const statusDiv = document.getElementById('status');

console.log('UI elements found:', { loginForm, agentInterface, accessCodeInput, connectButton });

// Update status display
function setStatus(message: string, type: 'connecting' | 'connected' | 'error') {
  if (!statusDiv) {
    console.error('statusDiv not found!');
    return;
  }
  statusDiv.textContent = message;
  statusDiv.className = `status ${type}`;
}

// Show error message
function showError(message: string) {
  if (!errorMessage) {
    console.error('errorMessage element not found!');
    return;
  }
  errorMessage.textContent = message;
}

// Clear error message
function clearError() {
  if (!errorMessage) return;
  errorMessage.textContent = '';
}

// Fetch token with access code
async function getToken(accessCode: string): Promise<string> {
  console.log('Fetching token for:', accessCode);
  
  // Use Vercel API URL when testing locally
  const apiUrl = window.location.hostname === 'localhost' 
    ? 'https://your-project-name.vercel.app/api/token'  // Your actual Vercel URL
    : '/api/token';
  
  console.log('API URL:', apiUrl);
  
  const response = await fetch(apiUrl, { 
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ accessCode })
  });
  
  console.log('Response status:', response.status);
  
  // Get the raw response text first
  const responseText = await response.text();
  console.log('Raw response:', responseText);
  
  if (!response.ok) {
    let errorData;
    try {
      errorData = JSON.parse(responseText);
    } catch (e) {
      console.error('Failed to parse error response as JSON');
      throw new Error(`Server error: ${response.status}`);
    }
    
    console.error('Error response:', errorData);
    
    if (response.status === 403) {
      throw new Error('Invalid access code');
    }
    
    throw new Error(errorData.details || errorData.error || 'Failed to connect to server');
  }
  
  // Parse the successful response
  let data;
  try {
    data = JSON.parse(responseText);
  } catch (e) {
    console.error('Failed to parse success response as JSON');
    throw new Error('Server returned invalid response');
  }
  
  console.log('Parsed data:', data);
  
  // Handle both old and new response formats
  let token;
  
  if (data.client_secret && data.client_secret.value) {
    // Old format: { client_secret: { value: "ek_..." } }
    token = data.client_secret.value;
    console.log('Using client_secret.value format');
  } else if (data.value) {
    // New format: { value: "ek_..." }
    token = data.value;
    console.log('Using direct value format');
  } else {
    console.error('Invalid response structure:', data);
    throw new Error('Server returned invalid response - no token found');
  }
  
  console.log('Token found:', token.substring(0, 20) + '...');
  return token;
}

// Handle connect button click
if (connectButton) {
  console.log('Adding click listener to connect button');
  
  connectButton.addEventListener('click', async () => {
    console.log('Connect button clicked!');
    
    const accessCode = accessCodeInput?.value.trim();
    console.log('Access code entered:', accessCode);
    
    // Validate input
    if (!accessCode) {
      showError('Please enter an access code');
      return;
    }
    
    clearError();
    connectButton.disabled = true;
    connectButton.textContent = 'Connecting...';
    
    try {
      console.log('Getting token...');
      const token = await getToken(accessCode);
      console.log('Token received (first 20 chars):', token.substring(0, 20));
      
      // Switch to agent interface
      if (loginForm) loginForm.style.display = 'none';
      if (agentInterface) agentInterface.style.display = 'block';
      setStatus('Connecting to voice agent...', 'connecting');
      
      // Create and connect agent
      console.log('Creating agent...');
      const agent = new RealtimeAgent({
        name: 'Assistant',
        instructions: 'You are a helpful assistant.',
      });

      console.log('Creating session...');
      const session = new RealtimeSession(agent, {
        model: 'gpt-realtime',
      });
      
      console.log('Connecting session...');
      await session.connect({ apiKey: token });
      
      console.log('Connected successfully!');
      setStatus('🎤 Connected! Start talking.', 'connected');
      
    } catch (error) {
      console.error('Connection error:', error);
      
      // Show error and reset form
      connectButton.disabled = false;
      connectButton.textContent = 'Connect';
      
      if (error instanceof Error) {
        showError(error.message);
      } else {
        showError('Failed to connect. Please try again.');
      }
    }
  });
} else {
  console.error('Connect button not found!');
}

// Allow Enter key to submit
if (accessCodeInput) {
  console.log('Adding keypress listener to input');
  
  accessCodeInput.addEventListener('keypress', (e) => {
    console.log('Key pressed:', e.key);
    if (e.key === 'Enter') {
      console.log('Enter key pressed, clicking button');
      connectButton?.click();
    }
  });
} else {
  console.error('Access code input not found!');
}

console.log('Script setup complete');