import { RealtimeAgent, RealtimeSession } from '@openai/agents-realtime';

console.log('=== MAIN.TS LOADED ===');

// Get UI elements
const loginForm = document.getElementById('login-form');
const wordInputForm = document.getElementById('word-input-form');
const agentInterface = document.getElementById('agent-interface');
const accessCodeInput = document.getElementById('access-code-input') as HTMLInputElement;
const connectButton = document.getElementById('connect-button') as HTMLButtonElement;
const wordListInput = document.getElementById('word-list-input') as HTMLTextAreaElement;
const startStudyButton = document.getElementById('start-study-button') as HTMLButtonElement;
const loginError = document.getElementById('login-error');
const wordError = document.getElementById('word-error');
const statusDiv = document.getElementById('status');

let sessionToken: string | null = null;

// Load prompt template
async function loadPromptTemplate(): Promise<string> {
  console.log('Loading prompt template...');
  const response = await fetch('/prompt2.md');
  if (!response.ok) {
    throw new Error(`Failed to load prompt template: ${response.status}`);
  }
  const text = await response.text();
  console.log('Prompt template loaded, length:', text.length);
  console.log('First 200 chars:', text.substring(0, 200));
  return text;
}

// Update status display
function setStatus(message: string, type: 'connecting' | 'connected' | 'error') {
  if (!statusDiv) return;
  statusDiv.textContent = message;
  statusDiv.className = `status ${type}`;
}

// Show error message
function showError(element: HTMLElement | null, message: string) {
  if (!element) return;
  element.textContent = message;
}

// Clear error message
function clearError(element: HTMLElement | null) {
  if (!element) return;
  element.textContent = '';
}

// Fetch token with access code
async function getToken(accessCode: string): Promise<string> {
  const apiUrl = window.location.hostname === 'localhost' 
    ? 'http://localhost:3000/api/token'
    : '/api/token';
  
  const response = await fetch(apiUrl, { 
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ accessCode })
  });
  
  const responseText = await response.text();
  
  if (!response.ok) {
    const errorData = JSON.parse(responseText);
    if (response.status === 403) {
      throw new Error('Invalid access code');
    }
    throw new Error(errorData.details || errorData.error || 'Failed to connect to server');
  }
  
  const data = JSON.parse(responseText);
  
  if (data.client_secret && data.client_secret.value) {
    return data.client_secret.value;
  } else if (data.value) {
    return data.value;
  } else {
    throw new Error('Server returned invalid response - no token found');
  }
}

// Fetch word definitions
async function getDefinitions(words: string[]): Promise<Record<string, string>> {
  console.log('Looking up definitions for:', words);
  
  const apiUrl = window.location.hostname === 'localhost' 
    ? 'http://localhost:3000/api/definitions'
    : '/api/definitions';
  
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ words })
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to look up definitions');
  }
  
  const data = await response.json();
  console.log('Definitions received:', data.vocabDict);
  return data.vocabDict;
}

// Start the voice agent with vocabulary
async function startVoiceAgent(vocabDict: Record<string, string>) {
  if (!sessionToken) {
    throw new Error('No session token available');
  }
  
  console.log('Starting voice agent with vocabulary:', vocabDict);
  
  // Load prompt template and fill in vocabulary
  const promptTemplate = await loadPromptTemplate();
  const vocabJson = JSON.stringify(vocabDict, null, 2);
  const instructions = promptTemplate.replace(
    '{{VOCABULARY_LIST}}',
    vocabJson
  );
  
  console.log('===== FINAL INSTRUCTIONS =====');
  console.log(instructions);
  console.log('===== END INSTRUCTIONS =====');
  
  // Verify the substitution happened
  if (instructions.includes('{{VOCABULARY_LIST}}')) {
    console.error('WARNING: Template variable was not replaced!');
  }
  
  // Create and connect agent
  console.log('Creating RealtimeAgent with name: Orðaflóð');
  const agent = new RealtimeAgent({
    name: 'Orðaflóð',
    instructions: instructions,
  });

  console.log('Creating RealtimeSession...');
  const session = new RealtimeSession(agent, {
    model: 'gpt-realtime',
  });
  
  console.log('Connecting session...');
  await session.connect({ apiKey: sessionToken });
  
  console.log('✅ Session connected successfully!');
  setStatus('🎤 Connected! Start talking.', 'connected');
}

// Handle connect button click (Step 1: Login)
if (connectButton) {
  connectButton.addEventListener('click', async () => {
    const accessCode = accessCodeInput?.value.trim();
    
    if (!accessCode) {
      showError(loginError, 'Please enter an access code');
      return;
    }
    
    clearError(loginError);
    connectButton.disabled = true;
    connectButton.textContent = 'Connecting...';
    
    try {
      sessionToken = await getToken(accessCode);
      console.log('✅ Token obtained');
      
      // Move to word input form
      if (loginForm) loginForm.style.display = 'none';
      if (wordInputForm) wordInputForm.style.display = 'flex';
      
      // Reset button
      connectButton.disabled = false;
      connectButton.textContent = 'Connect';
      
    } catch (error) {
      connectButton.disabled = false;
      connectButton.textContent = 'Connect';
      
      if (error instanceof Error) {
        showError(loginError, error.message);
      } else {
        showError(loginError, 'Failed to connect. Please try again.');
      }
    }
  });
}

// Handle start study button click (Step 2: Process words and start agent)
if (startStudyButton) {
  startStudyButton.addEventListener('click', async () => {
    const wordListText = wordListInput?.value.trim();
    
    if (!wordListText) {
      showError(wordError, 'Please enter at least one word');
      return;
    }
    
    // Parse words (one per line)
    const words = wordListText
      .split('\n')
      .map(w => w.trim())
      .filter(w => w.length > 0);
    
    console.log('Parsed words:', words);
    
    if (words.length === 0) {
      showError(wordError, 'Please enter at least one word');
      return;
    }
    
    clearError(wordError);
    startStudyButton.disabled = true;
    startStudyButton.textContent = 'Looking up definitions...';
    
    try {
      // Look up definitions
      const vocabDict = await getDefinitions(words);
      
      // Switch to agent interface
      if (wordInputForm) wordInputForm.style.display = 'none';
      if (agentInterface) agentInterface.style.display = 'block';
      setStatus('Preparing your study session...', 'connecting');
      
      // Start voice agent with vocabulary
      await startVoiceAgent(vocabDict);
      
    } catch (error) {
      startStudyButton.disabled = false;
      startStudyButton.textContent = 'Start Studying';
      
      console.error('Error starting study session:', error);
      
      if (error instanceof Error) {
        showError(wordError, error.message);
      } else {
        showError(wordError, 'Failed to start session. Please try again.');
      }
    }
  });
}

// Allow Enter key to submit on login form
if (accessCodeInput) {
  accessCodeInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      connectButton?.click();
    }
  });
}

console.log('Script setup complete');