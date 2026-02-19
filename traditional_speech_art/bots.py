from litellm import completion
from util import MeriamWebsterLookup, load_prompt
import asyncio
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import threading
import queue
from dotenv import load_dotenv
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

## set ENV variables
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

#Define fopath for whiseper model
WHISPER_MODEL_FP = "./models/whisper/base.pt"
# import certifi
# import ssl
# import urllib.request

# ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

class VocabStudyBot:

    def __init__(self, word_list: list[str]=None):
            
            if word_list is None:
                word_list = []
            if len(word_list)==0:
                raise ValueError("Word list cannot be empty")
            
            if not all(isinstance(word, str) for word in word_list):
                raise TypeError("All items in vocab list must be strings.")
            
            self.word_list = word_list
            self.vocab_dict = {}

            #Get definitions for each word in the list
            for word in word_list:
                 lookup = MeriamWebsterLookup(word)
                 definition = lookup.get_definition()
                 self.vocab_dict[word] = definition

            #Initialize conversation history
            self.conversation_history = []

            #Create studybot prompt
            self.prompt = load_prompt(CONFIG['prompt_fp'], self.vocab_dict)

            # Add system prompt to conversation history
            self.conversation_history.append({
                "role": "system", 
                "content": self.prompt
            })
            self.conversation_history.append(
                {'role': 'user',
                 'content': 'Please continue.'
                 })

            # Get initial greeting from the model
            initial_response = self._get_model_response()
            self.last_response = initial_response

    def _get_model_response(self):

        response = completion(
              model=CONFIG['llm_model'],
              messages=self.conversation_history
        )
        assistant_message = response['choices'][0]['message']['content']

         #Add assistant message to conversation history
        self.conversation_history.append({
              "role": "assistant",
              "content": assistant_message})
        
        return assistant_message

    def send_message(self, user_message: str) -> str:
        """Send a user message and get the assistant's response."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get and return response
        response = self._get_model_response()
        self.last_response = response
        return response
    
    def get_last_response(self) -> str:
        """Get the last response from the assistant."""
        return self.last_response
        

class SpeechVocabBot:
    """Speech-enabled wrapper with real-time VAD and interruption."""
    
    def __init__(self, word_list: list[str]):
        self.bot = VocabStudyBot(word_list)
        
        # Initialize STT model (Whisper)
        print("Loading speech recognition model...")
        self.stt_model = whisper.load_model(WHISPER_MODEL_FP)
        
        # Audio recording settings        
        self.RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_DURATION = 0.1  # 100ms chunks
        self.SILENCE_THRESHOLD = 0.01  # Amplitude threshold for silence
        self.SILENCE_DURATION = 1.5  # Seconds of silence before stopping
        self.MIN_SPEECH_DURATION = 0.5  # Minimum speech duration
        
        self.recording_path = "temp_recording.wav"
        self.session_active = False
        self.is_speaking = False  # Track if bot is speaking
        self.interrupt_flag = False  # Flag for interruptions
        
    def is_silence(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk is silence based on amplitude."""
        return np.abs(audio_chunk).mean() < self.SILENCE_THRESHOLD
    
    def detect_speech_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect if there's speech in the audio chunk."""
        return np.abs(audio_chunk).mean() > self.SILENCE_THRESHOLD
    
    def record_until_silence(self) -> str:
        """Record audio until silence is detected."""
        print("🎤 Listening... (speak now)")
        
        audio_queue = queue.Queue()
        recording = []
        silence_chunks = 0
        speech_detected = False
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream."""
            if status:
                print(f"Status: {status}")
            audio_queue.put(indata.copy())
        
        # Start audio stream
        stream = sd.InputStream(
            callback=audio_callback,
            channels=self.CHANNELS,
            samplerate=self.RATE,
            blocksize=int(self.RATE * self.CHUNK_DURATION)
        )
        
        with stream:
            while self.session_active:
                try:
                    # Get audio chunk
                    chunk = audio_queue.get(timeout=0.5)
                    recording.append(chunk)
                    
                    # Check for speech activity
                    if self.detect_speech_activity(chunk):
                        speech_detected = True
                        silence_chunks = 0
                    elif speech_detected:
                        # Count silence after speech started
                        silence_chunks += 1
                        
                        # Check if enough silence to stop
                        silence_duration = silence_chunks * self.CHUNK_DURATION
                        if silence_duration >= self.SILENCE_DURATION:
                            print("⏹️  Silence detected, processing...")
                            break
                    
                    # Check for minimum speech duration
                    total_duration = len(recording) * self.CHUNK_DURATION
                    
                except queue.Empty:
                    continue
        
        # Check if we got enough speech
        if not speech_detected:
            print("⚠️  No speech detected")
            return None
        
        # Combine all chunks
        full_recording = np.concatenate(recording, axis=0)
        
        # Save to file
        sf.write(self.recording_path, full_recording, self.RATE)
        
        return self.recording_path
    
    def monitor_for_interruption(self):
        """Monitor for user interruption while bot is speaking."""
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time, status):
            if self.is_speaking:
                audio_queue.put(indata.copy())
        
        stream = sd.InputStream(
            callback=audio_callback,
            channels=self.CHANNELS,
            samplerate=self.RATE,
            blocksize=int(self.RATE * self.CHUNK_DURATION)
        )
        
        with stream:
            while self.is_speaking and not self.interrupt_flag:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    
                    # Check if user is speaking
                    if self.detect_speech_activity(chunk):
                        print("\n⚠️  Interruption detected!")
                        self.interrupt_flag = True
                        self.tts_engine.stop()
                        break
                        
                except queue.Empty:
                    continue
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Convert speech to text using Whisper."""
        if audio_file_path is None:
            return ""
        
        result = self.stt_model.transcribe(audio_file_path)
        return result["text"]
    
    def speak(self, text: str):
        """Convert text to speech using macOS native 'say' command."""
        import subprocess
        
        self.is_speaking = True
        self.interrupt_flag = False
        
        # Start monitoring for interruptions in a separate thread
        monitor_thread = threading.Thread(target=self.monitor_for_interruption)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            print(f"🔊 Speaking...")
            # Use macOS native 'say' command)
            subprocess.run(['say', text], check=True)
            print("✓ Finished speaking")
                
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False
            monitor_thread.join(timeout=1)
    
    def process_voice_input(self) -> str:
        """Record with VAD, transcribe, and respond to voice input."""
        # 1. Record until silence detected
        audio_file = self.record_until_silence()
        
        if audio_file is None:
            return None
        
        # 2. Transcribe
        user_text = self.transcribe_audio(audio_file)
        
        if not user_text.strip():
            print("⚠️  No speech recognized")
            return None
            
        print(f"\n💬 You: {user_text}")
        
        # 3. Get bot response
        bot_response = self.bot.send_message(user_text)
        print(f"🤖 Bot: {bot_response}\n")
        
        # 4. Speak response (can be interrupted)
        self.speak(bot_response)
        
        # 5. If interrupted, handle the interruption
        if self.interrupt_flag:
            print("\n🎤 Listening for your interruption...")
            # The user was already speaking, so we should capture that
            return self.process_voice_input()
        
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        return bot_response
    
    def process_text_input(self, user_text: str) -> str:
        """Process text input without speech."""
        bot_response = self.bot.send_message(user_text)
        print(f"🤖 Bot: {bot_response}\n")
        self.speak(bot_response)
        return bot_response
    
    def start_session(self):
        """Start the study session with initial greeting."""
        self.session_active = True
        initial_greeting = self.bot.get_last_response()
        print(f"\n🤖 Bot: {initial_greeting}\n")
        self.speak(initial_greeting)
    
    def end_session(self):
        """End the study session."""
        self.session_active = False
        self.is_speaking = False
        farewell = "Thank you for studying with me today. Keep practicing!"
        print(f"\n🤖 Bot: {farewell}")
        self.speak(farewell)
        
        # Cleanup any temp files
        if os.path.exists(self.recording_path):
            os.remove(self.recording_path)