from bots import SpeechVocabBot
import sys

def main():
    # Create bot with vocabulary list
    vocab_list = ["ephemeral"]
    speech_bot = SpeechVocabBot(vocab_list)

    # Start the session
    print("\n=== Vocabulary Study Session ===")
    print("The bot will listen continuously and detect when you stop speaking.")
    print("You can also interrupt the bot while it's speaking!")
    print("\nCommands:")
    print("  - Just start speaking (the bot is always listening)")
    print("  - Type 'quit' or 'exit' to end session")
    print("  - Type a message to chat via text\n")
    
    speech_bot.start_session()

    # Main conversation loop
    while True:
        try:
            # Check if user wants to type instead
            print("\n[Listening... or type a command/message]: ")
            
            # Non-blocking check for keyboard input
            import select
            import sys
            
            # Give user 1 second to start typing, otherwise start voice mode
            if select.select([sys.stdin], [], [], 1.0)[0]:
                user_input = input().strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n✓ Ending study session. Great work!")
                    speech_bot.end_session()
                    break
                
                # Text mode
                elif user_input:
                    speech_bot.process_text_input(user_input)
                    continue
            
            # Voice mode - automatically listens and detects silence
            response = speech_bot.process_voice_input()
            
            if response is None:
                print("No input detected, listening again...")
                
        except KeyboardInterrupt:
            print("\n\n✓ Session interrupted. Goodbye!")
            speech_bot.end_session()
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing session...\n")

if __name__ == "__main__":
    main()