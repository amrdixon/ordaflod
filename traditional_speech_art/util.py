from dotenv import load_dotenv
import os

load_dotenv()
mw_api_key = os.getenv("MW_API_KEY")

import requests
import json

def load_prompt(prompt_fp, vocab_dict):
    with open(prompt_fp, 'r') as f:
        prompt = f.read()
    return prompt.replace('{{VOCABULARY_LIST}}', json.dumps(vocab_dict, indent=2))

class MeriamWebsterLookup:

    def __init__(self, word:str):
        self.word = word
        try:
            response = requests.get(f"https://dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={mw_api_key}", timeout=5)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            raise RuntimeError("The request to the dictionary API timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"An error occurred while requesting the dictionary API: {e}")
        
        self.data = data
        
        
    def get_definition(self):
        """Return the definition of the word."""
        return self.data[0]['shortdef'][0] if self.data and 'shortdef' in self.data[0] else "Definition not found."



    



