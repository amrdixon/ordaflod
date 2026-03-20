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


def lookup_words_to_file(words: list[str], output_fp: str, list_name: str = "list1"):
    """Look up a list of words using MeriamWebsterLookup and write definitions to a JSON file.

    Args:
        words: List of words to look up.
        output_fp: Path to the output JSON file.
        list_name: Key name for the word list in the output file (default "list1").
    """
    vocab = {}
    for word in words:
        try:
            lookup = MeriamWebsterLookup(word)
            vocab[word] = lookup.get_definition()
        except RuntimeError as e:
            print(f"Warning: Could not look up '{word}': {e}")
            vocab[word] = "Definition not found."

    output = {list_name: vocab}
    with open(output_fp, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Wrote {len(vocab)} definitions to {output_fp}")






