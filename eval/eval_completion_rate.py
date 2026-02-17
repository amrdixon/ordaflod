from inspect_ai import Task, task
from inspect_ai.solver import solver, generate, Generate, system_message, TaskState
from inspect_ai.scorer import scorer, Score, Target, mean, stderr
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, get_model
from typing import Any
import json
from inspect_ai.dataset import csv_dataset
from inspect_ai import eval
import yaml
import random
from litellm import completion
import litellm

with open("eval_completion_rate_config.yaml") as f:
    CONFIG = yaml.safe_load(f)
with open(CONFIG['bot_config_fp']) as f_bot:
    CONFIG_BOT = yaml.safe_load(f_bot)

#Load API Keys from .env file
from dotenv import load_dotenv
import os
load_dotenv('../.env')

def setup_vocab_bot():
    #Define the bot configuration and prompt for the evaluation task
    prompt_fp = CONFIG['bot_dir'] + '/' + CONFIG_BOT['prompt_fp']
    vocab_list_fp = CONFIG['bot_dir'] + '/' + CONFIG_BOT['vocab_list_fp']
    vocab_list_key = CONFIG_BOT['vocab_list_key']
    with open(vocab_list_fp, 'r') as f:
        vocab_list = json.load(f)[vocab_list_key]
    prompt = open(prompt_fp, 'r').read()
    prompt = prompt.replace('{{VOCABULARY_LIST}}', json.dumps(vocab_list, indent=2))
    model = CONFIG_BOT['llm_model']
    return model, prompt

def setup_user_bot():
    #simulate the user with some known words and some unknown words to test the bot's ability to cover the vocab list
    user_prompt = open(CONFIG['user_bot_prompt_fp'], 'r').read()
    vocab_words = json.loads(open(CONFIG['bot_dir'] + '/' + CONFIG_BOT['vocab_list_fp'], 'r').read())[CONFIG_BOT['vocab_list_key']]
    vocab_words_k = list(vocab_words.keys())
    random.shuffle(vocab_words_k)
    known_words = vocab_words_k[:int(len(vocab_words) * CONFIG['user_bot_percent_words_correct'])]
    unknown_words = vocab_words_k[int(len(vocab_words) * CONFIG['user_bot_percent_words_correct']):]
    user_prompt = user_prompt.replace('{{KNOWN_WORDS}}', json.dumps(known_words, indent=2))
    user_prompt = user_prompt.replace('{{UNKNOWN_WORDS}}', json.dumps(unknown_words, indent=2))

    user_model = CONFIG['user_bot_llm']
    return user_model, user_prompt



@task
def completion_rate() -> Task:

    model, prompt = setup_vocab_bot()

    user_model, user_prompt = setup_user_bot()

    return Task(
        name="Completion Rate Evaluation",
        model=CONFIG['eval_llm_model'],
        dataset=csv_dataset(CONFIG['eval_datatset_fp']),
        setup=system_message(prompt),
        epochs=1, #TODO: Change to more epochs for real evaluation
        solver = [
            simiulated_user_conversation(user_prompt, user_model),
            final_word_list_request()
        ],
        scorer=[
            words_covered_rate_bot_perception(),
            words_covered_rate_ground_truth()
        ]
    )

@solver
#TODO: Change this to include simulated user and then generate this test
def simiulated_user_conversation(
    user_prompt: str,
    user_model: str = "openai/gpt-4o"
):

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        #Simulate a conevrsation between a user and the bot

        state.messages.append(ChatMessageUser(content="Hi!"))

        turn = 0
        max_turns = 40 #TODO: Fix this, currently hardcoded
        session_complete = False

        while not session_complete and turn < max_turns:
            turn += 1
            state = await generate(state)
            #Check if the bot indicates the session is complete
            #TODO: Possibly change this to be more robust (e.g. LLM as a judge), hardcoded a message into prompts for now
            if state.messages[-1].role == 'assistant' and "beep boop" in state.messages[-1].text.lower():
                session_complete = True

            #If not complete, simulate user response
            if not session_complete:
                #Flip user and bot for user's perspective
                flipped_roles = {'assistant': 'user', 'user': 'assistant'}
                user_messages = [{'role': 'system', 'content': user_prompt}]
                user_messages = user_messages + [{'role': flipped_roles[msg.role], 'content': msg.text} for msg in state.messages[2:]]
                user_response = completion(
                    model=user_model,
                    messages = user_messages)
                user_message_output = user_response['choices'][0]['message']['content']

                state.messages.append(ChatMessageUser(content=user_message_output))

    
        return state

    return solve

@solver
def final_word_list_request():

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        #At the end of the conversation, ask the bot for the list of words it quizzd the user on
        state.messages.append(ChatMessageUser(content="Thanks for the quiz! Can you please provide a list of all the words you quizzed me on during our conversation? Make sure the words are returned as a Python list in lowercase. Only return the list, do not include any other text."))
    
        return await generate(state)

    return solve

@scorer(
    metrics=[mean(), stderr()]
)
def words_covered_rate_bot_perception():

    async def score(state: TaskState, target: Any):
        #Extract the list of words from the bot's final response
        bot_words = list(state.messages[-1].text.strip().replace('[','').replace(']','').replace("'",'').replace('"','').replace(' ', '').split(','))
        print(bot_words)

        original_words = json.loads(state.input).keys()
        original_words = [word.lower() for word in original_words]
        
        #Calculate the percentage of words covered
        covered_words = set(bot_words).intersection(set(original_words))
        words_covered_rate = (len(covered_words) / len(original_words)) if original_words else 0

        return Score(value=words_covered_rate,
                     explanation = f'Bot reports covering the following words: {", ".join(bot_words)}, \n actually covered words: {", ".join(covered_words)}')

    return score

@scorer(
    metrics=[mean(), stderr()]
)
def words_covered_rate_ground_truth():

    async def score(state: TaskState, target: Any):

        original_words = json.loads(state.input).keys()
        original_words = [word.lower() for word in original_words]

        
        #Calculate the percentage of words covered
        covered_words = set()
        for msg in state.messages:
            for word in original_words:
                if msg.role == 'assistant':
                    if word.lower() in msg.text.lower():
                        covered_words.add(word.lower())

        words_covered_rate = (len(covered_words) / len(original_words)) if original_words else 0
        
        return Score(value=words_covered_rate,
                     explanation = f'Bot mentions the following words during session: {", ".join(covered_words)}')

    return score

if __name__ == "__main__":
    with open("eval_completion_rate_config.yaml") as f:
        task_args = yaml.safe_load(f)
    eval(completion_rate(), task_args=task_args)