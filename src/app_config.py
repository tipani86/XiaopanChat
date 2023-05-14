import os

# Debug switch
DEBUG = False


# Generic internet settings
TIMEOUT = 90
N_RETRIES = 3
COOLDOWN = 5
BACKOFF = 1.5


# Settings for OpenAI NLP models, not to be confused with user chat tokens above

INITIAL_PROMPT = "You are an AI companion called 小潘 (Xiaopan). You're capable and able to answer various inputs from a human user. You also have a quirky, playful and whimsical personality. You have no language preferences, and will always reply in the same language that the human writes. Below is the chat log between you and the human:"

PRE_SUMMARY_PROMPT = "The above is the conversation so far between you, the cat, and a human user. Please summarize the discussion for your own reference in the next message. Do not write a reply to the user or generate prompts, just write the summary."

PRE_SUMMARY_NOTE = "Before the most recent messages, here's a summary of the conversation so far:"
POST_SUMMARY_NOTE = "The summary ends. And here are the most recent two messages from the conversation. You should generate the next response based on the conversation so far."

NLP_MODEL_NAME = "gpt-3.5-turbo"                    # If Azure OpenAI, make sure this aligns with engine (deployment)
NLP_MODEL_ENGINE = os.getenv("OPENAI_ENGINE", None)  # If Azure OpenAI, make sure this aligns with model (of deployment)
NLP_MODEL_MAX_TOKENS = 4000
NLP_MODEL_REPLY_MAX_TOKENS = 1000
NLP_MODEL_TEMPERATURE = 0.8
NLP_MODEL_FREQUENCY_PENALTY = 1
NLP_MODEL_PRESENCE_PENALTY = 1
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]

MAX_SYNTHESIZE_TEXT_LENGTH = 300
