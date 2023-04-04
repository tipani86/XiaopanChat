# Debug switch
DEBUG = False


# Generic internet settings
TIMEOUT = 90
N_RETRIES = 3
COOLDOWN = 5
BACKOFF = 1.5


# Settings for OpenAI NLP models, not to be confused with user chat tokens above

INITIAL_PROMPT = "You are an AI companion called 小潘 (Xiaopan). You're capable and able to answer various inputs from a human user. You also have a quirky, playful and whimsical personality. You have no language preferences, and will always reply in the same language that the human writes. Below is the chat log between you and the human:"

PRE_SUMMARY_PROMPT = "The above is the conversation so far between you, the AI assistant, and a human user. Please summarize the topics discussed for your own reference. Remember, do not write a direct reply to the user."

PRE_SUMMARY_NOTE = "Before the actual log, here's a summary of the conversation so far:"
POST_SUMMARY_NOTE = "The summary ends. And here are the last two messages from the conversation before your reply:"

NLP_MODEL_NAME = "gpt-4"
NLP_MODEL_MAX_TOKENS = 8000
NLP_MODEL_REPLY_MAX_TOKENS = 1500
NLP_MODEL_TEMPERATURE = 0.8
NLP_MODEL_FREQUENCY_PENALTY = 1.0
NLP_MODEL_PRESENCE_PENALTY = 1.0
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]

MAX_SYNTHESIZE_TEXT_LENGTH = 300

expand_sidebar_script = """
<script>
    // Expand the sidebar
    const streamlitDoc = window.parent.document
    const buttons = streamlitDoc.getElementsByClassName('css-9s5bis edgvbvh3')
    // Normally there are three buttons (so we press the index 1),
    // but on Streamlit hosted service there are five buttons, so we press index 3)
    if (buttons.length === 3) {
        buttons[1].click()
    } else if (buttons.length === 5) {
        buttons[3].click()
    }
</script>
"""
