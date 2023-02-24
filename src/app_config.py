# Debug switch
DEBUG = False


# App settings
DEMO_HISTORY_LIMIT = 5
NEW_USER_FREE_TOKENS = 15
FREE_TOKENS_PER_REFERRAL = 10


# Products on sale
# SET_NAMES = ["小白", "进阶"]  # Partial catalogue with two higher end products hidden
SET_NAMES = ["小白", "进阶", "王者", "钻石"]  # Full catalogue


# Azure Table names
USERS_TABLE = "users"
ORDERS_TABLE = "orders"
CONSUMPTION_TABLE = "tokenUse"


# Generic internet settings
TIMEOUT = 15
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5


# Settings for OpenAI NLP models, not to be confused with user chat tokens above

INITIAL_PROMPT = "You are an AI companion called 小潘 (Xiaopan). You're capable and able to answer various inputs from a human user. You also have a quirky, playful and whimsical personality. You have no language preferences, and will always reply in the same language that the human writes. Below is the chat log between you and the human:"

PRE_SUMMARY_PROMPT = "The above is the conversation so far between you, the AI assistant, and a human user. Please summarize the topics discussed for your own reference. Remember, do not write a direct reply to the user."

PRE_SUMMARY_NOTE = "Before the actual log, here's a summary of the conversation so far:"
POST_SUMMARY_NOTE = "The summary ends. And here are the last two messages from the conversation before your reply:"

NLP_MODEL_NAME = "text-davinci-003"
NLP_MODEL_MAX_TOKENS = 4000
NLP_MODEL_REPLY_MAX_TOKENS = 1500
NLP_MODEL_TEMPERATURE = 0.8
NLP_MODEL_FREQUENCY_PENALTY = 1.0
NLP_MODEL_PRESENCE_PENALTY = 1.0
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]


# Some helper javascript snippets
clear_input_script = """
<script>
    // Clear input value
    const streamlitDoc = window.parent.document
    // Find the target input element
    const inputs = Array.from(streamlitDoc.getElementsByTagName('input'))
    // Find all the inputs with aria-label '请输入:' and clear their value
    for (let i = 0; i < inputs.length; i++) {
        if (inputs[i].ariaLabel === '请输入:') {
            inputs[i].value = ''
        }
    }
    /*
    const input = inputs.find(input => input.ariaLabel === '请输入:')
        // Clear the input value if it has value or the value is other than ''
        if (input.value || input.value !== '') {
            input.value = ''
        }
    */
</script>
"""

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


# Settings for payment validation
ORDER_VALIDATION_KEYS = [
    ('body', 'title'),
    ('fee', 'money'),
    ('no', 'no'),
    ('pay_type', 'paytype'),
    ('remark', 'remark'),
]
