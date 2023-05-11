import os
import time
import json
import base64
import aiohttp
import traceback
import subprocess
from PIL import Image
import streamlit as st
from api_utils import *
from app_config import *
from transformers import AutoTokenizer
import streamlit.components.v1 as components
import azure.cognitiveservices.speech as speechsdk

# Set global variables

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check environment variables

errors = []
for key in [
    "OPENAI_API_KEY",                   # For OpenAI APIs
    "AZURE_SPEECH_KEY",                 # For Azure Speech APIs
    "RAPID_API_KEY",                    # For Rapid APIs
]:
    if key not in os.environ:
        errors.append(f"Please set the {key} environment variable.")
if len(errors) > 0:
    st.error("\n".join(errors))
    st.stop()


### FUNCTION DEFINITIONS ###


@st.cache_resource(show_spinner=False)
def get_synthesizer(config: dict):
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv('AZURE_SPEECH_KEY'),
        region=config['region']
    )
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    return speechsdk.SpeechSynthesizer(speech_config, audio_config=None)


@st.cache_data(show_spinner=False)
def get_local_img(file_path: str) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_favicon(file_path: str):
    # Load a byte image and return its favicon
    return Image.open(file_path)


@st.cache_data(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2", low_cpu_mem_usage=True)


@st.cache_data(show_spinner=False)
def get_json(file_path: str) -> dict:
    if not os.path.isfile(file_path):
        st.error(f"File {file_path} not found.")
        st.stop()
    # Load a json file and return its content
    return json.load(open(file_path, "r"))


@st.cache_data(show_spinner=False)
def get_js() -> str:
    # Read javascript web trackers code from script.js file
    with open(os.path.join(ROOT_DIR, "src", "script.js"), "r") as f:
        return f"""
            <audio id="voicePlayer" autoplay #voicePlayer></audio>
            <script type='text/javascript'>{f.read()}</script>
        """


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(ROOT_DIR, "src", "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"


def get_chat_message(
    contents: str = "",
    align: str = "left"
) -> str:
    # Formats the message in an chat fashion (user right, reply left)
    div_class = "AI-line"
    color = "rgb(240, 242, 246)"
    file_path = os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png")
    src = f"data:image/gif;base64,{get_local_img(file_path)}"
    if align == "right":
        div_class = "human-line"
        color = "rgb(165, 239, 127)"
        if "USER" in st.session_state:
            src = st.session_state.USER.avatar_url
        else:
            file_path = os.path.join(ROOT_DIR, "src", "assets", "user_icon.png")
            src = f"data:image/gif;base64,{get_local_img(file_path)}"
    icon_code = f"<img class='chat-icon' src='{src}' width=32 height=32 alt='avatar'>"
    formatted_contents = f"""
    <div class="{div_class}">
        {icon_code}
        <div class="chat-bubble" style="background: {color};">
        &#8203;{contents}
        </div>
    </div>
    """
    return formatted_contents


async def main(human_prompt: str) -> dict:
    res = {'status': 0, 'message': "Success"}
    try:
        # Strip the prompt of any potentially harmful html/js injections
        human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;")

        # Update both chat log and the model memory
        st.session_state.LOG.append("Human: " + human_prompt)
        st.session_state.MEMORY.append({'role': "user", 'content': human_prompt})

        # Clear the input box after human_prompt is read
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            line = st.session_state.LOG[-1]
            contents = line.split("Human: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

            reply_box = st.empty()
            reply_box.markdown(get_chat_message(), unsafe_allow_html=True)

            # This is one of those small three-dot animations to indicate the bot is "writing"
            writing_animation = st.empty()
            file_path = os.path.join(ROOT_DIR, "src", "assets", "loading.gif")
            writing_animation.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<img src='data:image/gif;base64,{get_local_img(file_path)}' width=30 height=10>", unsafe_allow_html=True)

            # Main process
            async with aiohttp.ClientSession() as httpclient:
                prompt_res = await generate_prompt_from_memory_async(
                    httpclient,
                    TOKENIZER,
                    st.session_state.MEMORY,
                    os.getenv("OPENAI_API_KEY")
                )

                if DEBUG:
                    with st.sidebar:
                        st.write("prompt_res: ")
                        st.json(prompt_res, expanded=False)

                if prompt_res['status'] != 0:
                    res['status'] = prompt_res['status']
                    res['message'] = prompt_res['message']
                    return res
            
            # Refresh memory from the prompt response
            st.session_state.MEMORY = prompt_res['data']['messages']

            # Call the OpenAI ChatGPT API for final result
            reply_text = ""
            async for chunk in await openai.ChatCompletion.acreate(
                model=NLP_MODEL_NAME,
                messages=st.session_state.MEMORY,
                temperature=NLP_MODEL_TEMPERATURE,
                max_tokens=NLP_MODEL_REPLY_MAX_TOKENS,
                frequency_penalty=NLP_MODEL_FREQUENCY_PENALTY,
                presence_penalty=NLP_MODEL_PRESENCE_PENALTY,
                stop=NLP_MODEL_STOP_WORDS,
                stream=True,
                timeout=TIMEOUT,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content", None)
                if content is not None:
                    reply_text += content

                    # Sanitizing output
                    if reply_text.startswith("AI: "):
                        reply_text = reply_text.split("AI: ", 1)[1]

                    # Continuously render the reply as it comes in
                    reply_box.markdown(get_chat_message(reply_text), unsafe_allow_html=True)

        # Clear the writing animation
        writing_animation.empty()

        # Update the chat LOG and memories with the actual response
        st.session_state.LOG.append(f"AI: {reply_text}")
        st.session_state.MEMORY.append({'role': "assistant", 'content': reply_text})

    except:
        res['status'] = 2
        res['message'] = traceback.format_exc()

    return res


### INITIALIZE AND LOAD ###


# Initialize page config
favicon = get_favicon(os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png"))
st.set_page_config(
    page_title="Chat Assistant",
    page_icon=favicon,
)


# Load Azure Speech configuration and validate its data
speech_cfg_path = os.path.join(ROOT_DIR, "cfg", "azure_speech.json")
speech_cfg = get_json(speech_cfg_path)
for key in ["voice", "rate", "pitch"]:
    if key not in speech_cfg:
        st.error(f"Key {key} not found in Azure Speech configuration file.")
        st.stop()

# Get query parameters
query_params = st.experimental_get_query_params()
if "debug" in query_params and query_params["debug"][0].lower() == "true":
    st.session_state.DEBUG = True

if "DEBUG" in st.session_state and st.session_state.DEBUG:
    DEBUG = True


# Initialize some useful class instances
with st.spinner("Initializing App..."):
    TOKENIZER = get_tokenizer()  # First time after deployment takes a few seconds
azure_synthesizer = get_synthesizer(speech_cfg)


build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()
else:
    try:
        build_date = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(f"Failed to get git commit hash: {e}")


### MAIN STREAMLIT UI STARTS HERE ###


# Define main layout
st.title("Hello!")
st.subheader("I am your AI assistant. How can I help you today?")
st.subheader("")
chat_box = st.container()
st.write("")
prompt_box = st.empty()
footer = st.container()

if DEBUG:
    with st.sidebar:
        st.markdown("<p><small>Debug tools:</small></p>", unsafe_allow_html=True)
        if st.button("Clear cache"):
            st.cache_data.clear()

    # Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)


# Load JS code
components.html(get_js(), height=0, width=0)


# Initialize/maintain a chat log and chat memory in Streamlit's session state
# Log is the actual line by line chat, while memory is limited by model's maximum token context length
if "MEMORY" not in st.session_state:
    st.session_state.MEMORY = [{'role': "system", 'content': INITIAL_PROMPT}]
    st.session_state.LOG = [INITIAL_PROMPT]


# Render footer
# with footer:
#     st.info("免责声明：聊天机器人基于海量互联网文本训练的大型语言模型，仅供娱乐。小潘AI不对信息的准确性、完整性、及时性等承担任何保证或责任。", icon="ℹ️")
#     st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)


with chat_box:
    for i, line in enumerate(st.session_state.LOG[1:]):
        # For AI response
        if line.startswith("AI: "):
            contents = line.split("AI: ")[1]
            st.markdown(get_chat_message(contents), unsafe_allow_html=True)

        # For human prompts
        if line.startswith("Human: "):
            contents = line.split("Human: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

# Define an input box for human prompts
with prompt_box:
    human_prompt = st.text_input("Enter:", value="", key=f"text_input_{len(st.session_state.LOG)}")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(human_prompt) > 0:

    run_res = asyncio.run(main(human_prompt))
    if run_res['status'] == 0 and not DEBUG:
        st.experimental_rerun()

    else:
        if run_res['status'] != 0:
            st.error(run_res['message'])
        with prompt_box:
            if st.button("Show text input box"):
                st.experimental_rerun()
