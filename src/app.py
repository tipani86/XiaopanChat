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
import streamlit.components.v1 as components
from transformers import AutoTokenizer
import azure.cognitiveservices.speech as speechsdk

# Set global variables

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check environment variables

errors = []
for key in [
    "OPENAI_API_KEY", "OPENAI_ORG_ID",  # For OpenAI APIs
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


async def main(human_prompt):
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
                if prompt_res['status'] != 0:
                    st.error(prompt_res['message'])
                    st.stop()

                chatbot_reply_res = await get_chatbot_reply_data_async(
                    httpclient,
                    prompt_res['data']['messages'],
                    os.getenv("OPENAI_API_KEY")
                )
                if chatbot_reply_res['status'] != 0:
                    st.error(chatbot_reply_res['message'])
                    st.stop()

                reply_text = chatbot_reply_res['data']['reply_text']
                languages = chatbot_reply_res['data']['language']

            audio_play_time, audio_chars = 0, 0
            for item in languages:
                if item['language'] == "zh":
                    # Synthesize the response and play it as audio
                    audio_play_time, b64 = synthesize_text(reply_text, speech_cfg, azure_synthesizer, speechsdk)
                    audio_chars = len(reply_text)
                    if audio_play_time > 0 and len(b64) > 0:
                        # This part works in conjunction with the initialized script.js and puts the audio data into the audio player
                        components.html(f"""<script>
                            window.parent.document.voicePlayer.src = "data:audio/mp3;base64,{b64}";
                            window.parent.document.voicePlayer.pause();
                            window.parent.document.voicePlayer.currentTime = 0;
                            // Wait for 0.5 seconds and play
                            setTimeout(() => {{
                                window.parent.document.voicePlayer.play();
                            }}, 1000);
                        </script>""", height=0, width=0)
                        audio_play_time += 1.0  # To account for slight delay in the beginning of the audio
                    break

            # Loop so that reply_text gets revealed one character at a time
            if audio_chars > 0:
                pause_per_char = 0.85 * audio_play_time / audio_chars  # 0.85 because we want the text to appear a bit faster than the audio
            else:
                pause_per_char = 0.02
            tic = time.time()
            for i in range(len(reply_text)):
                with reply_box.container():
                    st.markdown(get_chat_message(reply_text[:i+1]), unsafe_allow_html=True)
                    await asyncio.sleep(pause_per_char)
            toc = time.time()

            # Pause for the remaining time, if any
            await asyncio.sleep(max(0, audio_play_time - (toc - tic)))

            # Stop and clear the audio stream from voicePlayer
            if audio_play_time > 0 and len(b64) > 0:
                components.html(f"""<script>
                    window.parent.document.voicePlayer.pause();
                    window.parent.document.voicePlayer.src = "";
                </script>""", height=0, width=0)

        # Clear the writing animation
        writing_animation.empty()

        # Update the chat LOG and memories with the actual response
        st.session_state.LOG.append(f"AI: {reply_text}")
        st.session_state.MEMORY.append({'role': "assistant", 'content': reply_text})

    except:
        st.error(traceback.format_exc())

        # Await for 10 seconds
        await asyncio.sleep(10)

        st.stop()

    finally:
        st.experimental_rerun()


### INITIALIZE AND LOAD ###


# Initialize page config
favicon = get_favicon(os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png"))
st.set_page_config(
    page_title="小潘AI",
    page_icon=favicon,
    initial_sidebar_state="collapsed",
)


# Load Azure Speech configuration and validate its data
speech_cfg_path = os.path.join(ROOT_DIR, "cfg", "azure_speech.json")
speech_cfg = get_json(speech_cfg_path)
for key in ["voice", "rate", "pitch"]:
    if key not in speech_cfg:
        st.error(f"Key {key} not found in Azure Speech configuration file.")
        st.stop()


# Initialize some useful class instances
with st.spinner("应用首次初始化中..."):
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
st.title("你好，")
st.subheader("我是小潘AI，来跟我说点什么吧！")
st.subheader("")
chat_box = st.container()
st.write("")
prompt_box = st.empty()
footer = st.container()


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
with footer:
    st.info("免责声明：聊天机器人基于海量互联网文本训练的大型语言模型，仅供娱乐。小潘AI不对信息的准确性、完整性、及时性等承担任何保证或责任。", icon="ℹ️")
    st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)
    if DEBUG:
        st.markdown("<p><small>Debug tools:</small></p>", unsafe_allow_html=True)
        if st.button("Clear cache"):
            st.cache_data.clear()

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
    human_prompt = st.text_input("请输入:", value="", key=f"text_input_{len(st.session_state.LOG)}")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(human_prompt) > 0:

    asyncio.run(main(human_prompt))
