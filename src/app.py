import os
import re
import math
import time
import json
import base64
import openai
import qrcode
import random
import requests
import datetime
import calendar
import humanize
import traceback
import subprocess
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components
from utils import AzureTableOp, User, generate_event_id, use_consumables
from transformers import AutoTokenizer
import azure.cognitiveservices.speech as speechsdk

DEBUG = True

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check environment variables

for key in [
    "OPENAI_API_KEY", "OPENAI_ORG_ID",  # For OpenAI APIs
    "AZURE_STORAGE_CONNECTION_STRING",  # For Table Storage
    "AZURE_SPEECH_KEY",                 # For Azure Speech APIs
    "WX_LOGIN_SECRET",                  # WeChat Login
    "SEVENPAY_PID", "SEVENPAY_PKEY"     # Payment Gateway
]:
    if key not in os.environ:
        st.error(f"Please set the {key} environment variable.")
        st.stop()

# Set global variables

_t = humanize.i18n.activate("zh_CN")    # Initialize humanize time output in Simplified Chinese

DEMO_HISTORY_LIMIT = 5
NEW_USER_FREE_TOKENS = 15
FREE_TOKENS_PER_REFERRAL = 10

SET_NAMES = ["小白", "进阶"]
# SET_NAMES = ["小白", "进阶", "王者", "钻石"]

USERS_TABLE = "users"
ORDERS_TABLE = "orders"
TOKENUSE_TABLE = "tokenuse"
if DEBUG:
    for table_name in [USERS_TABLE, ORDERS_TABLE, TOKENUSE_TABLE]:
        table_name += "Test"

TIMEOUT = 15
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5

# Below are settings for OpenAI NLP models, not to be confused with user chat tokens above
NLP_MODEL_NAME = "text-davinci-003"
NLP_MODEL_MAX_TOKENS = 4000
NLP_MODEL_REPLY_MAX_TOKENS = 1500
NLP_MODEL_TEMPERATURE = 0.7
NLP_MODEL_FREQUENCY_PENALTY = 1.0
NLP_MODEL_PRESENCE_PENALTY = 1.0
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]

OUTPUT_OPTIONS = {
    "text": "文本",
    "text_audio": "文本+语音",
}


build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()
else:
    try:
        build_date = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(f"Failed to get git commit hash: {e}")

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

# Function definitions


@st.cache_resource(show_spinner=False)
def get_synthesizer(config: dict):
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv('AZURE_SPEECH_KEY'),
        region=config['region']
    )
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    return speechsdk.SpeechSynthesizer(speech_config, audio_config=None)


@st.cache_resource(show_spinner=False)
def get_table_op():
    return AzureTableOp()


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


# @st.cache_data(show_spinner=False)
def get_json(file_path: str) -> dict:
    if not os.path.isfile(file_path):
        st.error(f"File {file_path} not found.")
        st.stop()
    # Load a json file and return its content
    return json.load(open(file_path, "r"))


# @st.cache_data(show_spinner=False)
def get_js() -> str:
    # Read javascript web trackers code from script.js file
    with open(os.path.join(ROOT_DIR, "src", "script.js"), "r") as f:
        return f"""
            <audio id="voicePlayer" autoplay #voicePlayer></audio>
            <script type='text/javascript'>{f.read()}</script>
        """


# @st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(ROOT_DIR, "src", "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"


def warm_up_api_server():
    res = {'status': 0, 'msg': "Success"}
    # Warm up Xiaopan API server with retry, backoff etc.
    for i in range(N_RETRIES * 2):
        try:
            r = requests.get("https://xiaopan-chat-api.azurewebsites.net/", timeout=TIMEOUT)
            if r.status_code == 200:
                return res
        except Exception as e:
            time.sleep(COOLDOWN * BACKOFF ** i)
    res['status'] = 2
    res['msg'] = f"Failed to warm up API server!"
    return res


def get_chat_message(
    contents: str = "",
    align: str = "left"
) -> str:
    div_class = "AI-line"
    color = "rgb(240, 242, 246)"
    file_path = os.path.join(ROOT_DIR, "src", "AI_icon.png")
    src = f"data:image/gif;base64,{get_local_img(file_path)}"
    if align == "right":
        div_class = "human-line"
        color = "rgb(165, 239, 127)"
        if "USER" in st.session_state:
            src = st.session_state.USER.avatar_url
        else:
            file_path = os.path.join(ROOT_DIR, "src", "user_icon.png")
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


def update_header() -> None:
    if "NEW_USER" in st.session_state and st.session_state.NEW_USER:
        header_text = f"<font color=red>欢迎加入小潘AI</font> <b>{st.session_state.USER.nickname}</b> ！ "
    else:
        header_text = f"欢迎回来 <b>{st.session_state.USER.nickname}</b> ！ "
    if st.session_state.USER.n_tokens <= 0:
        header_text += "<font color=red>你的聊天币已经全部用完了</font>"
    else:
        header_text += f"你还有 <b>{st.session_state.USER.n_tokens}</b> 枚聊天币可以用哦"
    if st.session_state.USER.n_tokens < 10:
        header_text += "， 请立即前往用户中心充值"
    st.markdown(f"""
    <div class="human-line">
        <div><small>{header_text}</small></div>
    </div>
    """, unsafe_allow_html=True)


def update_sidebar() -> None:
    with st.container():
        st.title("用户中心")
        st.subheader(st.session_state.USER.nickname)
        # st.caption(f"<small>({st.session_state.USER.user_id})</small>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<img class='chat-icon' src='{st.session_state.USER.avatar_url}' width=64 height=64 alt='avatar'>", unsafe_allow_html=True)
        with col2:
            st.metric("**剩余聊天币**", f"{st.session_state.USER.n_tokens}枚")

        cat_expander = st.expander("聊天币充值")
        with cat_expander:
            tabs = st.tabs(SET_NAMES)
            for i, (tab, set_name_short) in enumerate(zip(tabs, SET_NAMES)):
                with tab:
                    set_name = f"{set_name_short}套餐"
                    display_name = set_name
                    col1, col2 = st.columns([5, 4])
                    with col1:
                        # Always show price with two decimals even if it's integer
                        price = f"{catalogue[set_name]['price']:.2f}"
                        if catalogue[set_name]["sale_price"]:
                            display_name += f" (:red[~~¥{catalogue[set_name]['price']:.2f}~~])"
                            price = f"{catalogue[set_name]['sale_price']:.2f}"
                        st.metric(
                            f"**{display_name}**",
                            f"¥{price}",
                            f"{catalogue[set_name]['amount']}{catalogue[set_name]['unit']}{catalogue[set_name]['product']}"
                        )
                    with col2:
                        st.write("**支付码：**")
                        vars()[f"generate_payment{i+1}"] = st.button("生成", key=f"generate_payment_{set_name}_{len(st.session_state.LOG)}")
            st.info("**聊天币**数量按照完整的消息来回为一枚，如：10枚聊天币等于20条消息（发送+回复）", icon="ℹ️")

        payment_code_placeholder = st.empty()
        for i, set_name_short in enumerate(SET_NAMES):
            var_name = f"generate_payment{i+1}"
            if var_name in vars() and vars()[var_name]:
                set_name = f"{set_name_short}套餐"
                with payment_code_placeholder.container():
                    st.write(f"**{set_name}** button pressed")
                    refresh = st.button("支付成功后点此按钮刷新页面", key=f"refresh_{len(st.session_state.LOG)}")
                if refresh:
                    st.experimental_rerun()

        st.write("")
        with st.expander("最近10次登录历史"):
            # Calculate and display user's IP history and time
            d = datetime.datetime.now()
            timestamp_now = calendar.timegm(d.timetuple())

            # We need to read the IP history in reverse order
            formatted_ip_history = []
            for timestamp, ip in reversed(st.session_state.USER.ip_history):
                formatted_ip_history.append(f"{ip}, {humanize.naturaltime(datetime.timedelta(seconds=timestamp_now - timestamp))}")
            st.caption("<br>".join(formatted_ip_history), unsafe_allow_html=True)


def render_login_popup(
    popup_container,
    table_op
) -> None:
    with popup_container:
        st.write("")
        # Step 1: Display QR code
        popup_content = st.empty()
        with popup_content:
            popup_container = st.container()
            with popup_container:
                with st.spinner("正在获取登录二维码..."):
                    # Call the WeChat login service with retry, cooldown and backoff
                    url = os.path.join(
                        wx_login_cfg['endpoint'],
                        f"tempUserId?secretKey={os.getenv('WX_LOGIN_SECRET')}"
                    )
                    for i in range(N_RETRIES):
                        try:
                            r = requests.get(url, timeout=TIMEOUT)
                            r.raise_for_status()
                            break
                        except Exception as e:
                            if i == N_RETRIES - 1:
                                login_popup.close()
                                st.error(f"小潘AI出错了: {e}")
                                st.stop()
                            else:
                                time.sleep(COOLDOWN * BACKOFF ** i)

                    # Parse results
                    if r.status_code != 200:
                        login_popup.close()
                        st.error(f"微信登录出错了: {r.text}")
                        st.stop()

                    wx_login_res = r.json()
                    if wx_login_res['errcode'] != 0:
                        login_popup.close()
                        st.error(f"微信登录出错了: {wx_login_res['message']}")
                        st.stop()

                    # (Re-)initiate an entry for this ID in the temp table
                    table_name = "tempUserIds"
                    if DEBUG:
                        table_name += "Test"
                    temp_user_id = wx_login_res['data']['tempUserId']
                    entity = {
                        'PartitionKey': "wx_user",
                        'RowKey': temp_user_id,
                        'data': None
                    }
                    table_res = table_op.update_entities(entity, table_name)

                    if table_res['status'] != 0:
                        login_popup.close()
                        st.error(f"无法更新用户表: {table_res['message']}")
                        st.stop()

                    # Finally, generate the QR code and show it to the user
                    qr_img = qrcode.make(wx_login_res['data']['qrCodeReturnUrl'])
                    qr_img = qr_img.resize((200, 200))
                    qr_img = np.asarray(qr_img) * 255   # The values are 0/1, we need to make it visible

                st.image(qr_img, caption="请用手机微信扫二维码登录", output_format="PNG")
                st.caption(":red[手机用户请长按复制二维码，粘贴到任意微信聊天框后再长按进行扫码登录]")

                # Poll the login status every 3 seconds and close the popup when login is successful, or timeout after 60 seconds.
                for i in range(20):
                    time.sleep(3)

                    query_filter = f"PartitionKey eq @channel and RowKey eq @user_id"
                    select = None
                    parameters = {'channel': "wx_user", 'user_id': temp_user_id}

                    table_res = table_op.query_entities(query_filter, select, parameters, table_name)
                    if table_res['status'] != 0:
                        login_popup.close()
                        st.error(f"无法获取用户表: {table_res['message']}")
                        st.stop()

                    entity = table_res['data'][0]

                    if 'data' in entity and entity['data'] is not None:
                        break
                if 'data' not in entity or entity['data'] is None:
                    login_popup.close()
                    st.error("登录超时，请重试")
                    st.stop()

        popup_content.empty()
        time.sleep(0.1)

        # Step 2: QR scanned, perform actual login and user initialization
        with popup_content:
            popup_container = st.container()
            with popup_container:
                with st.spinner("扫码成功，正在登录..."):
                    d = datetime.datetime.now()
                    timestamp = calendar.timegm(d.timetuple())
                    user_id = entity['user_id']
                    user_data = json.loads(entity['data'])
                    user_data['timestamp'] = timestamp

                    # Build user object from temporary login data. The object is easier to manipulate later on.
                    st.session_state.USER = User(
                        channel="wx_user",
                        user_id=user_id,
                        db_op=table_op,
                        users_table=USERS_TABLE,
                        orders_table=ORDERS_TABLE,
                        tokenuse_table=TOKENUSE_TABLE
                    )

                    # Delete the temp entry from the table
                    table_name = "tempUserIds"
                    if DEBUG:
                        table_name += "Test"
                    table_res = table_op.delete_entity(entity, table_name)

                    # Pull out the sidebar now that the user has logged in
                    with sidebar.container():
                        st.subheader("登录中...")
                    components.html(expand_sidebar_script, height=0, width=0)

                    # Create/update full user data
                    action_res = st.session_state.USER.sync_from_db()
                    if action_res['status'] == 3:   # User not found, new user
                        action_res = st.session_state.USER.initialize_on_db(
                            user_data,
                            NEW_USER_FREE_TOKENS
                        )
                        if action_res['status'] != 0:
                            login_popup.close()
                            st.error(f"无法初始化用户信息: {action_res['message']}")
                            st.stop()
                        if "NEW_USER" not in st.session_state:
                            st.session_state.NEW_USER = True
                    else:
                        # Normal login, we update ip history from user_data
                        action_res = st.session_state.USER.update_ip_history(user_data)
                        if action_res['status'] != 0:
                            login_popup.close()
                            st.error(f"无法更新IP地址: {action_res['message']}")
                            st.stop()


def generate_prompt_from_memory():
    # Check whether tokenized model memory so far + max reply length exceeds the max possible tokens
    memory_str = "\n".join(st.session_state.MEMORY)
    memory_tokens = tokenizer.tokenize(memory_str)
    tokens_used = 0  # NLP tokens (for OpenAI)
    if len(memory_tokens) + NLP_MODEL_REPLY_MAX_TOKENS > NLP_MODEL_MAX_TOKENS:
        # Strategy: We keep the first item of memory (original prompt), and last three items
        # (last AI message, human's reply, and the 'AI:' prompt) intact, and summarize the middle part
        summarizable_memory = st.session_state.MEMORY[1:-3]

        # We write a new prompt asking the model to summarize this middle part
        summarizable_memory = summarizable_memory + [
            "The above is the conversation so far between you, the AI assistant, and a human user. Please summarize the topics discussed for your own reference. Remember, do not write a direct reply to the user."
        ]
        summarizable_str = "\n".join(summarizable_memory)
        summarizable_tokens = tokenizer.tokenize(summarizable_str)
        tokens_used += len(summarizable_tokens)
        # Check whether the summarizable tokens + 75% of the reply length exceeds the max possible tokens.
        # If so, adjust down to 50% of the reply length and try again, lastly if even 25% of the reply tokens still exceed, call an error.
        for ratio in [0.75, 0.5, 0.25]:
            if len(summarizable_tokens) + int(NLP_MODEL_REPLY_MAX_TOKENS * ratio) <= NLP_MODEL_MAX_TOKENS:
                # Call the OpenAI API with retry and all that shebang
                for i in range(N_RETRIES):
                    try:
                        response = openai.Completion.create(
                            model=NLP_MODEL_NAME,
                            prompt=summarizable_str,
                            temperature=NLP_MODEL_TEMPERATURE,
                            max_tokens=int(NLP_MODEL_REPLY_MAX_TOKENS * ratio),
                            frequency_penalty=NLP_MODEL_FREQUENCY_PENALTY,
                            presence_penalty=NLP_MODEL_PRESENCE_PENALTY,
                            stop=NLP_MODEL_STOP_WORDS,
                        )
                        break
                    except Exception as e:
                        if i == N_RETRIES - 1:
                            st.error(f"小潘AI出错了: {e}")
                            st.stop()
                        else:
                            time.sleep(COOLDOWN * BACKOFF ** i)
                summary_text = response["choices"][0]["text"].strip()
                tokens_used += len(tokenizer.tokenize(summary_text))

                # Re-build memory so it consists of the original prompt, a note that a summary follows,
                # the actual summary, a second note that the last two conversation items follow,
                # then the last three items from the original memory
                new_memory = st.session_state.MEMORY[:1] + [
                    "Before the actual log, here's a summary of the conversation so far:"
                ] + [summary_text] + [
                    "The summary ends. And here are the last two messages from the conversation before your reply:"
                ] + st.session_state.MEMORY[-3:]

                st.session_state.MEMORY = new_memory

                # Re-generate prompt from new memory
                new_prompt = "\n".join(st.session_state.MEMORY)
                tokens_used += len(tokenizer.tokenize(new_prompt))

                if DEBUG:
                    st.info(f"Summarization triggered. New prompt:\n\n{new_prompt}")

                return new_prompt, tokens_used

        st.error("小潘AI出错了: 你的消息太长了，小潘AI无法处理")
        st.stop()

    # No need to summarize, just return the original prompt
    tokens_used += len(memory_tokens)
    return memory_str, tokens_used


def synthesize_text(
    text: str,
    config: dict,
    synthesizer,
) -> None:
    # Clean up the text so it doesn't contain weird tokens
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(CLEANR, '', text)
    # Add speaking style if configured
    if 'style' in config and config['style'] is not None:
        ssml_string = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                <voice name="{config['voice'] if 'voice' in config else 'zh-CN-XiaoxiaoNeural'}">
                <mstts:express-as style='{config['style']}'>
                    <prosody rate="{config['rate'] if 'rate' in config else 1.0}" pitch="{config['pitch'] if 'pitch' in config else '0%'}">
                        {text}
                    </prosody>
                </mstts:express-as>
                </voice>
            </speak>
        """
    else:
        ssml_string = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{config['voice'] if 'voice' in config else 'zh-CN-XiaoxiaoNeural'}">
                    <prosody rate="{config['rate'] if 'rate' in config else 1.0}" pitch="{config['pitch'] if 'pitch' in config else '0%'}">
                        {text}
                    </prosody>
                </voice>
            </speak>
        """
    result = synthesizer.speak_ssml_async(ssml_string).get()
    if DEBUG:
        print(result)
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        length = result.audio_duration.total_seconds()
        b64 = base64.b64encode(result.audio_data).decode()
        # This part works in conjunction with the initialized script.js and puts the audio data into the audio player
        components.html(f"""<script>window.parent.document.voicePlayer.src = "data:audio/mp3;base64,{b64}";</script>""", height=0, width=0)
        return length
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
    return 0


# Load Azure Speech configuration and validate its data
speech_cfg_path = os.path.join(ROOT_DIR, "cfg", "azure_speech.json")
speech_cfg = get_json(speech_cfg_path)
for key in ["voice", "rate", "pitch"]:
    if key not in speech_cfg:
        st.error(f"Key {key} not found in Azure Speech configuration file.")
        st.stop()

# Load WeChat login configuration and validate its data
wx_login_cfg_path = os.path.join(ROOT_DIR, "cfg", "wx_login.json")
wx_login_cfg = get_json(wx_login_cfg_path)
if 'endpoint' not in wx_login_cfg:
    st.error("WeChat login endpoint not found in configuration file.")
    st.stop()

# Load product catalogue configuration and validate its data
catalogue_path = os.path.join(ROOT_DIR, "cfg", "catalogue.json")
catalogue = get_json(catalogue_path)
for set_name in catalogue:
    for key in ["price", "product", "amount"]:
        if key not in catalogue[set_name]:
            st.error(f"Set {set_name} is missing key {key} in catalogue configuration file.")
            st.stop()

# OpenAI API settings
openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize page config
favicon = get_favicon(os.path.join(ROOT_DIR, "src", "AI_icon.png"))
st.set_page_config(
    page_title="小潘AI",
    page_icon=favicon,
    initial_sidebar_state="collapsed",
)

# Initialize some useful class instances
with st.spinner("应用首次初始化中..."):
    tokenizer = get_tokenizer()

azure_table_op = get_table_op()
azure_synthesizer = get_synthesizer(speech_cfg)

# Warm-up the API server when the user accesses the site
# (Azure tends to spin them down after some inactivity)
if "WARM_UP" not in st.session_state or not st.session_state.WARM_UP:
    with st.spinner("小潘AI正在预热中..."):
        warm_up_res = warm_up_api_server()

    if warm_up_res['status'] != 0:
        st.error(f"小潘AI启动失败！{warm_up_res['msg']}")
        st.stop()
    st.session_state.WARM_UP = True

### MAIN STREAMLIT UI STARTS HERE ###

# Define main layout
header = st.empty()
st.title("你好，")
st.subheader("我是小潘AI，来跟我说点什么吧！")
st.subheader("")
chat_box = st.container()
st.write("")
prompt_box = st.empty()
footer = st.container()

# Initialize sidebar placeholder
with st.sidebar:
    sidebar = st.empty()

# Initialize login popup
login_popup = Modal(title=None, key="login_popup", padding=40, max_width=200)

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# Load JS code
components.html(get_js(), height=0, width=0)

# # Read browser query params and save them in session state
# query_params = st.experimental_get_query_params()
# if DEBUG:
#     st.write(f"`Query params: {query_params}`")

# Initialize/maintain a chat log and chat memory in Streamlit's session state
# Log is the actual line by line chat, while memory is limited by model's maximum token context length
init_prompt = "You are an AI assistant called 小潘 (Xiaopan). You're very capable, able answer various messages from a human user and provide helpful replies. You can add HTML your responses, for example when asked to list something. You have no language preferences, and will always reply in the same language that the human writes. Below is the chat log between you and the human:"
if "MEMORY" not in st.session_state:
    st.session_state.MEMORY = [init_prompt]
    st.session_state.LOG = [init_prompt]

# Render header and sidebar depending on whether the user is logged in or not
if "USER" not in st.session_state:
    with sidebar.container():
        st.caption("登录后可以查看用户信息")
    with header.container():
        col1, col2 = st.columns([1, 9])
        with col1:
            if st.button("登录", key=f"login_button_{len(st.session_state.LOG)}"):
                login_popup.open()
        with col2:
            st.caption(f"<small>免登录试用版，登录后可以聊更多哦!</small>", unsafe_allow_html=True)
else:
    # Sync user info from database and refresh the sidebar and header displays
    st.session_state.USER.sync_from_db()
    with sidebar.container():
        update_sidebar()
    with header.container():
        update_header()

# Render footer
with footer:
    st.info("免责声明：聊天机器人基于海量互联网文本训练的大型语言模型，仅供娱乐。小潘AI不对信息的准确性、完整性、及时性等承担任何保证或责任。", icon="ℹ️")
    st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)

# Render the login popup with all the login logic included
if login_popup.is_open():
    with login_popup.container() as popup_container:
        new_registration = render_login_popup(popup_container, azure_table_op)
    login_popup.close()

if "NEW_USER" in st.session_state and st.session_state.NEW_USER:
    st.balloons()
    st.session_state.NEW_USER = False

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

# If the user has logged in and has no tokens left, will prompt him to recharge
if "USER" in st.session_state and st.session_state.USER.n_tokens <= 0:
    with prompt_box:
        st.warning("不好意思，你的聊天币已用完，请立即前往用户中心充值")
    components.html(expand_sidebar_script, height=0, width=0)
    time.sleep(1)
    st.stop()

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(human_prompt) > 0:

    # Strip the prompt of any potentially harmful html/js injections
    human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;")

    # Update both chat log and the model memory (copy two last entries from LOG to MEMORY)
    st.session_state.LOG.append("Human: " + human_prompt)
    st.session_state.LOG.append("AI: ")
    st.session_state.MEMORY.extend(st.session_state.LOG[-2:])

    # Run a special JS code to clear the input box after human_prompt is used
    # components.html(clear_input_script, height=0, width=0)
    prompt_box.empty()

    with chat_box:
        # Write the latest human message first
        line = st.session_state.LOG[-2]
        contents = line.split("Human: ")[1]
        st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

        reply_box = st.empty()

        # This is one of those small three-dot animations to indicate the bot is "writing"
        writing_animation = st.empty()
        file_path = os.path.join(ROOT_DIR, "src", "loading.gif")
        writing_animation.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<img src='data:image/gif;base64,{get_local_img(file_path)}' width=30 height=10>", unsafe_allow_html=True)

        # Call the OpenAI API to generate a response with retry, cooldown and backoff
        prompt, NLP_tokens_used = generate_prompt_from_memory()
        reply_box.markdown(get_chat_message(), unsafe_allow_html=True)
        for i in range(N_RETRIES):
            try:
                reply_text = openai.Completion.create(
                    model=NLP_MODEL_NAME,
                    prompt=prompt,
                    temperature=NLP_MODEL_TEMPERATURE,
                    max_tokens=NLP_MODEL_REPLY_MAX_TOKENS,
                    frequency_penalty=NLP_MODEL_FREQUENCY_PENALTY,
                    presence_penalty=NLP_MODEL_PRESENCE_PENALTY,
                    stop=NLP_MODEL_STOP_WORDS,
                ).choices[0].text.strip()
                break
            except Exception as e:
                if i == N_RETRIES - 1:
                    st.error(f"小潘AI出错了: {e}")
                    st.stop()
                else:
                    time.sleep(COOLDOWN * BACKOFF ** i)
        NLP_tokens_used += len(reply_text)
        # Synthesize the response and play it as audio
        audio_play_time = synthesize_text(reply_text, speech_cfg, azure_synthesizer)
        # Loop so that reply_text gets revealed one character at a time
        chars = len(reply_text)
        pause_per_char = 0.7 * audio_play_time / chars  # 0.8 because we want the text to appear a bit faster than the audio
        tic = time.time()
        for i in range(chars):
            with reply_box.container():
                st.markdown(get_chat_message(reply_text[:i+1]), unsafe_allow_html=True)
                time.sleep(pause_per_char)
        toc = time.time()
        # Pause for the remaining time, if any
        time.sleep(max(0, audio_play_time - (toc - tic)))

        # Clear the audio stream from voicePlayer
        components.html(f"""<script>window.parent.document.voicePlayer.src = "";</script>""", height=0, width=0)

        # Clear the writing animation
        writing_animation.empty()

        # Update the chat LOG and memories with the actual response
        st.session_state.LOG[-1] += reply_text
        st.session_state.MEMORY[-1] += reply_text

        # Use consumables
        if "USER" in st.session_state:
            partition_key = f"{st.session_state.USER.channel}_{st.session_state.USER.user_id}"
        else:
            partition_key = "unknown_user"
        action_res = use_consumables(
            azure_table_op,
            TOKENUSE_TABLE,
            partition_key,
            NLP_tokens_used,
            chars
        )
        if action_res['status'] != 0:
            st.error(f"无法消费聊天币: {action_res['message']}")
            st.stop()

        # Wrapping up one "round"
        if "USER" in st.session_state:
            # Update the sidebar and header token number
            with sidebar.container():
                update_sidebar()
            with header.container():
                update_header()

        elif len(st.session_state.LOG) > DEMO_HISTORY_LIMIT * 2:
            st.warning(f"**公测版，限{DEMO_HISTORY_LIMIT}次对话轮回**\n\n感谢您对小潘AI的兴趣。若想继续聊天，请在页面顶部进行登录！")
            prompt_box.empty()
            st.stop()

    st.experimental_rerun()
