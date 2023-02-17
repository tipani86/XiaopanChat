import os
import time
import json
import base64
import openai
import qrcode
import requests
import datetime
import calendar
import numpy as np
import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components
from utils import AzureTableOp, User
from transformers import AutoTokenizer

DEBUG = True

# Check environment variables

for key in ["OPENAI_API_KEY", "OPENAI_ORG_ID", "WX_LOGIN_SECRET", "AZURE_STORAGE_CONNECTION_STRING"]:
    if key not in os.environ:
        st.error(f"Please set the {key} environment variable.")
        st.stop()

# Set global variables

DEMO_HISTORY_LIMIT = 10
NEW_USER_FREE_TOKENS = 20
FREE_TOKENS_PER_REFERRAL = 10

# Below are settings for NLP models, not to be confused with user tokens
NLP_MODEL_NAME = "text-davinci-003"
NLP_MODEL_MAX_TOKENS = 4000
NLP_MODEL_REPLY_MAX_TOKENS = 1500
NLP_MODEL_TEMPERATURE = 0.7
NLP_MODEL_FREQUENCY_PENALTY = 1.0
NLP_MODEL_PRESENCE_PENALTY = 1.0
NLP_MODEL_STOP_WORDS = [" 我:", " 小潘:", " Human:", " AI:"]

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()

TIMEOUT = 15
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

clear_input_script = """
<script>
    // Clear input value
    const streamlitDoc = window.parent.document
    // Find the target input element
    const inputs = Array.from(streamlitDoc.getElementsByTagName('input'))
    const input = inputs.find(el=> el.ariaLabel === '请输入:')
    // Clear the input value if it has value
    if (input.value) {
        input.value = ''
    }
</script>
"""


@st.cache_resource(show_spinner=False)
def get_table_op():
    return AzureTableOp()


@st.cache_data(show_spinner=False)
def get_local_img(file_path):
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2", low_cpu_mem_usage=True)


def get_js():
    # Read javascript web trackers code from script.js file
    with open(os.path.join(ROOT_DIR, "src", "script.js"), "r") as f:
        return f"<script type='text/javascript'>{f.read()}</script>"


def get_css():
    # Read CSS code from style.css file
    with open(os.path.join(ROOT_DIR, "src", "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"


# Load WeChat login configuration
wx_login_cfg_path = os.path.join(ROOT_DIR, "cfg", "wx_login.json")

if not os.path.isfile(wx_login_cfg_path):
    st.error("WeChat login configuration file not found.")
    st.stop()

with open(wx_login_cfg_path, "r") as f:
    wx_login_cfg = json.load(f)

if 'endpoint' not in wx_login_cfg:
    st.error("WeChat login endpoint not found in configuration file.")
    st.stop()

# Set OpenAI settings
openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize/maintain a chat log and chat memory in Streamlit's session state
# Log is the actual line by line chat, while memory is limited by model's maximum token context length
init_prompt = "You are an AI assistant called 小潘 (Xiaopan). You're very capable, able to adjust to the various messages from a human and provide helpful replies in the same language as the question was asked in. Add line breaks to structure your reply in paragraphs if it's long. Below is the chat log:"
if "MEMORY" not in st.session_state:
    st.session_state.MEMORY = [init_prompt]
    st.session_state.LOG = [init_prompt]

### MAIN STREAMLIT UI STARTS HERE ###


st.set_page_config(
    page_title="小潘AI",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/377f6a405e/favicon.svg",
)


def get_chat_message(
    contents: str = "",
    align: str = "left"
) -> str:
    div_class = "AI-line"
    color = "rgb(240, 242, 246)"
    # Generate a random AI avatar from today's date string
    src = f"https://api.dicebear.com/5.x/lorelei/svg?seed={datetime.datetime.now().strftime('%Y%m%d')}"
    if align == "right":
        div_class = "human-line"
        color = "rgb(165, 239, 127)"
        if "USER" in st.session_state:
            src = st.session_state.USER.avatar_url
        else:
            file_path = os.path.join(ROOT_DIR, "src", "user_icon.png")
            src = f"data:image/gif;base64,{get_local_img(file_path)}"
    icon_code = f"<img src='{src}' width=56 height=56 alt='avatar'>"
    formatted_contents = f"""
    <div class="{div_class}">
        {icon_code}
        <div class="chat-bubble" style="background: {color};">
        &#8203;{contents}
        </div>
    </div>
    """
    return formatted_contents


def update_header():
    header_text = f"欢迎回来 <b>{st.session_state.USER.nickname}</b> ！ "
    if st.session_state.USER.n_tokens <= 0:
        header_text += "<font color=red>你的消息次数已经全部用完了</font>"
    else:
        header_text += f"你还有 <b>{st.session_state.USER.n_tokens}</b> 条消息可以发哦"
    if st.session_state.USER.n_tokens < 10:
        header_text += "， 请立即<b>充值</b>"
    st.markdown(f"""
    <table style="border-collapse: collapse; border: none;" cellspacing=0 cellpadding=0 width="100%"><tr style="border: none;"><td style="border: none;" align="right"><small>{header_text}</small></td><td style="border: none;" width=25 align="right"><img height=25 width=25 src="{st.session_state.USER.avatar_url}" alt="avatar"></td></tr></table>
    """, unsafe_allow_html=True)


def generate_prompt_from_memory():
    # Check whether tokenized model memory so far + max reply length exceeds the max possible tokens
    memory_str = "\n".join(st.session_state.MEMORY)
    memory_tokens = tokenizer.tokenize(memory_str)
    if len(memory_tokens) + NLP_MODEL_REPLY_MAX_TOKENS > NLP_MODEL_MAX_TOKENS:
        # Strategy: We keep the first item of memory (original prompt), and last three items
        # (last AI message, human's reply, and the 'AI:' prompt) intact, and summarize the middle part
        summarizable_memory = st.session_state.MEMORY[1:-3]

        # We write a new prompt asking the model to summarize this middle part
        summarizable_memory = summarizable_memory + [
            "What you saw above is the conversation so far between you, the AI assistant, and a human user. Please summarize the topics discussed. Remember, do not write a direct reply to the user."
        ]
        summarizable_str = "\n".join(summarizable_memory)
        summarizable_tokens = tokenizer.tokenize(summarizable_str)

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

                if DEBUG:
                    st.info(f"Summarization triggered. New prompt:\n\n{new_prompt}")

                return new_prompt

        st.error("小潘AI出错了: 你的消息太长了，小潘AI无法处理")
        st.stop()

    # No need to summarize, just return the original prompt
    return memory_str


with st.spinner("应用首次初始化中..."):
    azure_table_op = get_table_op()
    tokenizer = get_tokenizer()

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# Load JS code
components.html(get_js(), height=0, width=0)

# Read browser query params and save them in session state
query_params = st.experimental_get_query_params()
if DEBUG:
    st.write(f"`Query params: {query_params}`")

# Define overall layout
header = st.empty()

# Define header and login popups
login_popup = Modal(title=None, key="login_popup", padding=40, max_width=204)

with header:
    if "USER" not in st.session_state:
        header_container = st.container()
        with header_container:
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(f"<small>面登录试用版，最多</small>`{DEMO_HISTORY_LIMIT}`<small>条消息的对话，登录后可获取更多聊天资格哦!</small>", unsafe_allow_html=True)
            with col2:
                start_login = st.button("登录")
                if start_login:
                    login_popup.open()
    else:
        header_container = st.container()
        with header_container:
            update_header()

# Render the login popup with all the login logic included
if login_popup.is_open():
    with login_popup.container():
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
                        table_name = table_name + "Test"
                    temp_user_id = wx_login_res['data']['tempUserId']
                    entity = {
                        'PartitionKey': "wx_user",
                        'RowKey': temp_user_id,
                        'data': None
                    }
                    table_res = azure_table_op.update_entities(entity, table_name)

                    if table_res['status'] != 0:
                        login_popup.close()
                        st.error(f"无法更新用户表: {table_res['message']}")
                        st.stop()

                    # Finally, generate the QR code and show it to the user
                    qr_img = qrcode.make(wx_login_res['data']['qrCodeReturnUrl'])
                    qr_img = qr_img.resize((200, 200))
                    qr_img = np.asarray(qr_img) * 255   # The values are 0/1, we need to make it visible

                st.image(qr_img, caption="请使用微信扫描二维码登录", output_format="PNG")

                # Poll the login status every 3 seconds and close the popup when login is successful
                while True:
                    time.sleep(3)

                    query_filter = f"PartitionKey eq @channel and RowKey eq @user_id"
                    select = None
                    parameters = {'channel': "wx_user", 'user_id': temp_user_id}

                    table_res = azure_table_op.query_entities(query_filter, select, parameters, table_name)
                    if table_res['status'] != 0:
                        login_popup.close()
                        st.error(f"无法获取用户表: {table_res['message']}")
                        st.stop()

                    entity = table_res['data'][0]

                    if 'data' in entity and entity['data'] is not None:
                        break

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
                    table_name = "users"
                    if DEBUG:
                        table_name = table_name + "Test"

                    # Build user object from temporary login data. The object is easier to manipulate later on.
                    st.session_state.USER = User(
                        channel="wx_user",
                        user_id=user_id,
                        db_op=azure_table_op,
                        table_name=table_name
                    )

                    # Delete the temp entry from the table
                    table_res = azure_table_op.delete_entity(entity, table_name)

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
                    else:
                        # Normal login, we update ip history from user_data
                        action_res = st.session_state.USER.update_ip_history(user_data)
                        if action_res['status'] != 0:
                            login_popup.close()
                            st.error(f"无法更新IP地址: {action_res['message']}")
                            st.stop()

    login_popup.close()

# Main layout

st.subheader("")
st.subheader("你好，我是小潘AI，来跟我说点什么吧！")
st.subheader("")
chat_box = st.container()
prompt_box = st.empty()
footer = st.container()
with footer:
    st.info("免责声明：聊天机器人的输出基于用海量互联网文本数据训练的大型语言模型，仅供娱乐。对于信息的准确性、完整性、及时性等，小潘AI不承担任何责任。", icon="ℹ️")
    st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)

# Render the chat log (without the initial prompt, of course)
with chat_box:
    for i, line in enumerate(st.session_state.LOG[1:]):
        # For AI response
        if line.startswith("AI: "):
            contents = line.split("AI: ")[1]
            # st.markdown(f"`小潘`: {contents}")
            st.markdown(get_chat_message(contents), unsafe_allow_html=True)

        # For human prompts
        if line.startswith("Human: "):
            contents = line.split("Human: ")[1]
            # st.markdown(f"> `我`: {contents}")
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)


# Define an input box for human prompts
with prompt_box:
    human_prompt = st.text_input("请输入:", key="human_prompt_text")

# When the user presses Enter, we update the chat log and the model memory
if len(human_prompt) <= 0:
    st.stop()

# Strip the prompt of any potentially harmful html/js injections
human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;")

# Update both chat log and the model memory (copy two last entries from LOG to MEMORY)
st.session_state.LOG.append("Human: " + human_prompt)
st.session_state.LOG.append("AI: ")
st.session_state.MEMORY.extend(st.session_state.LOG[-2:])

# Run a special JS code to clear the input box after human_prompt is used
components.html(clear_input_script, height=0, width=0)

with chat_box:
    # Write the latest human message first
    line = st.session_state.LOG[-2]
    contents = line.split("Human: ")[1]
    st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

    # A cool streaming output method by constantly writing to a placeholder element
    # (Ref: https://medium.com/@avra42/how-to-stream-output-in-chatgpt-style-while-using-openai-completion-method-b90331c15e85)
    reply_box = st.empty()

    # This is one of those small three-dot animations to indicate the bot is "writing"
    writing_animation = st.empty()
    file_path = os.path.join(ROOT_DIR, "src", "loading.gif")
    writing_animation.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<img src='data:image/gif;base64,{get_local_img(file_path)}' width=30 height=10>", unsafe_allow_html=True)

    # Call the OpenAI API to generate a response with retry, cooldown and backoff
    for i in range(N_RETRIES):
        try:
            reply_box.markdown(get_chat_message(), unsafe_allow_html=True)
            reply = []
            prompt = generate_prompt_from_memory()

            for resp in openai.Completion.create(
                model=NLP_MODEL_NAME,
                prompt=prompt,
                temperature=NLP_MODEL_TEMPERATURE,
                max_tokens=NLP_MODEL_REPLY_MAX_TOKENS,
                frequency_penalty=NLP_MODEL_FREQUENCY_PENALTY,
                presence_penalty=NLP_MODEL_PRESENCE_PENALTY,
                stop=NLP_MODEL_STOP_WORDS,
                stream=True
            ):
                reply.append(resp.choices[0].text)
                reply_text = "".join(reply).strip()
                # Visualize the streaming output in real-time
                reply_box.markdown(get_chat_message(reply_text), unsafe_allow_html=True)
            break

        except Exception as e:
            if i == N_RETRIES - 1:
                st.error(f"小潘AI出错了: {e}")
                st.stop()
            else:
                time.sleep(COOLDOWN * BACKOFF ** i)

    # Clear the writing animation
    writing_animation.empty()

    if DEBUG:
        prompt_tokens = tokenizer.tokenize(prompt)
        st.write(f"Prompt length: {len(prompt_tokens)} tokens")

    # Update the chat LOG and memories with the actual response
    st.session_state.LOG[-1] += reply_text
    st.session_state.MEMORY[-1] += reply_text

    if "USER" in st.session_state:
        # Consume one user token
        action_res = st.session_state.USER.consume_token()
        if action_res['status'] != 0:
            st.error(f"无法消费消息次数: {action_res['message']}")
            st.stop()

        with header:
            header_container = st.container()
            with header_container:
                update_header()

        # If the user has logged in and has no tokens left, will prompt him to recharge
        if st.session_state.USER.n_tokens <= 0:
            prompt_box.empty()
            st.warning("你的消息次数已用完，请充值")
            st.stop()

    elif len(st.session_state.LOG) > DEMO_HISTORY_LIMIT:
        st.warning(f"**公测版，限{DEMO_HISTORY_LIMIT}条消息的对话**\n\n感谢您对我们的兴趣，想获取更多消息次数可以登录哦！")
        prompt_box.empty()
        st.stop()
