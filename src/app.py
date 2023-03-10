import os
import time
import json
import base64
import openai
import qrcode
import requests
import datetime
import calendar
import humanize
import subprocess
import numpy as np
from PIL import Image
import streamlit as st
from app_config import *
from streamlit_modal import Modal
import streamlit.components.v1 as components
from transformers import AutoTokenizer
import azure.cognitiveservices.speech as speechsdk
from concurrent.futures import ThreadPoolExecutor, wait
from utils import AzureTableOp, User
from utils import use_consumables, generate_event_id, warm_up_api_server, synthesize_text, detect_language, get_payment_QR

# Set global variables

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check environment variables

errors = []
for key in [
    "OPENAI_API_KEY", "OPENAI_ORG_ID",  # For OpenAI APIs
    "AZURE_STORAGE_CONNECTION_STRING",  # For Table Storage
    "AZURE_SPEECH_KEY",                 # For Azure Speech APIs
    "RAPID_API_KEY",                    # For Rapid APIs
    "WX_LOGIN_SECRET",                  # WeChat Login
    "SEVENPAY_PID", "SEVENPAY_PKEY",    # Payment Gateway
    "PAYMENT_CALLBACK_ROUTE"            # Payment Gateway callback route
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


def update_header() -> None:
    if "NEW_USER" in st.session_state and st.session_state.NEW_USER:
        header_text = f"<font color=red>??????????????????AI</font> <b>{st.session_state.USER.nickname}</b> ??? "
    else:
        header_text = f"???????????? <b>{st.session_state.USER.nickname}</b> ??? "
    if st.session_state.USER.n_tokens <= 0:
        header_text += "<font color=red>????????????????????????????????????</font>"
    else:
        header_text += f"????????? <b>{st.session_state.USER.n_tokens}</b> ????????????????????????"
    if st.session_state.USER.n_tokens < 10:
        header_text += "??? ?????????????????????????????????"
    st.markdown(f"""
    <div class="human-line">
        <div><small>{header_text}</small></div>
    </div>
    """, unsafe_allow_html=True)


def update_sidebar() -> None:
    with st.container():
        st.title("????????????")
        st.subheader(st.session_state.USER.nickname)
        # st.caption(f"<small>({st.session_state.USER.user_id})</small>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<img class='chat-icon' src='{st.session_state.USER.avatar_url}' width=64 height=64 alt='avatar'>", unsafe_allow_html=True)
        with col2:
            st.metric("**???????????????**", f"{st.session_state.USER.n_tokens}???")

        refresh = st.button("????????????", key=f"refresh_{len(st.session_state.LOG)}")
        if refresh:
            st.experimental_rerun()

        cat_expander = st.expander("???? ???????????????")
        with cat_expander:
            tabs = st.tabs(SET_NAMES)
            for i, (tab, set_name_short) in enumerate(zip(tabs, SET_NAMES)):
                with tab:
                    set_name = f"{set_name_short}??????"
                    display_name = set_name
                    col1, col2 = st.columns([5, 4])
                    with col1:
                        # Always show price with two decimals even if it's integer
                        price = f"{catalogue[set_name]['price']:.2f}"
                        if catalogue[set_name]["sale_price"]:
                            display_name += f" (:red[~~??{catalogue[set_name]['price']:.2f}~~])"
                            price = f"{catalogue[set_name]['sale_price']:.2f}"
                        st.metric(
                            f"**{display_name}**",
                            f"??{price}",
                            f"{catalogue[set_name]['amount']}{catalogue[set_name]['unit']}{catalogue[set_name]['product']}"
                        )
                    with col2:
                        st.write("**????????????**")
                        vars()[f"generate_payment{i+1}"] = st.button("??????", key=f"generate_payment_{set_name}_{len(st.session_state.LOG)}")
            st.info("**?????????**???????????????????????????????????????????????????10??????????????????20??????????????????+?????????", icon="??????")

        payment_code_placeholder = st.empty()
        for i, set_name_short in enumerate(SET_NAMES):
            var_name = f"generate_payment{i+1}"
            if var_name in vars() and vars()[var_name]:
                set_name = f"{set_name_short}??????"
                row_key, timestamp = generate_event_id()
                price = catalogue[set_name]['sale_price'] if catalogue[set_name]['sale_price'] else catalogue[set_name]['price']
                remark = f"{catalogue[set_name]['amount']}{catalogue[set_name]['unit']}{catalogue[set_name]['product']}"
                if DEBUG:
                    price = 0.1
                    remark += " (DEBUG PRICE)"
                # Format requirements: http://7-pay.cn/doc.php#d2 (In my experience, using GET works. Didn't work with POST when I tried.)
                order_info = {
                    'body': str(set_name),
                    'fee': round(price, 2),
                    'pay_type': "alipay",
                    'no': int(row_key),
                    'notify_url': f"{endpoints['callback_endpoint']}{os.getenv('PAYMENT_CALLBACK_ROUTE')}",
                    'pid': os.getenv('SEVENPAY_PID'),
                    'remark': str(remark)
                }

                with payment_code_placeholder.container():
                    with st.spinner("???????????????..."):
                        # First, add an open order to the database
                        action_res = st.session_state.USER.add_order(catalogue[set_name]['amount'], order_info)
                        if action_res['status'] != 0:
                            st.error(f"?????????????????????{action_res['message']}")
                    with st.spinner("??????????????????..."):
                        # Second, send the order info to the payment gateway and request QR code
                        payment_res = get_payment_QR(
                            endpoints['sevenpay_endpoint'],
                            order_info
                        )
                        if payment_res['code'] != "success":
                            st.error(f"????????????????????????{payment_res['msg']}")
                    if payment_res['code'] == "success":
                        alipay_logo_path = os.path.join(ROOT_DIR, "src", "assets", "120_348_alipay.png")
                        st.image(f"data:image/gif;base64,{get_local_img(alipay_logo_path)}", width=200)
                        st.image(payment_res['img'], width=200)
                        st.caption(f"????????????????????????????????????????????????????????????????????????")

        # st.write("")
        with st.expander("???? ???10???????????????"):
            # Calculate and display user's IP history and time
            d = datetime.datetime.now()
            timestamp_now = calendar.timegm(d.timetuple())

            # We need to read the IP history in reverse order
            formatted_ip_history = []
            for timestamp, ip in reversed(st.session_state.USER.ip_history):
                formatted_ip_history.append(f"{ip}, {humanize.naturaltime(datetime.timedelta(seconds=timestamp_now - timestamp))}")
            st.caption("<br>".join(formatted_ip_history), unsafe_allow_html=True)

        with st.expander("???? ????????????"):
            d = datetime.datetime.now()
            timestamp_now = calendar.timegm(d.timetuple())

            # We need to sort the transactions dataframe in reverse order and only show positive token values
            transactions = st.session_state.USER.transactions.sort_values(by="eventtime", ascending=False)
            transactions = transactions[transactions["tokens"] > 0]
            formatted_transactions = []
            for row in transactions.itertuples():
                if "[SYSTEM]" in row.comments:
                    message = f"[????????????] +{row.tokens}???"
                else:
                    message = f"?????? +{row.tokens}???"
                formatted_transactions.append(f"{message}, {humanize.naturaltime(datetime.timedelta(seconds=timestamp_now - row.eventtime))}")
            st.caption("<br>".join(formatted_transactions), unsafe_allow_html=True)

        st.header("")


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
                with st.spinner("???????????????????????????..."):
                    # Call the WeChat login service with retry, cooldown and backoff
                    url = os.path.join(
                        endpoints['wx_login_endpoint'],
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
                                st.error(f"??????AI?????????: {e}")
                                st.stop()
                            else:
                                time.sleep(COOLDOWN * BACKOFF ** i)

                    # Parse results
                    if r.status_code != 200:
                        login_popup.close()
                        st.error(f"?????????????????????: {r.text}")
                        st.stop()

                    wx_login_res = r.json()
                    if wx_login_res['errcode'] != 0:
                        login_popup.close()
                        st.error(f"?????????????????????: {wx_login_res['message']}")
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
                        st.error(f"?????????????????????: {table_res['message']}")
                        st.stop()

                    # Finally, generate the QR code and show it to the user
                    qr_img = qrcode.make(wx_login_res['data']['qrCodeReturnUrl'])
                    qr_img = qr_img.resize((200, 200))
                    qr_img = np.asarray(qr_img) * 255   # The values are 0/1, we need to make it visible

                st.image(qr_img, caption="????????????????????????????????????", output_format="PNG")
                st.caption(":red[???????????????????????????????????????????????????????????????????????????????????????????????????]")

                # Poll the login status every 3 seconds and close the popup when login is successful, or timeout after 60 seconds.
                for i in range(20):
                    time.sleep(3)

                    query_filter = f"PartitionKey eq @channel and RowKey eq @user_id"
                    select = None
                    parameters = {'channel': "wx_user", 'user_id': temp_user_id}

                    table_res = table_op.query_entities(query_filter, select, parameters, table_name)
                    if table_res['status'] != 0:
                        login_popup.close()
                        st.error(f"?????????????????????: {table_res['message']}")
                        st.stop()

                    entity = table_res['data'][0]

                    if 'data' in entity and entity['data'] is not None:
                        break
                if 'data' not in entity or entity['data'] is None:
                    login_popup.close()
                    st.error("????????????????????????")
                    st.stop()

        popup_content.empty()
        time.sleep(0.1)

        # Step 2: QR scanned, perform actual login and user initialization
        with popup_content:
            popup_container = st.container()
            with popup_container:
                with st.spinner("???????????????????????????..."):
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
                        consumption_table=CONSUMPTION_TABLE
                    )

                    # Delete the temp entry from the table
                    table_name = "tempUserIds"
                    if DEBUG:
                        table_name += "Test"
                    table_res = table_op.delete_entity(entity, table_name)

                    # Pull out the sidebar now that the user has logged in
                    with sidebar.container():
                        st.subheader("?????????...")
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
                            st.error(f"???????????????????????????: {action_res['message']}")
                            st.stop()
                        if "NEW_USER" not in st.session_state:
                            st.session_state.NEW_USER = True
                    elif action_res['status'] != 0:
                        login_popup.close()
                        st.error(f"????????????????????????: {action_res['message']}")
                        st.stop()
                    else:
                        # Normal login, we update current tokens amount and ip history from user_data
                        st.session_state.USER.n_tokens = action_res['n_tokens']
                        action_res = st.session_state.USER.update_ip_history(user_data)
                        if action_res['status'] != 0:
                            login_popup.close()
                            st.error(f"????????????IP??????: {action_res['message']}")
                            st.stop()
                    login_popup.close()


def generate_messages_from_memory():
    # Check whether tokenized model memory so far + max reply length exceeds the max possible tokens
    memory_str = "\n".join([m['content'] for m in st.session_state.MEMORY])
    memory_tokens = tokenizer.tokenize(memory_str)
    tokens_used = 0  # NLP tokens (for OpenAI)
    if len(memory_tokens) + NLP_MODEL_REPLY_MAX_TOKENS > NLP_MODEL_MAX_TOKENS:
        # Strategy: We keep the first item of memory (original prompt), and last three items
        # (last AI message, human's reply, and the 'AI:' prompt) intact, and summarize the middle part
        summarizable_memory = st.session_state.MEMORY[1:-3]

        # We write a new prompt asking the model to summarize this middle part
        summarizable_memory += [{'role': "system", 'content': PRE_SUMMARY_PROMPT}]
        summarizable_str = ["\n".join([m['content'] for m in summarizable_memory])]
        summarizable_tokens = tokenizer.tokenize(summarizable_str)
        tokens_used += len(summarizable_tokens)

        # Check whether the summarizable tokens + 75% of the reply length exceeds the max possible tokens.
        # If so, adjust down to 50% of the reply length and try again, lastly if even 25% of the reply tokens still exceed, call an error.
        for ratio in [0.75, 0.5, 0.25]:
            if len(summarizable_tokens) + int(NLP_MODEL_REPLY_MAX_TOKENS * ratio) <= NLP_MODEL_MAX_TOKENS:
                # Call the OpenAI API with retry and all that shebang
                for i in range(N_RETRIES):
                    try:
                        response = openai.ChatCompletion.create(
                            model=NLP_MODEL_NAME,
                            messages=summarizable_str,
                            temperature=NLP_MODEL_TEMPERATURE,
                            max_tokens=int(NLP_MODEL_REPLY_MAX_TOKENS * ratio),
                            frequency_penalty=NLP_MODEL_FREQUENCY_PENALTY,
                            presence_penalty=NLP_MODEL_PRESENCE_PENALTY,
                            stop=NLP_MODEL_STOP_WORDS,
                        )
                        break
                    except Exception as e:
                        if i == N_RETRIES - 1:
                            st.error(f"??????AI?????????: {e}")
                            st.stop()
                        else:
                            time.sleep(COOLDOWN * BACKOFF ** i)
                summary_text = response['choices'][0]['message']['content'].strip()
                tokens_used += len(tokenizer.tokenize(summary_text))

                # Re-build memory so it consists of the original prompt, a note that a summary follows,
                # the actual summary, a second note that the last two conversation items follow,
                # then the last three items from the original memory
                new_memory = st.session_state.MEMORY[:1] + \
                    [{'role': "system", 'content': PRE_SUMMARY_NOTE}] + \
                    [{'role': "assistant", 'content': summary_text}] + \
                    [{'role': "system", 'content': POST_SUMMARY_NOTE}] + \
                    st.session_state.MEMORY[-3:]

                st.session_state.MEMORY = new_memory

                # Re-generate prompt from new memory
                new_prompt = st.session_state.MEMORY
                new_prompt_str = "\n".join([m['content'] for m in new_prompt])
                new_prompt_tokens = tokenizer.tokenize(new_prompt_str)
                tokens_used += len(new_prompt_tokens)

                if DEBUG:
                    st.info(f"Summarization triggered. New prompt:\n\n{new_prompt}")

                return new_prompt, tokens_used

        st.error("??????AI?????????: ??????????????????????????????AI????????????")
        st.stop()

    # No need to summarize, just return the original prompt
    tokens_used += len(memory_tokens)
    return st.session_state.MEMORY, tokens_used


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
        summarizable_memory += [PRE_SUMMARY_PROMPT]
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
                            st.error(f"??????AI?????????: {e}")
                            st.stop()
                        else:
                            time.sleep(COOLDOWN * BACKOFF ** i)
                summary_text = response["choices"][0]["text"].strip()
                tokens_used += len(tokenizer.tokenize(summary_text))

                # Re-build memory so it consists of the original prompt, a note that a summary follows,
                # the actual summary, a second note that the last two conversation items follow,
                # then the last three items from the original memory
                new_memory = st.session_state.MEMORY[:1] + [PRE_SUMMARY_NOTE] + [summary_text] + [POST_SUMMARY_NOTE] + st.session_state.MEMORY[-3:]

                st.session_state.MEMORY = new_memory

                # Re-generate prompt from new memory
                new_prompt = "\n".join(st.session_state.MEMORY)
                tokens_used += len(tokenizer.tokenize(new_prompt))

                if DEBUG:
                    st.info(f"Summarization triggered. New prompt:\n\n{new_prompt}")

                return new_prompt, tokens_used

        st.error("??????AI?????????: ??????????????????????????????AI????????????")
        st.stop()

    # No need to summarize, just return the original prompt
    tokens_used += len(memory_tokens)
    return memory_str, tokens_used


### INITIALIZE AND LOAD ###


# Initialize page config
favicon = get_favicon(os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png"))
st.set_page_config(
    page_title="??????AI",
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


# Load endpoints configuration and validate its data
endpoints_path = os.path.join(ROOT_DIR, "cfg", "endpoints.json")
endpoints = get_json(endpoints_path)
for key in ["sevenpay_endpoint", "wx_login_endpoint", "callback_endpoint"]:
    if key not in endpoints:
        st.error(f"Key {key} not found in endpoints configuration file.")
        st.stop()


# Load product catalogue configuration and validate its data
catalogue_path = os.path.join(ROOT_DIR, "cfg", "catalogue.json")
catalogue = get_json(catalogue_path)
for set_name in catalogue:
    for key in ["price", "product", "amount"]:
        if key not in catalogue[set_name]:
            st.error(f"Set {set_name} is missing key {key} in catalogue configuration file.")
            st.stop()


# Initialize humanized time output in Simplified Chinese
_t = humanize.i18n.activate("zh_CN")


# Initialize some useful class instances
with st.spinner("????????????????????????..."):
    tokenizer = get_tokenizer()  # First time after deployment takes a few seconds
azure_table_op = get_table_op()
azure_synthesizer = get_synthesizer(speech_cfg)
openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")
EXECUTOR = ThreadPoolExecutor(2)

if DEBUG:
    for table_name_var in ["USERS_TABLE", "ORDERS_TABLE", "CONSUMPTION_TABLE"]:
        vars()[table_name_var] += "Test"


build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()
else:
    try:
        build_date = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(f"Failed to get git commit hash: {e}")


# Warm-up the API server when the user accesses the site
# (Azure tends to spin them down after some inactivity)
if "WARM_UP" not in st.session_state or not st.session_state.WARM_UP:
    with st.spinner("??????AI???????????????..."):
        warm_up_res = warm_up_api_server(endpoints['callback_endpoint'])

    if warm_up_res['status'] != 0:
        st.error(f"??????AI???????????????{warm_up_res['msg']}")
        st.stop()
    st.session_state.WARM_UP = True


### MAIN STREAMLIT UI STARTS HERE ###


# Define main layout
header = st.empty()
st.title("?????????")
st.subheader("????????????AI??????????????????????????????")
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


# Initialize/maintain a chat log and chat memory in Streamlit's session state
# Log is the actual line by line chat, while memory is limited by model's maximum token context length
if "MEMORY" not in st.session_state:
    st.session_state.MEMORY = [{'role': "system", 'content': INITIAL_PROMPT}]
    st.session_state.LOG = [INITIAL_PROMPT]


# Render header and sidebar depending on whether the user is logged in or not
if "USER" not in st.session_state:
    with sidebar.container():
        st.caption("?????????????????????????????????")
        if st.button("??????", key=f"login_button_{len(st.session_state.LOG)}"):
            login_popup.open()
    with header.container():
        st.caption(f"<small>??????????????????????????????????????????????????? [ <b>></b> ] ???????????????????????????????????????!</small>", unsafe_allow_html=True)

else:
    # Sync user info from database and refresh the sidebar and header displays
    action_res = st.session_state.USER.sync_from_db()
    if action_res['status'] != 0:
        st.error(f"???????????????????????????{action_res['msg']}")
        st.stop()
    if action_res['n_tokens'] > st.session_state.USER.n_tokens:
        st.session_state.BALLOONS = True
    st.session_state.USER.n_tokens = action_res['n_tokens']
    with sidebar.container():
        update_sidebar()
    with header.container():
        update_header()


# Render footer
with footer:
    st.info("????????????????????????????????????????????????????????????????????????????????????????????????????????????AI?????????????????????????????????????????????????????????????????????????????????", icon="??????")
    st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)


# Render the login popup with all the login logic included
if login_popup.is_open():
    with login_popup.container() as popup_container:
        new_registration = render_login_popup(popup_container, azure_table_op)
    login_popup.close()

if "NEW_USER" in st.session_state and st.session_state.NEW_USER:
    st.session_state.BALLOONS = True
    st.session_state.NEW_USER = False
if "BALLOONS" in st.session_state and st.session_state.BALLOONS:
    st.balloons()
    st.session_state.BALLOONS = False

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
    human_prompt = st.text_input("?????????:", value="", key=f"text_input_{len(st.session_state.LOG)}")

# If the user has logged in and has no tokens left, will prompt him to recharge
if "USER" in st.session_state and st.session_state.USER.n_tokens <= 0:
    with prompt_box:
        st.warning("???????????????????????????????????????????????????????????????????????????")
    components.html(expand_sidebar_script, height=0, width=0)
    time.sleep(1)
    st.stop()

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(human_prompt) > 0:

    # Strip the prompt of any potentially harmful html/js injections
    human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;")

    # Update both chat log and the model memory (copy two last entries from LOG to MEMORY)
    st.session_state.LOG.append("Human: " + human_prompt)
    st.session_state.MEMORY.append({'role': "user", 'content': human_prompt})

    # Run a special JS code to clear the input box after human_prompt is used
    # components.html(clear_input_script, height=0, width=0)
    prompt_box.empty()

    with chat_box:
        # Write the latest human message first
        line = st.session_state.LOG[-1]
        contents = line.split("Human: ")[1]
        st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

        reply_box = st.empty()

        # This is one of those small three-dot animations to indicate the bot is "writing"
        writing_animation = st.empty()
        file_path = os.path.join(ROOT_DIR, "src", "assets", "loading.gif")
        writing_animation.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<img src='data:image/gif;base64,{get_local_img(file_path)}' width=30 height=10>", unsafe_allow_html=True)

        # Call the OpenAI API to generate a response with retry, cooldown and backoff
        # prompt, NLP_tokens_used = generate_prompt_from_memory()
        prompt, NLP_tokens_used = generate_messages_from_memory()
        reply_box.markdown(get_chat_message(), unsafe_allow_html=True)
        for i in range(N_RETRIES):
            try:
                reply_text = openai.ChatCompletion.create(
                    model=NLP_MODEL_NAME,
                    messages=prompt,
                    temperature=NLP_MODEL_TEMPERATURE,
                    max_tokens=NLP_MODEL_REPLY_MAX_TOKENS,
                    frequency_penalty=NLP_MODEL_FREQUENCY_PENALTY,
                    presence_penalty=NLP_MODEL_PRESENCE_PENALTY,
                    stop=NLP_MODEL_STOP_WORDS,
                ).choices[0].message.content.strip()
                break
            except Exception as e:
                if i == N_RETRIES - 1:
                    reply_box.empty()
                    st.error(f"??????AI?????????: {e}")
                    st.stop()
                else:
                    time.sleep(COOLDOWN * BACKOFF ** i)
        language_res = detect_language(reply_text)
        if language_res['status'] != 0:
            reply_box.empty()
            st.error(f"??????????????????: {language_res['msg']}")
            st.stop()
        if DEBUG:
            print(f"Language detection result: {language_res['data']}")
        audio_play_time, audio_chars = 0, 0
        for item in language_res['data']:
            if item['language'] == "zh" and item['isReliable']:
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
                        }}, 500);
                    </script>""", height=0, width=0)
                    audio_play_time += 0.5  # To account for slight delay in the beginning of the audio

        # Start executing the consumption job in a separate thread
        if "USER" in st.session_state:
            partition_key = f"{st.session_state.USER.channel}_{st.session_state.USER.user_id}"
        else:
            partition_key = "unknown_user"
        consumption_task = EXECUTOR.submit(
            use_consumables,
            azure_table_op,
            CONSUMPTION_TABLE,
            partition_key,
            1,  # Chat tokens used
            NLP_tokens_used,
            audio_chars
        )

        # Loop so that reply_text gets revealed one character at a time
        if audio_chars > 0:
            pause_per_char = 0.7 * audio_play_time / audio_chars  # 0.7 because we want the text to appear a bit faster than the audio
        else:
            pause_per_char = 0.1
        tic = time.time()
        for i in range(len(reply_text)):
            with reply_box.container():
                st.markdown(get_chat_message(reply_text[:i+1]), unsafe_allow_html=True)
                time.sleep(pause_per_char)
        toc = time.time()
        # Pause for the remaining time, if any
        time.sleep(max(0, audio_play_time - (toc - tic)))

        # Stop and clear the audio stream from voicePlayer
        components.html(f"""<script>
            window.parent.document.voicePlayer.pause();
            window.parent.document.voicePlayer.src = "";
        </script>""", height=0, width=0)

        # Wait for the consumption job to finish
        consumption_task_res = wait([consumption_task], timeout=TIMEOUT)
        if len(consumption_task_res.not_done) > 0:
            reply_box.empty()
            st.error(f"?????????????????????: ??????")
            st.stop()

        for task_res in consumption_task_res.done:
            action_res = task_res.result()

        # Clear the writing animation
        writing_animation.empty()

        if action_res['status'] != 0:
            reply_box.empty()
            st.error(f"?????????????????????: {action_res['message']}")
            st.stop()

        # Update the chat LOG and memories with the actual response
        st.session_state.LOG.append(f"AI: {reply_text}")
        st.session_state.MEMORY.append({'role': "assistant", 'content': reply_text})

        if "USER" not in st.session_state and len(st.session_state.LOG) > DEMO_HISTORY_LIMIT * 2:
            st.warning(f"**???????????????{DEMO_HISTORY_LIMIT}???????????????**\n\n??????????????????AI??????????????????????????????????????????????????????????????????")
            prompt_box.empty()
            st.stop()

    st.experimental_rerun()
