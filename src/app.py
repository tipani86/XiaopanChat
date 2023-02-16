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

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()

TIMEOUT = 15
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource(show_spinner=False)
def get_table_op():
    return AzureTableOp()


@st.cache_data(show_spinner=False)
def get_loading_gif():
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(os.path.join(ROOT_DIR, "src", "loading.gif"), "rb").read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2", low_cpu_mem_usage=True)


@st.cache_data(show_spinner=False)
def get_js():
    # Add Javascript web trackers and a cool hack to enable Enter key to submit
    # (Ref: https://www.youtube.com/watch?v=SLyS0v8br20)
    return f"""
<script type="text/javascript">

(function(c,l,a,r,i,t,y){{
    c[a]=c[a]||function(){{(c[a].q=c[a].q||[]).push(arguments)}};
    t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
    y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
}})(window, document, "clarity", "script", "fuwdd48n5i");

(function(window, document, dataLayerName, id) {{
    window[dataLayerName]=window[dataLayerName]||[],window[dataLayerName].push({{start:(new Date).getTime(),event:"stg.start"}});var scripts=document.getElementsByTagName('script')[0],tags=document.createElement('script');
    function stgCreateCookie(a,b,c){{var d="";if(c){{var e=new Date;e.setTime(e.getTime()+24*c*60*60*1e3),d="; expires="+e.toUTCString()}}document.cookie=a+"="+b+d+"; path=/"}}
    var isStgDebug=(window.location.href.match("stg_debug")||document.cookie.match("stg_debug"))&&!window.location.href.match("stg_disable_debug");stgCreateCookie("stg_debug",isStgDebug?1:"",isStgDebug?14:-1);
    var qP=[];dataLayerName!=="dataLayer"&&qP.push("data_layer_name="+dataLayerName),isStgDebug&&qP.push("stg_debug");var qPString=qP.length>0?("?"+qP.join("&")):"";
    tags.async=!0,tags.src="https://xiaopan.containers.piwik.pro/"+id+".js"+qPString,scripts.parentNode.insertBefore(tags,scripts);
    !function(a,n,i){{a[n]=a[n]||{{}};for(var c=0;c<i.length;c++)!function(i){{a[n][i]=a[n][i]||{{}},a[n][i].api=a[n][i].api||function(){{var a=[].slice.call(arguments,0);"string"==typeof a[0]&&window[dataLayerName].push({{event:n+"."+i+":"+a[0],parameters:[].slice.call(arguments,1)}})}}}}(i[c])}}(window,"ppms",["tm","cm"]);
}})(window, document, 'dataLayer', '84ddea31-5408-4d83-a6fe-ffe81f25b029');

/*
const streamlitDoc = window.parent.document;

const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
console.log(buttons) // find buttons in console tab

const submit_button = buttons.find(el => el.innerText === '发送');

streamlitDoc.addEventListener('keydown', function(e) {{
    switch (e.key) {{
        case 'Enter':
            console.log('Enter pressed');
            submit_button.click();
            break;
    }}
}});
*/
</script>
"""


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

# Maintain a chat history in session state
if "HISTORY" not in st.session_state:
    st.session_state.HISTORY = [
        "You are an AI assistant called 小潘 (Xiaopan). You're very capable, able to adjust to the various messages from a human and provide helpful replies in the same language as the question was asked in. Below is the chat log:",
    ]

### MAIN STREAMLIT UI STARTS HERE ###


st.set_page_config(
    page_title="小潘AI",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/377f6a405e/favicon.svg",
)


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


with st.spinner("应用首次初始化中..."):
    azure_table_op = get_table_op()
    tokenizer = get_tokenizer()

padding = 2
st.markdown(f"""
<style>
    # MainMenu {{
        visibility: hidden;
    }}
    footer {{
        visibility: hidden;
    }}
    blockquote {{
        text-align: right;
    }}
    .appview-container .main .block-container {{
        padding-top: {padding}rem;
        padding-bottom: 0rem;
    }}
    # .row-widget .stButton {{
    #     display: none;
    # }}
</style>
""", unsafe_allow_html=True)


# Load JS code
components.html(get_js(), height=0, width=0)

# Define overall layout
query_params = st.experimental_get_query_params()
if DEBUG:
    st.write(f"`Query params: {query_params}`")

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

st.subheader("你好，我是小潘AI:robot_face:，跟我说点什么吧！")
chat_box = st.container()
prompt_box = st.empty()
footer = st.container()
with footer:
    st.info("免责声明：聊天机器人的输出基于用海量互联网文本数据训练的大型语言模型，仅供娱乐。对于信息的准确性、完整性、及时性等，小潘AI不承担任何责任。", icon="ℹ️")
    st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)

# Initialize chat history element

with chat_box:
    for line in st.session_state.HISTORY[1:]:
        # For AI response
        if line.startswith("AI: "):
            contents = line.split("AI: ")[1]
            st.markdown(f"`小潘`: {contents}")
        # For human prompts
        if line.startswith("Human: "):
            contents = line.split("Human: ")[1]
            st.markdown(f"> `我`: {contents}")

# Define prompt element which is just a simple form
    with prompt_box:
        with st.form(key="prompt", clear_on_submit=True):
            human_prompt = st.text_input("请输入:")
            clicked = st.form_submit_button("发送")

# If the user has submitted a prompt, we update the history, generate a response and show the response in chat box
if clicked:
    if len(human_prompt) == 0:
        st.warning("请输入内容")
        st.stop()

    st.session_state.HISTORY.append("Human: " + human_prompt)
    st.session_state.HISTORY.append("AI: ")
    with chat_box:
        # Write the latest human message first
        line = st.session_state.HISTORY[-2]
        contents = line.split("Human: ")[1]
        st.markdown(f"> `我`: {contents}")

        # Call the OpenAI API to generate a response with retry, cooldown and backoff with a cool streaming method
        # (Ref: https://medium.com/@avra42/how-to-stream-output-in-chatgpt-style-while-using-openai-completion-method-b90331c15e85)
        prompt = "\n".join(st.session_state.HISTORY)

        # Streaming text method:
        for i in range(N_RETRIES):
            try:
                reply_box = st.empty()
                reply_box.markdown(f"`小潘`: &nbsp;<img src='data:image/gif;base64,{get_loading_gif()}' width=30 height=10>", unsafe_allow_html=True)
                # Tokenize the prompt to count if we have reached the max tokens
                tokens = tokenizer.tokenize(prompt)
                st.write(f"Token count: {len(tokens)}")
                st.stop()

                reply = []
                for resp in openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=500,
                    frequency_penalty=1.0,
                    presence_penalty=1.0,
                    stop=[" 我:", " 小潘:"],
                    stream=True
                ):
                    reply.append(resp.choices[0].text)
                    reply_text = "".join(reply).strip()
                    reply_box.markdown(f"`小潘`: {reply_text}")
                break
            except Exception as e:
                if i == N_RETRIES - 1:
                    st.error(f"小潘AI出错了: {e}")
                    st.stop()
                else:
                    time.sleep(COOLDOWN * BACKOFF ** i)

        # Update the history with the response
        st.session_state.HISTORY[-1] += reply_text

        # Write the AI response
        # line = st.session_state.HISTORY[-1]
        # contents = line.split("AI: ")[1]
        # st.markdown(f"`小潘`: {contents}")

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

        elif len(st.session_state.HISTORY) > DEMO_HISTORY_LIMIT:
            st.warning(f"**公测版，限{DEMO_HISTORY_LIMIT}条消息的对话**\n\n感谢您对我们的兴趣，想获取更多消息次数可以登录哦！")
            prompt_box.empty()
            st.stop()
