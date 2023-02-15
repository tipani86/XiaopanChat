import os
import time
import json
import openai
import qrcode
import requests
import numpy as np
import streamlit as st
from streamlit_modal import Modal
from azure_table_op import AzureTableOp

DEBUG = True

# Check environment variables

for key in ["OPENAI_API_KEY", "OPENAI_ORG_ID", "WX_LOGIN_SECRET", "AZURE_STORAGE_CONNECTION_STRING"]:
    if key not in os.environ:
        st.error(f"Please set the {key} environment variable.")
        st.stop()

# Set global variables

DEMO_HISTORY_LIMIT = 10

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()

TIMEOUT = 15
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        "You are an AI assistant called 小潘. You're very capable, able to adjust to the various messages from a human and provide helpful replies in the same language as the question was asked in. Below is the chat log:",
    ]

### MAIN STREAMLIT UI STARTS HERE ###

# Define overall layout
st.set_page_config(
    page_title="小潘AI",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/377f6a405e/favicon.svg",
)

@st.cache_resource
def get_table_op():
    return AzureTableOp()

azure_table_op = get_table_op()

padding = 2
st.markdown(f"""<style>
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
</style>""", unsafe_allow_html=True)

header = st.container()

if "USER_DATA" not in st.session_state:

    login_popup = Modal(title=None, key="login_popup", padding=40, max_width=204)
    with header:
        col1, col2 = st.columns([9, 1])
        with col1:
            st.write(":warning: 公测试用版，限10条消息，登录后可获取更多聊天资格哦!")
        with col2:
            start_login = st.button("登录")
        if start_login:
            with login_popup.container():
                st.write("")
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
                                st.error(f"小潘AI出错了: {e}")
                                st.stop()
                            else:
                                time.sleep(COOLDOWN * BACKOFF ** i)

                    # Parse results
                    if r.status_code != 200:
                        st.error(f"微信登陆出错了: {r.text}")
                        st.stop()

                    wx_login_res = r.json()
                    # st.json(wx_login_res, expanded=False)
                    if wx_login_res['errcode'] != 0:
                        st.error(f"微信登陆出错了: {wx_login_res['message']}")
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
                        st.error(f"无法更新用户表: {table_res['message']}")
                        st.stop()

                    # Finally, generate the QR code and show it to the user
                    qr_img = qrcode.make(wx_login_res['data']['qrCodeReturnUrl'])
                    qr_img = qr_img.resize((200, 200))
                    qr_img = np.asarray(qr_img) * 255   # The values are 0/1, we need to make it visible
                    st.image(qr_img, caption="请使用微信扫描二维码登录", output_format="PNG")

                # Poll the login status every 5 seconds and close the popup when login is successful

                while True:
                    time.sleep(5)

                    query_filter = f"PartitionKey eq @channel and RowKey eq @user_id"
                    select = None
                    parameters = {'channel': "wx_user", 'user_id': temp_user_id}

                    table_res = azure_table_op.query_entities(query_filter, select, parameters, table_name)

                    if table_res['status'] != 0:
                        st.error(f"无法获取用户表: {table_res['message']}")
                        st.stop()

                    if 'data' in table_res['data'][0] and table_res['data'][0]['data'] is not None:
                        break

                user_id = table_res['data'][0]['user_id']
                user_data = json.loads(table_res['data'][0]['data'])
                st.session_state['USER_DATA'] = {
                    'user_id': user_id,
                }
                st.session_state['USER_DATA'].update(user_data)

                login_popup.close()

else:
    with header:
        col1, col2 = st.columns([9, 1])
        with col1:
            st.write(f"欢迎回来，**{st.session_state.USER_DATA['nickname']}** :wave:")
        with col2:
            st.image(st.session_state.USER_DATA['avatar_url'], width=30)


# Main layout
st.subheader("跟小潘AI:robot_face:说点什么吧！")
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

        # Call the OpenAI API to generate a response with retry, cooldown and backoff
        prompt = "\n".join(st.session_state.HISTORY)
        with st.spinner("`小潘正在输入中...`"):
            for i in range(N_RETRIES):
                try:
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        temperature=0.5,
                        max_tokens=500,
                        frequency_penalty=1.0,
                        presence_penalty=1.0,
                        stop=[" 我:", " 小潘:"]
                    )
                    break
                except Exception as e:
                    if i == N_RETRIES - 1:
                        st.error(f"小潘AI出错了: {e}")
                        st.stop()
                    else:
                        time.sleep(COOLDOWN * BACKOFF ** i)
        response_text = response["choices"][0]["text"]

        # Update the history with the response
        st.session_state.HISTORY[-1] += response_text

        # Write the AI response
        line = st.session_state.HISTORY[-1]
        contents = line.split("AI: ")[1]
        st.markdown(f"`小潘`: {contents}")

        if len(st.session_state.HISTORY) > DEMO_HISTORY_LIMIT:
            st.warning(f"**公测版，限{DEMO_HISTORY_LIMIT}条对话**\n\n感谢您对我们的兴趣，我们会尽快上线更多功能！")
            prompt_box.empty()
            st.stop()