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

### FUNCTION DEFINITIONS ###


@st.cache_data(show_spinner=False)
def get_local_img(file_path: str) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_favicon(file_path: str):
    # Load a byte image and return its favicon
    return Image.open(file_path)


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(ROOT_DIR, "src", "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"


### INITIALIZE AND LOAD ###


# Initialize page config
favicon = get_favicon(os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png"))
st.set_page_config(
    page_title="小潘AI",
    page_icon=favicon,
)


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

if DEBUG:
    with st.sidebar:
        st.markdown("<p><small>Debug tools:</small></p>", unsafe_allow_html=True)
        if st.button("Clear cache"):
            st.cache_data.clear()

    # Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)


# Render footer
with footer:
    st.info("暂停GPT-4对外公开服务的通知：亲爱的小潘AI用户，您好！因GPT-4模型成本实在过高，打赏金额远远支撑不了API接口调用的费用，我们不得不暂停GPT-4模型无条件对外公开的服务。同时，我们依然欢迎您使用基于GPT-3.5-Turbo（ChatGPT大众版）模型的服务：**chat.xiaopan.ai**", icon="ℹ️")
    st.write("如果您想加入小潘AI未来推出基于GPT-4模型的产品或服务的候补名单，第一时间了解我们的新产品，请扫码留下您的信息，非常感谢！")
    st.image(os.path.join(ROOT_DIR, "src", "assets", "waitlist.jpg"), width=300)
    """
    st.success("小潘AI背后由OpenAI的GPT4付费模型驱动。但我们依然想给国内的用户提供无门槛服务。如果您感觉小潘AI对你带来了价值，我们非常感谢您的爱心捐赠🙏！")
    with st.expander("显示支付码"):
        left_col, right_col = st.columns(2, gap="medium")
        with left_col:
            st.image(os.path.join(ROOT_DIR, "src", "assets", "alipay.png"))
        with right_col:
            st.image(os.path.join(ROOT_DIR, "src", "assets", "wechat_pay.png"))
    """
    st.markdown(f"<p style='text-align: right'><small><i><font color=gray>Build: {build_date}</font></i></small></p>", unsafe_allow_html=True)
