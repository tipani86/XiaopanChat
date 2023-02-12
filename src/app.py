import os
import openai
import streamlit as st

# Load OpenAI related settings

for key in ["OPENAI_API_KEY", "OPENAI_ORG_ID"]:
    if key not in os.environ:
        st.error(f"Please set the {key} environment variable.")
        st.stop()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Maintain a chat history in session state

if "HISTORY" not in st.session_state:
    st.session_state.HISTORY = [
        "You are an AI assistant called 小潘. You're very capable, able to adjust to the various messages from a human and provide helpful replies in the same language as the question was asked in. Below is the chat log:",
    ]

# Global variables

DEMO_HISTORY_LIMIT = 10

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()

### MAIN STREAMLIT UI STARTS HERE ###

# Define overall layout
st.set_page_config(
    page_title="小潘AI",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/377f6a405e/favicon.svg",
)

padding = 1.5
st.markdown(f"""<style>
    #MainMenu {{
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

st.subheader("跟小潘AI:robot_face:说点什么吧！")
chat_box = st.container()
prompt_box = st.empty()
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

        # Call the OpenAI API to generate a response
        prompt = "\n".join(st.session_state.HISTORY)
        with st.spinner("`小潘正在输入中...`"):
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.5,
                max_tokens=500,
                frequency_penalty=1.0,
                presence_penalty=1.0,
                stop=[" 我:", " 小潘:"]
            )
        response_text = response["choices"][0]["text"]

        # Add a way to detect whether a long response is cut off and then splitting on the last line break
        if len(response_text) > 200 and "\n" in response_text:
            response_text = response_text[:response_text.rfind("\n")]

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
