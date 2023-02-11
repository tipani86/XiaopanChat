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

if "HISTORY" not in st.session_state:
    st.session_state.HISTORY = [
        "The following is a conversation with an AI assistant called 小潘. The assistant is very capable, able to adjust to the human's various requests and provide helpful answers. It can even reply in different languages to match whatever language the latest prompt was asked in.",
    ]

### MAIN STREAMLIT UI STARTS HERE ###

# Define overall layout

st.set_page_config(
    page_title="小潘AI",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/377f6a405e/favicon.svg",
)
st.subheader("跟小潘说点什么吧！")

chat_box = st.container()
prompt_box = st.container()

# Initialize chat history element

with chat_box:
    for line in st.session_state.HISTORY[1:]:
        st.write(line)

# Define prompt element which is just a simple form
    with prompt_box:
        with st.form(key="prompt", clear_on_submit=True):
            human_prompt = st.text_input("请输入:")
            clicked = st.form_submit_button("Enter")

# If the user has submitted a prompt, we update the history, generate a response and show the response in chat box

if clicked:
    st.session_state.HISTORY.append("我: " + human_prompt)
    st.session_state.HISTORY.append("小潘: ")
    with chat_box:
        st.write(st.session_state.HISTORY[-2])
        prompt = "\n".join(st.session_state.HISTORY)
        with st.spinner("`小潘正在打字中...`"):
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.9,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0.2,
                presence_penalty=0.6,
                stop=[" 我:", " 小潘:"]
            )
        response_text = response["choices"][0]["text"]
        st.session_state.HISTORY[-1] += response_text
        st.write(st.session_state.HISTORY[-1])
