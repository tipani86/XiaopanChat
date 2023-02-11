import os
import openai
import streamlit as st

### MAIN STREAMLIT UI STARTS HERE ###

# Define overall layout

st.set_page_config(
    page_title="小潘AI",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/377f6a405e/favicon.svg",
)
st.title("小潘AI Chatbot")

# Load OpenAI related settings

for key in ["OPENAI_API_KEY", "OPENAI_ORG_ID"]:
    if key not in os.environ:
        st.error(f"Please set the {key} environment variable.")
        st.stop()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

st.write(
    openai.Model.list()
)
