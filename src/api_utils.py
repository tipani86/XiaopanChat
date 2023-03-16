import os
import re
import base64
import asyncio
from app_config import *


async def call_post_api_async(
    httpclient,
    url: str,
    headers: dict = None,
    data: dict = None,
) -> dict:
    res = {'status': 0, 'message': 'success', 'data': None}

    # Make an async post request to the API with timeout, retry, backoff etc.
    for i in range(N_RETRIES):
        async with httpclient.post(url, headers=headers, json=data, timeout=TIMEOUT) as response:
            if response.status == 200:
                res['data'] = await response.json()
                return res
            else:
                if i == N_RETRIES - 1:
                    res['status'] = response.status
                    res['message'] = response.reason
                    return res
                else:
                    await asyncio.sleep(COOLDOWN * BACKOFF ** i)


async def generate_prompt_from_memory_async(
    httpclient,
    tokenizer,
    memory: list,
    api_key: str,
) -> dict:
    res = {'status': 0, 'message': 'success', 'data': None}
    # Check whether tokenized memory so far + max reply length exceeds the max possible tokens for the model.
    # If so, summarize the middle part of the memory using the model itself, re-generate the memory.

    memory_str = "\n".join(x['content'] for x in memory)
    memory_tokens = tokenizer.tokenize(memory_str)
    tokens_used = 0  # NLP tokens (for OpenAI)
    if len(memory_tokens) + NLP_MODEL_REPLY_MAX_TOKENS > NLP_MODEL_MAX_TOKENS:
        # Strategy: We keep the first item of memory (original prompt), and last two items
        # (last AI message and human's reply) intact, and summarize the middle part
        summarizable_memory = memory[1:-2]

        # We write a new prompt asking the model to summarize this middle part
        summarizable_memory += [{
            'role': "system",
            'content': PRE_SUMMARY_PROMPT
        }]
        summarizable_str = "\n".join(x['content'] for x in summarizable_memory)
        summarizable_tokens = tokenizer.tokenize(summarizable_str)
        tokens_used += len(summarizable_tokens)

        # Check whether the summarizable tokens + 75% of the reply length exceeds the max possible tokens.
        # If so, adjust down to 50% of the reply length and try again, lastly if even 25% of the reply tokens still exceed, call an error.
        for ratio in [0.75, 0.5, 0.25]:
            if len(summarizable_tokens) + int(NLP_MODEL_REPLY_MAX_TOKENS * ratio) <= NLP_MODEL_MAX_TOKENS:
                # Call the OpenAI API
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    'Content-Type': "application/json",
                    'Authorization': f"Bearer {api_key}",
                }
                data = {
                    'model': NLP_MODEL_NAME,
                    'messages': summarizable_memory,
                    'temperature': NLP_MODEL_TEMPERATURE,
                    'max_tokens': int(NLP_MODEL_REPLY_MAX_TOKENS * ratio),
                    'frequency_penalty': NLP_MODEL_FREQUENCY_PENALTY,
                    'presence_penalty': NLP_MODEL_PRESENCE_PENALTY,
                    'stop': NLP_MODEL_STOP_WORDS,
                }
                api_res = await call_post_api_async(httpclient, url, headers, data)
                if api_res['status'] != 0:
                    res['status'] = api_res['status']
                    res['message'] = api_res['message']
                    return res

                summary_text = api_res['data']['choices'][0]['message']['content'].strip()
                tokens_used += len(tokenizer.tokenize(summary_text))

                # Re-build memory so it consists of the original prompt, a note that a summary follows,
                # the actual summary, a second note that the last two conversation items follow,
                # then the last three items from the original memory
                new_memory = memory[:1] + [{
                    'role': "system",
                    'content': text
                } for text in [PRE_SUMMARY_NOTE, summary_text, POST_SUMMARY_NOTE]] + memory[-2:]

                # Calculate the tokens used, including the new prompt
                new_prompt = "\n".join(x['content'] for x in new_memory)
                tokens_used += len(tokenizer.tokenize(new_prompt))

                if DEBUG:
                    print("Summarization triggered. New prompt:")
                    print(new_memory)

                # Build the output
                res['data'] = {
                    'messages': new_memory,
                    'tokens_used': tokens_used,
                }
                return res

        # If we reach here, it means that even 25% of the reply tokens still exceed the max possible tokens.
        res['status'] = 2
        res['message'] = "Summarization triggered but failed to generate a summary that fits the model's token limit."
        return res

    # No need to summarize, just return the original prompt
    tokens_used += len(memory_tokens)
    res['data'] = {
        'messages': memory,
        'tokens_used': tokens_used,
    }
    return res


async def get_chatbot_reply_data_async(
    httpclient,
    memory: list,
    api_key: str,
) -> dict:
    res = {'status': 0, 'message': "success", 'data': None}

    # Call the OpenAI API
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': "application/json",
        'Authorization': f"Bearer {api_key}",
    }
    data = {
        'model': NLP_MODEL_NAME,
        'messages': memory,
        'temperature': NLP_MODEL_TEMPERATURE,
        'max_tokens': NLP_MODEL_REPLY_MAX_TOKENS,
        'frequency_penalty': NLP_MODEL_FREQUENCY_PENALTY,
        'presence_penalty': NLP_MODEL_PRESENCE_PENALTY,
        'stop': NLP_MODEL_STOP_WORDS,
    }
    api_res = await call_post_api_async(httpclient, url, headers, data)
    print(api_res)
    if api_res['status'] != 0:
        res['status'] = api_res['status']
        res['message'] = api_res['message']
        return res

    reply_text = api_res['data']['choices'][0]['message']['content'].strip()

    # Detect language of the reply
    language_res = await detect_language_async(reply_text)
    if language_res['status'] != 0:
        res['status'] = language_res['status']
        res['message'] = language_res['message']
        return res

    # Build the output
    res['data'] = {
        'reply_text': reply_text,
        'language': language_res['data'],
    }
    return res


async def detect_language_async(
    httpclient,
    text: str
) -> dict:
    # Detect language of the text using a call to Rapid API with retry logic
    res = {'status': 0, 'msg': "success", 'data': None}

    url = "https://community-language-detection.p.rapidapi.com/detect"
    payload = {'q': text}
    headers = {
        'content-type': "application/json",
        'X-RapidAPI-Key': f"{os.getenv('RAPID_API_KEY')}",
        'X-RapidAPI-Host': "community-language-detection.p.rapidapi.com"
    }
    api_res = await call_post_api_async(httpclient, url, headers, payload)
    if api_res['status'] != 0:
        res['status'] = api_res['status']
        res['msg'] = api_res['msg']
        return res
    resp = api_res['data']
    if 'data' not in resp or 'detections' not in resp['data']:
        res['status'] = 2
        res['msg'] = f"No detections in response: {resp}"
        return res
    res['data'] = resp['data']['detections']
    return res


def synthesize_text(
    text: str,
    config: dict,
    synthesizer,
    speechsdk
) -> tuple:
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
        return length, b64
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
    return 0, ""