import json
import torch
import streamlit as st
import os
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit import runtime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


st.set_page_config(page_title="Baichuan-13B-Chat")
st.title("Baichuan-13B-Chat")


def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip

def save_chat(user = 'baiyan'):
    if len(st.session_state.messages) > 0:
        f = open(f'users/{user}/{st.session_state.messages[0]["content"]}.txt', 'w')
        f.write(json.dumps(st.session_state.messages))
        f.close()

def init_chat_user(user = 'baiyan'):
    st.session_state.user = user
    path = f'./users/{user}'
    if not os.path.isdir(path):
        os.mkdir(path)
    all_files = os.listdir(path)
    all_files.sort(key=lambda fn: os.path.getmtime(f'./users/{user}/{fn}'), reverse=True)
    with st.sidebar:
        st.button("New", on_click=clear_chat_history)
        for fileN in all_files:
            file = fileN
            c1, c2 = st.columns([0.8, 0.2], gap="small")
            with c1:
                st.button(file, on_click=init_chat_history_in_file, args=[file], use_container_width=True)
            with c2:
                st.button("✖︎", key=file+"_del", on_click=delete_chat_history, args=[file])


@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        "baichuan-inc/Baichuan-13B-Chat",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "baichuan-inc/Baichuan-13B-Chat"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "baichuan-inc/Baichuan-13B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages

def init_chat_history_in_file(file_name = ""):
    if file_name == "":
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown("您好，我是442A实验室的百川大模型，很高兴为您服务🥰")
        st.session_state.messages = []
    else:
        with open(f'./users/{st.session_state.user}/'+file_name) as f:
            row = f.read()
            st.session_state.messages = json.loads(row)

def delete_chat_history(file_name = ""):
    os.remove(f'./users/{st.session_state.user}/' + file_name)
    pass

def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是442A实验室的百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()

    user = get_remote_ip()
    # global messages
    init_chat_user(user=user)
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)
        save_chat(user=user)



if __name__ == "__main__":
    main()
