'''
First Chatbot Application using Tiny LLama
'''

#imports
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import streamlit as st

@st.cache_resource(show_spinner="Loading TinyLlama…")
def load_model():
    #Load Model
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

    print('\nLoading Model...')
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto", torch_dtype=torch.float16,  low_cpu_mem_usage=True)
        print("Cuda detected")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,device_map = 'auto')
        print("auto device loading")

    print('\nModel Loaded')

    #Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('\nTokenizer Loaded')

    print(f"\nMODEL LOADED SUCCESFULLY on decive: {model.device}\n")
    return model, tokenizer

def type_message(chat_history):
    prompt = st.chat_input("Type Here")
    if prompt is None: 
        st.stop()
    
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
    message = {"role": "user", "content": f"{prompt}"}
    chat_history.append(message)
    input_tokenized = tokenizer.apply_chat_template(
                    chat_history, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)
    while input_tokenized.shape[1] > 1000:
        chat_history = chat_history[:1] + chat_history[2:]
        input_tokenized = tokenizer.apply_chat_template(
                    chat_history, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)
    return chat_history, input_tokenized, message


def run_message(input_tokenized,input_len, chat_history,temp,max_tokens):
    output = model.generate(input_tokenized, max_new_tokens = max_tokens, do_sample = True,  top_p = .8, temperature = temp)
    output_only = output[:,input_len:]
    outputonly_decoded = tokenizer.decode(output_only[0], skip_special_tokens=True)
    chat_history.append({"role": "assistant", "content": f"{outputonly_decoded}"})
    if outputonly_decoded:
        with st.chat_message("assistant"):
            st.markdown(outputonly_decoded)

    return outputonly_decoded, chat_history

def clear_chat():
    if st.sidebar.button('Clear Chat'):
        st.session_state.messages = [
        { "role": "system", "content": "You are a chatbot that responds in as few words as possible. Do NOT include role names or labels. Do not repeat system message"},
        {"role": "assistant", "content": "Hello 👋, how may I help you today?"},
        ]
        st.rerun()

def sidebar_options():
    with st.sidebar:
        temp = st.slider('Temperature:',0.0,2.0,.1,.1)
        max_tokens = st.slider('Max Tokens:',10,500,100,1)
    
    return temp, max_tokens


if __name__ == '__main__':

    model, tokenizer = load_model()

    st.title('Tiny LLama 1.1B Chatbot')
    st.write('By Aadit Patel')

    #SIDE BAR
    st.sidebar.title('Controls:')
    temp, max_tokens = sidebar_options()
    clear_chat()

    if "messages" not in st.session_state:
        st.session_state.messages = [
        { "role": "system", "content": "You are a chatbot that responds in as few words as possible. Do NOT include role names or labels. Do not repeat system message"},
        {"role": "assistant", "content": "Hello 👋, how may I help you today?"},
        ]
        
        

    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.session_state.messages, input_tokenized, message = type_message(st.session_state.messages)
    output, st.session_state.messages = run_message(input_tokenized,input_tokenized.shape[1],st.session_state.messages,temp, max_tokens)
    



    