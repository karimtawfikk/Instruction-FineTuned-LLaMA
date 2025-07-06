import streamlit as st 
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch 

st.set_page_config(page_title="LLaMA Chat", layout="centered") 
st.title("🦙 Chat with Fine-Tuned LLaMA") 
 
model_checkpoint="Karimtawfik/llama-merged-quantized"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource 
def load_model(): 
    model = AutoModelForCausalLM.from_pretrained( 
        model_checkpoint,
        device_map=None,
        low_cpu_mem_usage=False 
        ).to(device) 
 
    tokenizer = AutoTokenizer.from_pretrained( 
        model_checkpoint) 
    tokenizer.pad_token = tokenizer.eos_token 
    return model.eval(), tokenizer 
 
model, tokenizer = load_model() 
 
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = [] 

user_instruction = st.chat_input("Ask a question...") 
if user_instruction: 
    st.session_state.chat_history.append({"role": "user", "content": user_instruction}) 
 
for msg in st.session_state.chat_history: 
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"]) 
 
if user_instruction:  
    with st.chat_message("assistant"): 
        message_placeholder = st.empty()
        with st.spinner("Thinking..."): 
            prompt = f"""### Instruction:\n{user_instruction}\n\n### Input:\n\n### Response:""" 
            #prompt = f"User: {user_instruction}\nAssistant:" 
    
            tokens = tokenizer(prompt, return_tensors="pt").to(device) 
            with torch.no_grad(): 
                output_ids = model.generate( 
                    **tokens, 
                    max_new_tokens=250, 
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.7 
                ) 
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True) 
            response = decoded.split("### Response:")[-1].strip() 
             
            if response.endswith("**") or response.endswith("-") or response.endswith("•"): 
                response = response.rsplit("\n", 1)[0].strip() 
             
            if len(response.split("\n")[-1].split()) <= 2: 
                response = "\n".join(response.split("\n")[:-1]).strip() 
 
        message_placeholder.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
