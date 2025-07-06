import streamlit as st 
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch 

st.set_page_config(page_title="LLaMA Chat", layout="centered") 
st.title("🦙 Chat with Fine-Tuned LLaMA") 
 
model_checkpoint="Karimtawfik/llama-merged-quantized"
 
#model_checkpoint="./llama-merged-quantized"

@st.cache_resource 
def load_model(): 
    model = AutoModelForCausalLM.from_pretrained( 
        model_checkpoint, 
        device_map="auto") 
    tokenizer = AutoTokenizer.from_pretrained( 
        model_checkpoint) 
    tokenizer.pad_token = tokenizer.eos_token 
    return model.eval(), tokenizer 
 
model, tokenizer = load_model() 
 
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = [] 

# Save new user message into session state 
user_instruction = st.chat_input("Ask a question...") 
if user_instruction: 
    st.session_state.chat_history.append({"role": "user", "content": user_instruction}) 
 
# Render entire conversation in order 
for msg in st.session_state.chat_history: 
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"]) 
 
# If a message was just added, now generate LLM response 
if user_instruction:  # Check if there's a new user message
    with st.chat_message("assistant"): 
        message_placeholder = st.empty()
        with st.spinner("Thinking..."): 
            prompt = f"""### Instruction:\n{user_instruction}\n\n### Input:\n\n### Response:""" 
            #prompt = f"User: {user_instruction}\nAssistant:" 
    
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device) 
            with torch.no_grad(): 
                output_ids = model.generate( 
                    **tokens, 
                    max_new_tokens=250, 
                    do_sample=True, #allow sampling(not greedy search)
                    top_p=0.9, #do random sampling from the top words that their cum is>0.9
                    temperature=0.7 
                ) 
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True) 
            response = decoded.split("### Response:")[-1].strip() 
             
            if response.endswith("**") or response.endswith("-") or response.endswith("•"): 
                response = response.rsplit("\n", 1)[0].strip() 
             
            if len(response.split("\n")[-1].split()) <= 2: 
                response = "\n".join(response.split("\n")[:-1]).strip() 
 
        # Display the response immediately
        message_placeholder.markdown(response)
        # Add to chat history for persistence
        st.session_state.chat_history.append({"role": "assistant", "content": response})
