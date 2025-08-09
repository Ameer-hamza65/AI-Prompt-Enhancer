import streamlit as st
import requests

st.title('AI Prompt Optimizer')
user_input=st.text_input('Enter your prompt')
if st.button('Optimize'):
    with st.spinner('Opimizing...'):
        response = requests.post('http://localhost:8000/optimize', json={'user_input':user_input})
        if response.status_code==200:
            result=response.json()
            st.subheader('Optimized Prompt:')
            st.text(result['enhanced_prompt'])
        else:
            st.error('Error in optimization process.')
            