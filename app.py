from utils import semantics,load_model
import streamlit as st

st.sidebar.title("Welcome !")
st.sidebar.write("Effortlessly navigate the patent landscape with Team 1NeuRon's expert guidance, simplifying your journey through the complex world of patents.")
st.sidebar.info("Powered by : 1NeuRon ")

# Main chat interface
st.title("Discovering Patents Effortlessly")
st.write("Feel free to inquire about patents, and let the AI effortlessly guide you to the precise information you seek.")

# User input
user_input = st.text_input("What is your question about patents?", "")
load_model()
if st.button("Send"):
    
    if user_input:
        # Here you'd normally call a function that processes the question and fetches the answer
        # For example, using a predefined pipeline:
        # response = nlp_model(question=user_input, context="The context text for the model")
        # answer = response["answer"]
        
        # title,abstract=semantics(user_input)
        title,abstract=semantics(user_input)
        # answer = "This is where the answer from the chatbot would appear."  # Placeholder
        
        for Title,Abstract in zip(title,abstract):
                st.markdown(f"**Title:** ***{Title}***")
                st.markdown(f'**Abstract:** {Abstract}')
                st.markdown("----")
        # else:
        #     st.warning("Query not in the knowledge of Model.")

# Footer
st.sidebar.markdown("---")

