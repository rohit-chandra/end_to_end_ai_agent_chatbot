import streamlit as st
import requests

# Streamlit App Configuration
st.set_page_config(page_title="AI AGENT CHATBOT", layout="centered")

# Define API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Predefined models
MODEL_NAMES = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    
]

st.markdown(
    """
    <style>
    .big-font {
        font-size:26px !important;
    }
    .title-font {
        font-size:50px !important;
        font-weight: bold;
        color: #007BFF;
    }
    .response-font {
        font-size:26px !important;
        font-weight: bold;
        color: #28A745;
    }
    .text-area-label {
        font-size: 30px !important;
        font-weight: bold;
        color: #007BFF;
        margin-bottom: -30px !important;
        display: block;
    }
    </style>
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI Elements
st.markdown('<p class="title-font">AI AGENT CHATBOT</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">Interact with the LangGraph-based agent using this interface.</p>', unsafe_allow_html=True)

# Input box for system prompt
st.markdown('<p class="text-area-label">Define your AI Agent:</p>', unsafe_allow_html=True)
given_system_prompt = st.text_area("", height=150, placeholder="Type your system prompt here...")

# Dropdown for selecting the model
st.markdown('<p class="text-area-label">Select Model:</p>', unsafe_allow_html=True)
selected_model = st.selectbox("", MODEL_NAMES)

# Input box for user messages
st.markdown('<p class="text-area-label">Enter your message(s):</p>', unsafe_allow_html=True)
user_input = st.text_area("", height=150, placeholder="Type your message here...")


# Button to send the query
if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Processing... Please wait"):
            try:
                # Send the input to the FastAPI backend
                payload = {"messages": [user_input], "model_name": selected_model, 'system_prompt': given_system_prompt}
                response = requests.post(API_URL, json=payload)

                # Display the response
                if response.status_code == 200:
                    response_data = response.json()
                    if "error" in response_data:
                        st.error(response_data["error"])
                    else:
                        ai_responses = [
                            message.get("content", "")
                            for message in response_data.get("messages", [])
                            if message.get("type") == "ai"
                        ]

                        if ai_responses:
                            st.markdown('<p class="response-font">Agent Response:</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="response-font">Final Response: {ai_responses[-1]}</p>', unsafe_allow_html=True)
                        else:
                            st.warning("No AI response found in the agent output.")
                else:
                    st.error(f"Request failed with status code {response.status_code}.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message before clicking 'Submit'.")
