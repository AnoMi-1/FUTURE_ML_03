import streamlit as st
from kplc_assistant import agent, run_kplc_chat  # Import your existing agent

# Set page configurations
st.set_page_config(page_title="KPLC Assistant", page_icon="💡", layout="wide")

# Initialize session state for chat messages and config
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    import uuid
    session_id = str(uuid.uuid4())
    st.session_state.config = {"configurable": {"thread_id": f"kplc_session_{session_id}"}}

# Header
st.title("🔌 KPLC Assistant")
st.markdown("**Ask about tariffs, connections, prepaid/postpaid meters, and more!** 😊")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about electricity connections, tokens, tariffs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching KPLC docs..."):
            # Use your existing agent!
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}, 
                config=st.session_state.config
            )
            clean_response = response["messages"][-1].content
        
        st.markdown(clean_response)
        st.session_state.messages.append({"role": "assistant", "content": clean_response})


