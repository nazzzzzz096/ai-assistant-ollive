import logging
import streamlit as st

logger = logging.getLogger(__name__)

# --- Model import (handled separately so Streamlit can show a clean error) ---
try:
    from assistants.oss_assistant import generate_response
    model_ready = True

except Exception as e:
    model_ready = False
    model_error = str(e)
    logger.critical(f"Failed to import oss_assistant: {e}", exc_info=True)


# --- UI ---
st.title("Open Source AI Assistant")

if not model_ready:
    st.error(
        f" Failed to load the AI model. Check `assistant.log` for details.\n\n"
        f"**Error:** `{model_error}`"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = generate_response(user_input)
                logger.info("Response delivered to UI successfully.")

            except Exception as e:
                response = "Sorry, an unexpected error occurred. Please try again."
                logger.error(f"Unhandled exception in UI layer: {e}", exc_info=True)
                st.error("An unexpected error occurred. Check `assistant.log` for details.")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})