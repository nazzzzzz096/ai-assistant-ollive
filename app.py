import logging
import streamlit as st

logger = logging.getLogger(__name__)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Assistant Comparison",
    page_icon="🤖",
    layout="centered"
)

# =========================
# Import Models
# =========================
oss_ready = False
frontier_ready = False

try:
    from assistants.oss_assistant import generate_response
    oss_ready = True

except Exception as e:
    oss_error = str(e)
    logger.critical(f"Failed to import OSS assistant: {e}", exc_info=True)

try:
    from assistants.frontier_assistant import generate_frontier_response
    frontier_ready = True

except Exception as e:
    frontier_error = str(e)
    logger.critical(f"Failed to import Frontier assistant: {e}", exc_info=True)

# =========================
# UI Header
# =========================
st.title("🤖 AI Assistant Comparison Platform")
st.caption("Compare Open Source and Frontier Language Models")

# =========================
# Sidebar
# =========================
model_choice = st.sidebar.selectbox(
    "Choose Assistant",
    [
        "OSS - Qwen2.5",
        "Frontier - Gemini"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"Active Model:\n\n**{model_choice}**")

# =========================
# Availability Checks
# =========================
if "OSS" in model_choice and not oss_ready:
    st.error(
        f"OSS model failed to load.\n\n"
        f"Check `assistant.log`.\n\n"
        f"Error: `{oss_error}`"
    )
    st.stop()

if "Frontier" in model_choice and not frontier_ready:
    st.error(
        f"Frontier model failed to load.\n\n"
        f"Check `assistant.log`.\n\n"
        f"Error: `{frontier_error}`"
    )
    st.stop()

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Display Chat History
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Chat Input
# =========================
user_input = st.chat_input("Ask something...")

if user_input:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                # OSS Model
                if "OSS" in model_choice:
                    response = generate_response(user_input)

                # Frontier Model
                else:
                    response = generate_frontier_response(user_input)

                logger.info(
                    f"Response generated successfully using {model_choice}"
                )

            except Exception as e:

                response = (
                    "Sorry, an unexpected error occurred. "
                    "Please try again."
                )

                logger.error(
                    f"Unhandled exception in UI layer: {e}",
                    exc_info=True
                )

                st.error(
                    "An unexpected error occurred. "
                    "Check `assistant.log` for details."
                )

        st.markdown(response)

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })