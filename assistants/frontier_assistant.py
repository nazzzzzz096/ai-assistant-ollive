import logging
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import (
    GoogleAPICallError,
    RetryError,
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
)

load_dotenv()

logger = logging.getLogger(__name__)

# --- API Key Validation ---
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logger.critical("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

logger.info("GEMINI_API_KEY loaded successfully.")

# --- Model Initialization ---
MODEL_NAME = "gemini-2.0-flash"

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction="You are a helpful AI assistant."
    )
    logger.info(f"Gemini model initialized: {MODEL_NAME}")

except Exception as e:
    logger.critical(f"Failed to initialize Gemini model: {e}", exc_info=True)
    raise

conversation_history = []


def generate_frontier_response(user_input: str) -> str:
    """Generate a response from Gemini for the given user input.

    Returns the assistant's reply, or a descriptive error message string on failure.
    """

    if not user_input or not user_input.strip():
        logger.warning("Received empty or whitespace-only input.")
        return "Please enter a valid message."

    logger.info(f"User input received ({len(user_input)} chars).")

    conversation_history.append({
        "role": "user",
        "parts": [user_input]
    })

    try:
        response = model.generate_content(conversation_history)
        logger.debug("Gemini API call completed.")

        # Blocked or empty response guard
        if not response.candidates:
            logger.warning("Gemini returned no candidates. Prompt may have been blocked.")
            conversation_history.pop()
            return "My response was blocked. Please rephrase your message."

        candidate = response.candidates[0]

        if candidate.finish_reason.name not in ("STOP", "MAX_TOKENS"):
            logger.warning(f"Unexpected finish reason: {candidate.finish_reason.name}")

        # .text raises if parts are empty — access safely
        try:
            assistant_response = response.text.strip()
        except ValueError as e:
            logger.warning(f"response.text unavailable (likely blocked content): {e}")
            conversation_history.pop()
            return "The response was blocked or empty. Please rephrase your message."

        if not assistant_response:
            logger.warning("Gemini returned an empty text response.")
            conversation_history.pop()
            return "I received an empty response. Please try again."

        conversation_history.append({
            "role": "model",
            "parts": [assistant_response]
        })

        logger.info(f"Response generated ({len(assistant_response)} chars).")
        return assistant_response

    except PermissionDenied as e:
        logger.error(f"API key invalid or unauthorized: {e}", exc_info=True)
        conversation_history.pop()
        return "Authentication failed. Check your GEMINI_API_KEY."

    except ResourceExhausted as e:
        logger.error(f"Gemini quota exceeded: {e}", exc_info=True)
        conversation_history.pop()
        return "Rate limit reached. Please wait a moment and try again."

    except InvalidArgument as e:
        logger.error(f"Invalid request sent to Gemini: {e}", exc_info=True)
        conversation_history.pop()
        return "The request was invalid. Try rephrasing your message."

    except RetryError as e:
        logger.error(f"Gemini API retry timeout: {e}", exc_info=True)
        conversation_history.pop()
        return "The request timed out. Please try again."

    except GoogleAPICallError as e:
        logger.error(f"Gemini API call error: {e}", exc_info=True)
        conversation_history.pop()
        return "A network or API error occurred. Please try again."

    except Exception as e:
        logger.error(f"Unexpected error during Gemini generation: {e}", exc_info=True)
        conversation_history.pop()
        return "Sorry, something unexpected went wrong."