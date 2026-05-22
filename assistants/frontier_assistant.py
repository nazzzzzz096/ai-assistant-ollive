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


def generate_frontier_response(messages) -> str:
    """
    Generate a response from Gemini
    using conversation history.
    """

    try:

        if not messages:

            logger.warning(
                "Empty conversation history received."
            )

            return "Please enter a valid message."

        logger.info(
            f"Generating Gemini response using "
            f"{len(messages)} messages."
        )

        # =========================
        # Convert messages format
        # =========================
        formatted_history = []

        for msg in messages:

            role = msg["role"]
            content = msg["content"]

            # Gemini expects:
            # user / model
            gemini_role = (
                "model"
                if role == "assistant"
                else "user"
            )

            formatted_history.append({
                "role": gemini_role,
                "parts": [content]
            })

        # =========================
        # Generate Response
        # =========================
        response = model.generate_content(
            formatted_history
        )

        logger.debug(
            "Gemini API call completed."
        )

        # =========================
        # Safety / Empty Checks
        # =========================
        if not response.candidates:

            logger.warning(
                "Gemini returned no candidates."
            )

            return (
                "My response was blocked. "
                "Please rephrase your message."
            )

        try:

            assistant_response = (
                response.text.strip()
            )

        except ValueError as e:

            logger.warning(
                f"Blocked/empty response: {e}"
            )

            return (
                "The response was blocked "
                "or empty."
            )

        if not assistant_response:

            logger.warning(
                "Gemini returned empty response."
            )

            return (
                "I received an empty response."
            )

        logger.info(
            f"Gemini response generated "
            f"({len(assistant_response)} chars)."
        )

        return assistant_response

    except PermissionDenied as e:

        logger.error(
            f"Unauthorized Gemini access: {e}",
            exc_info=True
        )

        return (
            "Authentication failed. "
            "Check your GEMINI_API_KEY."
        )

    except ResourceExhausted as e:

        logger.error(
            f"Quota exceeded: {e}",
            exc_info=True
        )

        return (
            "Rate limit reached. "
            "Please wait and try again."
        )

    except InvalidArgument as e:

        logger.error(
            f"Invalid Gemini request: {e}",
            exc_info=True
        )

        return (
            "Invalid request sent to Gemini."
        )

    except RetryError as e:

        logger.error(
            f"Gemini timeout: {e}",
            exc_info=True
        )

        return (
            "The request timed out."
        )

    except GoogleAPICallError as e:

        logger.error(
            f"Gemini API error: {e}",
            exc_info=True
        )

        return (
            "A network/API error occurred."
        )

    except Exception as e:

        logger.error(
            f"Unexpected Gemini error: {e}",
            exc_info=True
        )

        return (
            "Sorry, something unexpected "
            "went wrong."
        )