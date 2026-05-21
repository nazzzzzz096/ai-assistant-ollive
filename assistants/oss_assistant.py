import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --- Model Loading ---
try:
    logger.info(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer loaded successfully.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
     )
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully. Device: {model.device}")

except OSError as e:
    logger.critical(f"Model files not found or inaccessible: {e}", exc_info=True)
    raise

except RuntimeError as e:
    logger.critical(f"Runtime error while loading model (possible OOM or device issue): {e}", exc_info=True)
    raise

except Exception as e:
    logger.critical(f"Unexpected error during model loading: {e}", exc_info=True)
    raise


conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]


def generate_response(user_input: str) -> str:
    """Generate a response for the given user input.
    
    Returns the assistant's reply, or an error message string on failure.
    """

    if not user_input or not user_input.strip():
        logger.warning("Received empty or whitespace-only user input.")
        return "Please enter a message."

    logger.info(f"User input received ({len(user_input)} chars).")

    conversation_history.append({"role": "user", "content": user_input})

    try:
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.debug("Chat template applied successfully.")

    except Exception as e:
        logger.error(f"Failed to apply chat template: {e}", exc_info=True)
        conversation_history.pop()  # Roll back the appended user message
        return "Sorry, I encountered an error formatting your message."

    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        logger.debug(f"Input tokenized. Token count: {inputs.input_ids.shape[-1]}")

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        logger.debug("Model inference completed.")

    except RuntimeError as e:
        logger.error(f"RuntimeError during inference (possible OOM): {e}", exc_info=True)
        conversation_history.pop()
        return "Sorry, I ran into a memory or device error. Try a shorter message."

    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}", exc_info=True)
        conversation_history.pop()
        return "Sorry, something went wrong while generating a response."

    try:
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        assistant_response = response.split("assistant")[-1].strip()

        if not assistant_response:
            logger.warning("Model returned an empty response after decoding.")
            assistant_response = "I'm not sure how to respond to that."

        logger.info(f"Response generated ({len(assistant_response)} chars).")

    except Exception as e:
        logger.error(f"Failed to decode model output: {e}", exc_info=True)
        conversation_history.pop()
        return "Sorry, I couldn't decode the response."

    conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response