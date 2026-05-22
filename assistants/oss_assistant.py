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

def generate_response(messages) -> str:
    """
    Generate response using conversation history.
    """

    try:

        # Build prompt messages
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            }
        ]

        # Add sliding-window memory
        for msg in messages:

            prompt_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        logger.info(
            f"Generating response using "
            f"{len(messages)} conversation messages."
        )

        # Apply chat template
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        logger.debug("Chat template applied successfully.")

    except Exception as e:

        logger.error(
            f"Failed to apply chat template: {e}",
            exc_info=True
        )

        return (
            "Sorry, I encountered an error "
            "formatting your message."
        )

    try:

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(model.device)

        logger.debug(
            f"Input tokenized. "
            f"Token count: "
            f"{inputs.input_ids.shape[-1]}"
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        logger.debug("Model inference completed.")

    except RuntimeError as e:

        logger.error(
            f"RuntimeError during inference: {e}",
            exc_info=True
        )

        return (
            "Sorry, I ran into a memory "
            "or device issue."
        )

    except Exception as e:

        logger.error(
            f"Unexpected inference error: {e}",
            exc_info=True
        )

        return (
            "Sorry, something went wrong "
            "while generating a response."
        )

    try:

        generated_tokens = outputs[0][
            inputs.input_ids.shape[-1]:
        ]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        assistant_response = response.strip()

        if not assistant_response:

            logger.warning(
                "Model returned empty response."
            )

            assistant_response = (
                "I'm not sure how to respond."
            )

        logger.info(
            f"Response generated "
            f"({len(assistant_response)} chars)."
        )

        return assistant_response

    except Exception as e:

        logger.error(
            f"Failed to decode response: {e}",
            exc_info=True
        )

        return (
            "Sorry, I couldn't decode "
            "the model response."
        )