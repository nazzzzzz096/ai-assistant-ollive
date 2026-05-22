import time
import json
from datetime import datetime

LOG_FILE = "assistant_logs.jsonl"

def log_interaction(
    model_name,
    user_input,
    response,
    latency,
    token_count
):

    log = {
        "timestamp": str(datetime.now()),
        "model": model_name,
        "user_input": user_input,
        "response": response,
        "latency_sec": latency,
        "token_count": token_count
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log) + "\n")