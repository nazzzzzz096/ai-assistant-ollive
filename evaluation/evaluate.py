import json
import pandas as pd

from assistants.oss_assistant import generate_response
from assistants.frontier_assistant import (
    generate_frontier_response
)

# =========================
# Load Prompts
# =========================
with open("evaluation/prompts.json", "r") as file:
    prompts = json.load(file)

results = []

# =========================
# Run Evaluation
# =========================
for item in prompts:

    category = item["category"]
    prompt = item["prompt"]

    print(f"Testing: {prompt}")

    # Message format
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # OSS
    oss_response = generate_response(messages)

    # Frontier
    frontier_response = generate_frontier_response(messages)

    results.append({
        "category": category,
        "prompt": prompt,
        "oss_response": oss_response,
        "frontier_response": frontier_response
    })

# =========================
# Save Results
# =========================
df = pd.DataFrame(results)

df.to_csv(
    "evaluation/results.csv",
    index=False
)

print("Evaluation completed.")