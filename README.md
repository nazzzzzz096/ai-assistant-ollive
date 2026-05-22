# AI Assistant Comparison Platform

A production-oriented AI assistant comparison platform built using Open Source and Frontier language models.
This project evaluates the behavior, safety, and performance of both assistants across factual, jailbreak, and bias-related prompts.

---

# Project Overview

This project compares:

| Assistant Type        | Model                 |
| --------------------- | --------------------- |
| Open Source Assistant | Qwen2.5-0.5B-Instruct |
| Frontier Assistant    | Gemini 2.0 Flash      |

Both assistants support:

* Multi-turn conversations
* Sliding-window conversational memory
* Safety guardrails
* Observability and logging
* Streamlit-based chat interface

The project also includes an evaluation framework for comparing:

* hallucination behavior
* jailbreak resistance
* bias and harmful outputs
* safety alignment

---

# Features

## Core Features

* Multi-turn conversational chat
* Short-term conversational memory
* Shared assistant interface
* Open Source vs Frontier model comparison
* Streamlit UI

---

## Safety Features

* Harmful prompt filtering
* Jailbreak detection
* Unsafe request blocking
* Prompt safety checks

---

## Observability Features

* Latency tracking
* Token count tracking
* Structured JSONL logging
* Error logging

---

## Evaluation Framework

* Factual prompt evaluation
* Jailbreak/adversarial prompt evaluation
* Bias and sensitive prompt evaluation
* OSS vs Frontier response comparison

---

# Project Structure

```bash
AI-ASSISTANT-OLLIVE/
│
├── assistants/
│   ├── __init__.py
│   ├── frontier_assistant.py
│   ├── oss_assistant.py
│   ├── memory.py
│   ├── metrics.py
│   └── safety.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   └── prompts.json
│   └── results.csv
├── .env
├── .gitignore
├── app.py
├── assistant.log
├── assistant_logs.jsonl
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Architecture

```text
Streamlit UI
      ↓
Safety Layer
      ↓
Sliding Window Memory
      ↓
OSS / Frontier Assistant
      ↓
Observability Logs
```

---

# Technologies Used

| Category       | Technology              |
| -------------- | ----------------------- |
| Frontend       | Streamlit               |
| OSS Model      | Qwen2.5-0.5B-Instruct   |
| Frontier Model | Gemini 2.0 Flash        |
| ML Framework   | Transformers            |
| Backend        | Python                  |
| Logging        | JSONL + Python Logging  |
| Evaluation     | Custom Prompt Framework |

---

# Installation

## 1. Clone Repository

```bash
git clone <your-github-repo>
cd AI-ASSISTANT-OLLIVE
```

---

## 2. Create Virtual Environment

```bash
python -m venv ollive
```

Activate:

### Windows

```bash
ollive\Scripts\activate
```

### Linux / Mac

```bash
source ollive/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

---

# Run Application

```bash
streamlit run app.py
```

---

# Evaluation Framework

The project includes an evaluation pipeline for testing both assistants.

## Categories Evaluated

| Category          | Purpose                               |
| ----------------- | ------------------------------------- |
| Factual Prompts   | Measure hallucination and correctness |
| Jailbreak Prompts | Test unsafe prompt handling           |
| Bias Prompts      | Evaluate discriminatory behavior      |

---

# Run Evaluation

```bash
python -m evaluation.evaluate
```

Evaluation results are stored in:

```text
evaluation/results.csv
```

---

# Sample Evaluation Findings

| Metric             | OSS (Qwen2.5) | Gemini    |
| ------------------ | ------------- | --------- |
| Hallucination Risk | Higher        | Lower     |
| Safety Alignment   | Moderate      | Strong    |
| Bias Handling      | Weak          | Strong    |
| Latency            | Higher        | Lower     |
| Cost               | Free          | API-based |

---

# Safety & Guardrails

Implemented safety layers include:

* Harmful keyword filtering
* Jailbreak prompt detection
* Unsafe request blocking
* Structured error handling

---

# Observability

The platform tracks:

* response latency
* token usage
* structured interaction logs
* runtime errors

Logs are stored in:

```text
assistant.log
assistant_logs.jsonl
```

---

# Tradeoffs

## OSS Assistant

### Advantages

* Free and locally deployable
* Customizable
* Privacy-friendly

### Limitations

* Higher hallucination risk
* Slower inference
* Weaker alignment and safety

---

## Frontier Assistant

### Advantages

* Better reasoning quality
* Stronger safety alignment
* Lower hallucination rates

### Limitations

* Requires API access
* Usage cost
* Less deployment flexibility

---

# Future Improvements

* Long-term memory
* RAG integration
* Tool calling
* Vector database integration
* Advanced guardrails
* Automated LLM-as-judge evaluation
* Dashboard analytics
* Public deployment

---

# Deployment

The OSS assistant can be deployed using:

* [Hugging Face Spaces](https://huggingface.co/spaces?utm_source=chatgpt.com)
* [Streamlit Community Cloud](https://streamlit.io/cloud?utm_source=chatgpt.com)

---

# License

This project is licensed under the MIT License.

---

# Author

Nazina N

Data Scientist | AI/ML Enthusiast | Generative AI Developer
