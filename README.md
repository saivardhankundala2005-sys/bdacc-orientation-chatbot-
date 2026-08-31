# 🚀 BDACC AI Lab — Step-by-Step AI Architecture Demo & Assistant

An interactive, presenter-friendly web application created for the **BDACC (Big Data Analytics & Consulting Cell)** orientation at **National Institute of Technology (NIT), Warangal**.

This app walks students and developers through the fundamental building blocks of modern AI development using a **"Show Code → Run → Output + Live Architecture Flow"** pattern.

---

## 🎯 Core Concept & Teaching Modules

Instead of a black-box AI chatbot interface, this application reveals the exact 10–20 line Python code snippet for each architectural stage, executes it live on stage, and displays the visual data flow architecture:

1. **Step 1 — Connect to Gemini API**: Loading `.env` and initializing direct connection to Google Gemini LLM (`gemini-2.5-flash`).
2. **Step 2 — Add LangChain Orchestration**: Prompt templates and runnable chains (`prompt | llm`).
3. **Step 3 — BDACC System Prompt Grounding**: Directing model behavior with system prompts (*"We didn't train a new model. We changed the instructions."*).
4. **Step 4 — Think Like a PM (Persona Toggle)**: Dynamic prompt engineering switching into Product Manager persona (RICE, JTBD, MoSCoW).
5. **Step 5 — RAG Document Ingestion**: Pre-upload failure -> document loading/chunking/indexing in Chroma Vector Store -> post-upload retrieval success with citations (*"The model didn't get smarter. We gave it a document to look things up in."*).
6. **Step 6 — Conversational Memory**: Retaining multi-turn student dialogue context using LangChain chat history buffers.
7. **Step 7 — Final Combined AI Playground**: Unified production application combining **BDACC Knowledge + RAG Search + Conversational Memory + PM Persona Toggle**.

---

## 🎨 UI & Presenter Controls Features

- **Sleek High-Contrast Dark Theme**: Custom slate theme designed for maximum visibility on auditorium screens and projectors.
- **Audience Font Zoom Toolbar**: One-click font scaling (`Standard 15px`, `Audience 19px`, `Large 23px`, `XL 27px`) for audience readability.
- **Code Cell Expansion**: Click `↔️ Expand Code` to dynamically expand all code blocks without vertical scrollbars or truncation.
- **Side Control Panel**: Intuitive step navigation to jump between any of the 7 architecture stages seamlessly.

---

## 🛠️ Tech Stack

- **UI Framework**: Gradio 6 (`gr.Blocks`)
- **Orchestration**: LangChain (`langchain`, `langchain-core`, `langchain-community`)
- **LLM Engine**: Google Gemini API (`gemini-2.5-flash` via `langchain-google-genai`)
- **Vector Database**: Chroma (`langchain-chroma`)
- **Embedding Engine**: Google Generative AI Embeddings (`gemini-embedding-001`)
- **Document Loaders**: `pypdf`, `TextLoader`, `RecursiveCharacterTextSplitter`

---

## 📋 Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/saivardhankundala2005-sys/bdacc-orientation-chatbot-.git
cd bdacc-orientation-chatbot-
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=AIzaSy...YourActualGeminiApiKey
```

---

## 🚀 Running the Application

Run the Python application:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## ⚙️ Model Customization

The active model names are configured at the top of [`app.py`](file:///c:/Users/saiva/Downloads/bdacc%20chotbot%20demonstration/app.py):

```python
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
```

---

## 🔍 Troubleshooting & Live Presentation Safety

| Feature / Issue | Behavior & Safety Net |
|---|---|
| **No API Key / Quota Limits** | Built-in **Fallback Mock Mode** guarantees the presentation never crashes live on stage! |
| **Model Version Selection** | Standardized on `gemini-2.5-flash` and `gemini-embedding-001` for fast REST API response times. |
| **RAG Document Ingestion** | Supports `.txt`, `.pdf`, and `.md` files (includes `sample_bdacc_facts.txt` for instant testing). |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
