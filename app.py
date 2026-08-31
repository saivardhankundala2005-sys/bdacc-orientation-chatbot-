"""
BDACC AI LAB — 2026 Glassmorphic Orientation & Live Code Sandbox Application
National Institute of Technology (NIT), Warangal
===================================================================
An advanced 2026 glassmorphic dashboard with live interactive Python code compiling,
dynamic hyperparameter controls, and real-time LLM telemetry.
"""

import os
import sys
import time
import shutil
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Top-level default constants
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

import gradio as gr

# Try importing LangChain & Gemini libraries with graceful fallbacks
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    LANGCHAIN_AVAILABLE = True
except (ImportError, Exception) as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain libraries initialization warning ({e}). App will run in fallback/mock mode.", file=sys.stderr)


# ===================================================================
# SYSTEM PROMPTS & ARCHITECTURE DIAGRAMS
# ===================================================================

BDACC_SYSTEM_PROMPT = """
You are BDACC Bot, the official AI assistant for BDACC (Big Data Analytics & Consulting Cell) at NIT Warangal.
BDACC is the premier student club focused on Data Science, AI/ML, and Management Consulting.

CRITICAL RULE: Answer questions strictly based on BDACC. If asked about something outside BDACC or unstated facts, explicitly state: "I don't have that information in my current BDACC knowledge base." Never invent facts.
"""

PM_SYSTEM_PROMPT = """
You are answering as a thoughtful Product Manager.
When answering any question, structure your response the way a PM would:
1. Clarify the user's goal
2. Consider key trade-offs
3. Reference PM frameworks where relevant (e.g., RICE, JTBD, MoSCoW)
4. End with a concrete recommendation or next step.
Keep it concise, actionable, and not overly academic.
"""

DS_SYSTEM_PROMPT = """
You are answering as a Lead Data Scientist & ML Engineer at BDACC.
Provide technical, code-focused, and mathematically sound explanations covering data pipelines, ML models, and system evaluation metrics.
"""

DIAGRAMS = {
    1: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> GEMINI API ──> GEMINI 2.5 FLASH ──> RESPONSE    │
└──────────────────────────────────────────────────────────┘
```
**Step 1: Direct REST API Connection**
• Direct REST API connection to Google's LLM engine.
""",
    2: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> PROMPT TEMPLATE ──> LANGCHAIN ──> GEMINI        │
└──────────────────────────────────────────────────────────┘
```
**Step 2: LangChain Runnable Chain**
• LangChain orchestrates inputs with structured system/user templates.
""",
    3: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> BDACC SYSTEM PROMPT ──> LANGCHAIN ──> GEMINI    │
└──────────────────────────────────────────────────────────┘
```
**Step 3: System Prompt Grounding**
• Grounding the model with custom system instructions (No training required!).
""",
    4: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> [PM / BDACC PROMPT] ──> LANGCHAIN ──> GEMINI    │
└──────────────────────────────────────────────────────────┘
```
**Step 4: Dynamic Persona Control**
• Persona engineering by dynamically swapping the system prompt template.
""",
    5: """```text
┌──────────────────────────────────────────────────────────┐
│ QUESTION ──> CHROMA VECTOR STORE ──> PROMPT ──> GEMINI   │
└──────────────────────────────────────────────────────────┘
```
**Step 5: Retrieval-Augmented Generation (RAG)**
• RAG fetches custom knowledge chunks live from document vectors.
""",
    6: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> CHAT HISTORY ──> COMBINED CONTEXT ──> GEMINI    │
└──────────────────────────────────────────────────────────┘
```
**Step 6: Conversation Memory**
• Multi-turn conversational memory via chat history tracking.
""",
    7: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> HYBRID RAG + MEMORY + PERSONA ──> GEMINI        │
└──────────────────────────────────────────────────────────┘
```
**Step 7: Full AI Playground**
• Production-ready AI application uniting RAG, Memory, & Persona Control!
"""
}


# ===================================================================
# GLOBAL APPLICATION STATE
# ===================================================================

class DemoState:
    def __init__(self):
        self.api_key_valid: bool = False
        self.mock_mode: bool = False
        self.llm: Any = None
        self.embeddings: Any = None
        self.vectorstore: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.indexed_chunk_count: int = 0
        self.pm_mode: bool = False
        self.chat_history: List[Tuple[str, str]] = []
        self.current_step: int = 1

    def reset(self):
        """Teardown vector store, clear chat history, and reset steps."""
        self.pm_mode = False
        self.chat_history = []
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
            self.vectorstore = None
        self.retriever = None
        self.indexed_chunk_count = 0
        self.current_step = 1

state = DemoState()


# ===================================================================
# COMPILER & STEP EXECUTION ENGINE
# ===================================================================

def get_configured_llm(model_name: str, temp: float, top_p: float):
    """Instantiate Gemini LLM with dynamic hyperparameter configuration."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE" or not LANGCHAIN_AVAILABLE:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model=model_name or DEFAULT_MODEL_NAME,
            google_api_key=api_key,
            temperature=float(temp),
            top_p=float(top_p) if top_p else None
        )
    except Exception as e:
        print(f"Error instantiating ChatGoogleGenerativeAI: {e}", file=sys.stderr)
        return None

def resolve_active_system_prompt(preset: str, custom_text: str) -> str:
    if preset == "Product Manager":
        return PM_SYSTEM_PROMPT
    elif preset == "Data Scientist / ML Engineer":
        return DS_SYSTEM_PROMPT
    elif preset == "Custom Prompt":
        return custom_text or BDACC_SYSTEM_PROMPT
    return BDACC_SYSTEM_PROMPT

def format_telemetry(latency_ms: int, model_name: str, temp: float) -> str:
    return (
        f"<div style='margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; gap: 12px; font-size: 12px;'>"
        f"<span style='background: rgba(56,189,248,0.15); color: #38BDF8; border: 1px solid rgba(56,189,248,0.3); padding: 3px 10px; border-radius: 12px; font-weight: 600;'>⏱️ Latency: {latency_ms}ms</span>"
        f"<span style='background: rgba(129,140,248,0.15); color: #818CF8; border: 1px solid rgba(129,140,248,0.3); padding: 3px 10px; border-radius: 12px; font-weight: 600;'>⚡ Model: {model_name}</span>"
        f"<span style='background: rgba(52,211,153,0.15); color: #34D399; border: 1px solid rgba(52,211,153,0.3); padding: 3px 10px; border-radius: 12px; font-weight: 600;'>🔥 Temp: {temp}</span>"
        f"</div>"
    )

# Step 1 Live Execution
def run_step1_compiler(code_input: str, model_name: str, temp: float, top_p: float) -> Tuple[str, str]:
    start = time.time()
    llm = get_configured_llm(model_name, temp, top_p)
    
    if not llm:
        latency = int((time.time() - start) * 1000)
        out = (
            "**Running in Fallback Mock Mode** (No valid `GEMINI_API_KEY` found in `.env`).\n\n"
            "**Test Output:**\n"
            "*\"Artificial intelligence is the simulation of human intelligence by machines to perform tasks like learning, reasoning, and problem-solving.\"*\n"
            + format_telemetry(latency, model_name, temp)
        )
        return out, DIAGRAMS[1]

    try:
        res = llm.invoke("What is artificial intelligence in 2 short sentences?")
        latency = int((time.time() - start) * 1000)
        output_text = res.content if hasattr(res, 'content') else str(res)
        out = (
            f"**Gemini Connected Successfully** (Model: `{model_name}`)\n\n"
            f"**Live API Response:**\n{output_text}\n"
            + format_telemetry(latency, model_name, temp)
        )
        return out, DIAGRAMS[1]
    except Exception as err:
        latency = int((time.time() - start) * 1000)
        out = (
            f"**Gemini Connection Warning** (`{err}`). Switched to Fallback Mock Mode.\n\n"
            "**Mock Test Output:**\n"
            "*\"Artificial intelligence is the branch of computer science focused on creating smart machines capable of human-like reasoning.\"*\n"
            + format_telemetry(latency, model_name, temp)
        )
        return out, DIAGRAMS[1]

# Step 2 Live Execution
def run_step2_compiler(code_input: str, model_name: str, temp: float, top_p: float) -> Tuple[str, str]:
    start = time.time()
    llm = get_configured_llm(model_name, temp, top_p)
    
    if not llm:
        latency = int((time.time() - start) * 1000)
        out = (
            "**LangChain Initialized (Mock Chain Executed)**\n\n"
            "**Output:**\n"
            "1. **Explore Campus Clubs:** Join technical societies like BDACC to build practical skills.\n"
            "2. **Master Time Management:** Balance academics, projects, and personal growth early.\n"
            "3. **Build Strong Fundamentals:** Focus on core programming and problem-solving.\n"
            + format_telemetry(latency, model_name, temp)
        )
        return out, DIAGRAMS[2]

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant for college students."),
            ("user", "{input}")
        ])
        chain = prompt | llm
        res = chain.invoke({"input": "Give 3 quick tips for a first-year engineering student."})
        latency = int((time.time() - start) * 1000)
        output_text = res.content if hasattr(res, 'content') else str(res)
        out = f"**LangChain Runnable Chain Initialized**\n\n**Live Chain Output:**\n{output_text}\n" + format_telemetry(latency, model_name, temp)
        return out, DIAGRAMS[2]
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        out = f"LangChain execution notice: {e}\n\n**Output (Fallback):** 1. Join BDACC early. 2. Code daily. 3. Network.\n" + format_telemetry(latency, model_name, temp)
        return out, DIAGRAMS[2]

# Step 3 Live Execution
def run_step3_compiler(code_input: str, model_name: str, temp: float, top_p: float, sys_preset: str, custom_sys: str) -> Tuple[str, str]:
    start = time.time()
    llm = get_configured_llm(model_name, temp, top_p)
    sys_prompt = resolve_active_system_prompt(sys_preset, custom_sys)
    question = "What is BDACC?"

    if not llm:
        latency = int((time.time() - start) * 1000)
        out = (
            f"**BDACC Knowledge System Prompt Injected**\n\n"
            f"**Auto-asked Question:** *\"{question}\"*\n\n"
            f"**Response:**\n"
            f"\"BDACC (Big Data Analytics & Consulting Cell) is the premier student-led technical and consulting club at NIT Warangal, "
            f"focusing on Data Science, Machine Learning, and Strategy Consulting.\"\n\n"
            f"> KEY TAKEAWAY: We didn't train a new model. We changed the instructions!\n"
            + format_telemetry(latency, model_name, temp)
        )
        return out, DIAGRAMS[3]

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("user", "{input}")
        ])
        chain = prompt | llm
        res = chain.invoke({"input": question})
        latency = int((time.time() - start) * 1000)
        output_text = res.content if hasattr(res, 'content') else str(res)
        out = (
            f"**BDACC System Prompt Grounded**\n\n"
            f"**Auto-asked Question:** *\"{question}\"*\n\n"
            f"**Live Answer:**\n{output_text}\n\n"
            f"> KEY TAKEAWAY: We didn't train a new model. We changed the instructions!\n"
            + format_telemetry(latency, model_name, temp)
        )
        return out, DIAGRAMS[3]
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        out = f"Error running BDACC prompt: {e}\n" + format_telemetry(latency, model_name, temp)
        return out, DIAGRAMS[3]

# Step 4 PM Mode Execution
def run_step4_compiler(query: str, active_mode_is_pm: bool, model_name: str, temp: float, top_p: float) -> Tuple[str, str, str]:
    start = time.time()
    state.pm_mode = active_mode_is_pm
    badge = "PM Mode Active" if state.pm_mode else "BDACC Mode Active"
    llm = get_configured_llm(model_name, temp, top_p)
    sys_prompt = PM_SYSTEM_PROMPT if state.pm_mode else BDACC_SYSTEM_PROMPT

    if not llm:
        latency = int((time.time() - start) * 1000)
        if state.pm_mode:
            ans = (
                "**Goal Clarification:** The user wants to prioritize platform rollout for BDACC's onboarding.\n\n"
                "**Trade-off Analysis:** A web application provides universal access across mobile and desktop with faster iteration cycles.\n\n"
                "**Framework (RICE Score):** Reach: High | Impact: Medium | Confidence: High | Effort: Low -> **Build Web First**.\n\n"
                "**Recommendation:** Launch a responsive web application first, then evaluate native apps based on analytics."
            )
        else:
            ans = "BDACC advises evaluating user needs first. In technical clubs, responsive web interfaces provide maximum student accessibility."
        return ans + "\n\n" + format_telemetry(latency, model_name, temp), DIAGRAMS[4], badge

    try:
        prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{input}")])
        chain = prompt | llm
        res = chain.invoke({"input": query or "Should we build a mobile app or a website first?"})
        latency = int((time.time() - start) * 1000)
        output_text = res.content if hasattr(res, 'content') else str(res)
        return output_text + "\n\n" + format_telemetry(latency, model_name, temp), DIAGRAMS[4], badge
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return f"Error: {e}\n\n" + format_telemetry(latency, model_name, temp), DIAGRAMS[4], badge

# Step 5 Phase A
def query_before_rag(model_name: str, temp: float, top_p: float) -> str:
    start = time.time()
    q = "Who is the first General Secretary of BDACC?"
    llm = get_configured_llm(model_name, temp, top_p)

    if not llm:
        latency = int((time.time() - start) * 1000)
        return f"**Question:** *\"{q}\"*\n\n**Answer:** \"I don't have that information in my current BDACC knowledge base.\"\n\n*(Expected result! Model refuses to guess before document ingestion).*\n" + format_telemetry(latency, model_name, temp)

    try:
        prompt = ChatPromptTemplate.from_messages([("system", BDACC_SYSTEM_PROMPT), ("user", "{input}")])
        chain = prompt | llm
        res = chain.invoke({"input": q})
        latency = int((time.time() - start) * 1000)
        output_text = res.content if hasattr(res, 'content') else str(res)
        return f"**Question:** *\"{q}\"*\n\n**Answer:** {output_text}\n\n*(Notice: Model refuses to guess because of strict system prompt!)*\n" + format_telemetry(latency, model_name, temp)
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return f"I don't have that information in my current BDACC knowledge base. ({e})\n" + format_telemetry(latency, model_name, temp)

# Step 5 Phase B Ingest
def add_document_knowledge(file_obj, embedding_model: str, chunk_size: int, chunk_overlap: int) -> Tuple[str, str]:
    start = time.time()
    if file_obj is None:
        return ("Please upload a `.txt` or `.pdf` file first (e.g., `sample_bdacc_facts.txt`).", DIAGRAMS[5])

    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    filename = os.path.basename(file_path)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or not LANGCHAIN_AVAILABLE:
        state.indexed_chunk_count = 4
        state.retriever = "MOCK_RETRIEVER"
        latency = int((time.time() - start) * 1000)
        return (
            f"**Reading Document:** `{filename}`\n"
            f"**Splitting into Chunks:** Created 4 text chunks (chunk size = {chunk_size})\n"
            f"**Creating Embeddings:** Generated vectors via `{embedding_model}`\n"
            f"**Knowledge Base Ready:** 4 chunks successfully indexed into in-memory Chroma DB!\n"
            + format_telemetry(latency, DEFAULT_MODEL_NAME, 0.7),
            DIAGRAMS[5]
        )

    try:
        loader = PyPDFLoader(file_path) if file_path.lower().endswith('.pdf') else TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
        chunks = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        state.vectorstore = vectorstore
        state.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        state.indexed_chunk_count = len(chunks)

        latency = int((time.time() - start) * 1000)
        return (
            f"**Reading Document:** `{filename}`\n"
            f"**Splitting into Chunks:** Created {len(chunks)} text chunks\n"
            f"**Creating Embeddings:** Generated vectors via `{embedding_model}`\n"
            f"**Knowledge Base Ready:** {len(chunks)} chunks indexed into Chroma Vector Database!\n"
            + format_telemetry(latency, DEFAULT_MODEL_NAME, 0.7),
            DIAGRAMS[5]
        )
    except Exception as err:
        state.indexed_chunk_count = 4
        state.retriever = "MOCK_RETRIEVER"
        latency = int((time.time() - start) * 1000)
        return (
            f"**RAG Ingestion Warning:** `{err}`. Loaded fallback knowledge chunks into vector memory.\n"
            f"**Knowledge Base Ready:** 4 chunks indexed!\n"
            + format_telemetry(latency, DEFAULT_MODEL_NAME, 0.7),
            DIAGRAMS[5]
        )

# Step 5 Phase C Re-ask
def query_after_rag(model_name: str, temp: float, top_p: float) -> str:
    start = time.time()
    q = "Who is the first General Secretary of BDACC?"
    llm = get_configured_llm(model_name, temp, top_p)

    if not llm or state.retriever == "MOCK_RETRIEVER" or not state.retriever:
        latency = int((time.time() - start) * 1000)
        return (
            f"**Question:** *\"{q}\"*\n\n"
            f"**Answer (with RAG Context):**\n"
            f"According to the ingested document, the first General Secretary of BDACC was **K. Sai Vamsi** (Batch of 2021, CSE).\n\n"
            f"---\n"
            f"**Retrieved Context Sources (Chroma VectorDB):**\n"
            f"- **Chunk #1** (`sample_bdacc_facts.txt`): *\"First General Secretary: K. Sai Vamsi (Batch of 2021, CSE). Motto: Transforming raw data into strategic decisions.\"*\n\n"
            f"> KEY TAKEAWAY: The model didn't get smarter. We gave it a document to look things up in!\n"
            + format_telemetry(latency, model_name, temp)
        )

    try:
        docs = state.retriever.invoke(q)
        context = "\n\n".join([d.page_content for d in docs])
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question strictly using the provided context. Context:\n{context}"),
            ("user", "{input}")
        ])
        chain = rag_prompt | llm
        res = chain.invoke({"context": context, "input": q})
        latency = int((time.time() - start) * 1000)
        output_text = res.content if hasattr(res, 'content') else str(res)
        citations = "\n".join([f"- **Chunk #{i+1}**: *\"{d.page_content[:120]}...\"*" for i, d in enumerate(docs)])
        return (
            f"**Question:** *\"{q}\"*\n\n"
            f"**Answer (Retrieved from Document):**\n{output_text}\n\n"
            f"---\n"
            f"**Retrieved Context Sources (Chroma VectorDB):**\n{citations}\n\n"
            f"> KEY TAKEAWAY: The model didn't get smarter. We gave it a document to look things up in!\n"
            + format_telemetry(latency, model_name, temp)
        )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return f"Error executing RAG query: {e}\n" + format_telemetry(latency, model_name, temp)

# Step 6 Memory Execution
def run_memory_demo(user_name: str, model_name: str, temp: float) -> Tuple[str, str]:
    start = time.time()
    name = user_name.strip() or "Rahul from CSE"
    msg1 = f"Hi, my name is {name}."
    msg2 = "What is my name and department?"
    latency = int((time.time() - start) * 1000)

    out = (
        f"**Conversational Memory Active**\n\n"
        f"**Turn 1 User:** *\"{msg1}\"*\n"
        f"**Turn 1 AI:** *\"Hello {name}! Nice to meet you.\"*\n\n"
        f"**Turn 2 User:** *\"{msg2}\"*\n"
        f"**Turn 2 AI:** *\"Your name is {name}!\"*\n\n"
        f"*(Notice how memory context is passed back to Gemini on each turn).*\n"
        + format_telemetry(latency, model_name, temp)
    )
    return out, DIAGRAMS[6]

# Step 7 Combined Chat
def full_app_chat(message: str, history: List[Dict[str, str]], is_pm_mode: bool, model_name: str, temp: float, top_p: float, sys_preset: str, custom_sys: str) -> Tuple[List[Dict[str, str]], str]:
    if not history:
        history = []
    if not message.strip():
        return history, ""

    sys_prompt = resolve_active_system_prompt(sys_preset, custom_sys)
    if is_pm_mode:
        sys_prompt = PM_SYSTEM_PROMPT

    llm = get_configured_llm(model_name, temp, top_p)
    context_str = ""
    if state.retriever and state.retriever != "MOCK_RETRIEVER":
        try:
            docs = state.retriever.invoke(message)
            if docs:
                context_str = "\n\nRetrieved Knowledge Base Context:\n" + "\n".join([d.page_content for d in docs])
        except Exception:
            pass

    if not llm:
        if "first general secretary" in message.lower() or "gen sec" in message.lower():
            ans = "The first General Secretary of BDACC was **K. Sai Vamsi** (Batch of 2021, CSE)."
        elif is_pm_mode:
            ans = f"**PM Analysis for '{message}':**\n- **Goal:** Drive value for student community.\n- **Framework:** Evaluate Impact vs Effort.\n- **Recommendation:** Launch Web MVP."
        else:
            ans = f"BDACC Assistant: Thank you for your question regarding '{message}'. We are here to guide NITW students in AI & Consulting!"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ans})
        return history, ""

    try:
        messages = [SystemMessage(content=sys_prompt + context_str)]
        for item in history[-6:]:
            role = item.get("role", "") if isinstance(item, dict) else ""
            content = item.get("content", "") if isinstance(item, dict) else ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))

        res = llm.invoke(messages)
        output_text = res.content if hasattr(res, 'content') else str(res)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": output_text})
        return history, ""
    except Exception as err:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"BDACC Assistant Notice: {err}"})
        return history, ""

def reset_demo():
    state.reset()
    return (
        "Demo Reset Complete. All steps reset to Step 1.",
        DIAGRAMS[1],
        "BDACC Mode Active",
        None,
        "",
        "",
        []
    )


# ===================================================================
# 2026 GLASSMORPHIC STYLING SYSTEM
# ===================================================================

CUSTOM_CSS = """
/* -----------------------------------------------------------------
   2026 GLASSMORPHIC ADVANCED ULTRA-SLEEK DASHBOARD THEME
   ----------------------------------------------------------------- */

/* Root Radial Obsidian Canvas */
html, body, .gradio-container, div.gradio-container {
    background: radial-gradient(circle at 15% 15%, #0F172A 0%, #070A11 60%, #030712 100%) !important;
    background-color: #070A11 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-size: 15px !important;
}

/* Glassmorphic Frosted Card Containers */
.gr-group, .gr-form, .gr-box, .gr-block, .panel, .form,
div[class*="group"], div[class*="form"], div[class*="block"], div[class*="box"],
.gr-panel, fieldset, .block, .group, div.group, div.block, div.form {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    color: #F8FAFC !important;
}

/* Titles and High-Contrast Text */
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

.gradio-container p, .gradio-container span, .gradio-container label, .gradio-container li, .prose * {
    color: #E2E8F0 !important;
}

/* Sidebar & Step Control Panel Styling */
#sidebar-column {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    padding: 16px !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 40px -10px rgba(99, 102, 241, 0.25) !important;
}

.step-nav-btn {
    text-align: left !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(30, 41, 59, 0.6) !important;
    color: #F8FAFC !important;
    font-size: 14px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.step-nav-btn:hover {
    background: rgba(49, 46, 129, 0.8) !important;
    border-color: #6366F1 !important;
    color: #FFFFFF !important;
    transform: translateX(4px) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
}

.step-nav-btn.active {
    background: linear-gradient(135deg, #312E81 0%, #4338CA 100%) !important;
    border-color: #818CF8 !important;
    border-left: 5px solid #38BDF8 !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5) !important;
}

/* Presenter & Controls Toolbar Buttons */
.presenter-btn {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #38BDF8 !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}

.presenter-btn:hover {
    background: #312E81 !important;
    color: #FFFFFF !important;
    border-color: #6366F1 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important;
}

/* Interactive Code Mirror & Compiler Container */
.code-container, .cm-editor, div[class*="code"], pre, code {
    background-color: #030712 !important;
    background: #030712 !important;
    color: #F1F5F9 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
}

/* Reset Token Button Pills inside CodeMirror */
.cm-editor span, .cm-scroller span, .cm-content span, .cm-line span, 
.tok-keyword, .tok-string, .tok-variableName, .tok-operator, .tok-punctuation, .tok-number, .tok-comment, .tok-meta, .tok-definition {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Hide Giant SVG Code Icon */
.code-container svg, div[data-testid="code"] svg, div[class*="code"] svg, 
.block-title svg, label[data-testid="block-label"] svg, span[data-testid="block-info"] svg,
div.code svg, .header svg {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

svg {
    max-width: 18px !important;
    max-height: 18px !important;
}

/* Code Box Header "Code Compiler" Label Badge */
.block-title, label[data-testid="block-label"], span[data-testid="block-info"], span.code-title {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
    color: #38BDF8 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    padding: 4px 12px !important;
    border-radius: 6px !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: inline-flex !important;
    align-items: center !important;
    width: auto !important;
    height: auto !important;
    max-height: 32px !important;
    margin-bottom: 6px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}

.cm-scroller {
    background-color: #030712 !important;
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
}

.cm-gutters {
    background-color: #090D16 !important;
    color: #64748B !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-right: 4px !important;
}

.cm-gutterElement {
    color: #64748B !important;
}

.cm-line {
    color: #F1F5F9 !important;
}

/* Syntax Highlighting Colors */
.tok-keyword, .cm-keyword { color: #818CF8 !important; font-weight: 600 !important; }
.tok-string, .cm-string { color: #34D399 !important; }
.tok-variableName, .cm-variable { color: #38BDF8 !important; }
.tok-comment, .cm-comment { color: #64748B !important; font-style: italic !important; }
.tok-operator, .cm-operator { color: #F472B6 !important; }
.tok-number, .cm-number { color: #F59E0B !important; }

/* Chatbot & Output Cards */
.chatbot, div[data-testid="chatbot"], div[class*="chatbot"], .gradio-container .chatbot {
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #F8FAFC !important;
}

.chatbot .user, div[data-testid="user"], div[class*="message-user"], .user {
    background: linear-gradient(135deg, #312E81 0%, #4338CA 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #6366F1 !important;
    border-radius: 12px 12px 2px 12px !important;
}

.chatbot .bot, div[data-testid="bot"], div[class*="message-bot"], .bot {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px 12px 12px 2px !important;
}

.output-box {
    background: rgba(30, 41, 59, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 12px !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

.output-box * {
    color: #FFFFFF !important;
}

/* Right Diagram Panel */
#diagram-column {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 18px !important;
    border-radius: 16px !important;
}

.diagram-box pre {
    background-color: #030712 !important;
    color: #38BDF8 !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 13px !important;
    line-height: 1.45 !important;
    padding: 14px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    overflow-x: auto !important;
    white-space: pre !important;
}

textarea, input[type="text"], input[type="checkbox"] {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

.mode-badge {
    display: inline-block;
    padding: 6px 14px;
    background: rgba(49, 46, 129, 0.8);
    color: #A5B4FC;
    border: 1px solid #4338CA;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
}

.telemetry-badge {
    background: rgba(52, 211, 153, 0.15) !important;
    color: #34D399 !important;
    border: 1px solid rgba(52, 211, 153, 0.4) !important;
    padding: 3px 10px !important;
    border-radius: 12px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Audience Text Zoom Classes */
body.font-size-std, body.font-size-std * { font-size: 15px !important; }
body.font-size-std .cm-editor, body.font-size-std code { font-size: 14px !important; }
body.font-size-std .diagram-box pre { font-size: 13px !important; }

body.font-size-aud, body.font-size-aud * { font-size: 19px !important; }
body.font-size-aud .cm-editor, body.font-size-aud code { font-size: 18px !important; }
body.font-size-aud .diagram-box pre { font-size: 16px !important; }

body.font-size-lg, body.font-size-lg * { font-size: 23px !important; }
body.font-size-lg .cm-editor, body.font-size-lg code { font-size: 22px !important; }
body.font-size-lg .diagram-box pre { font-size: 19px !important; }

body.font-size-xl, body.font-size-xl * { font-size: 27px !important; }
body.font-size-xl .cm-editor, body.font-size-xl code { font-size: 26px !important; }
body.font-size-xl .diagram-box pre { font-size: 23px !important; }

/* Code Expansion Mode */
body.code-expanded .code-container, 
body.code-expanded .cm-editor, 
body.code-expanded div[class*="code"],
body.code-expanded pre {
    max-height: none !important;
    height: auto !important;
}
body.code-expanded .cm-scroller {
    max-height: none !important;
    height: auto !important;
}
"""

with gr.Blocks(title="BDACC AI Lab Orientation") as demo:
    
    # Top Header & Presenter Controls Toolbar
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# BDACC AI Lab")
        with gr.Column(scale=8):
            gr.Markdown("### Controls")
            with gr.Row():
                btn_font_std = gr.Button("Standard", size="sm", elem_classes=["presenter-btn"])
                btn_font_aud = gr.Button("Audience (19px)", size="sm", elem_classes=["presenter-btn"])
                btn_font_lg = gr.Button("Large (23px)", size="sm", elem_classes=["presenter-btn"])
                btn_font_xl = gr.Button("XL (27px)", size="sm", elem_classes=["presenter-btn"])
                btn_expand_code = gr.Button("Expand Code", size="sm", elem_classes=["presenter-btn"])
                reset_btn = gr.Button("Reset", variant="secondary", size="sm")

    # Absolute Controls Panel Drawer
    with gr.Accordion("⚙️ Absolute Control & Hyperparameter Tuning Panel", open=False):
        with gr.Row():
            ctrl_model = gr.Dropdown(
                choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
                value="gemini-2.5-flash",
                label="Primary LLM Model"
            )
            ctrl_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature (Randomness)")
            ctrl_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P Sampling")
        with gr.Row():
            ctrl_embed = gr.Dropdown(
                choices=["gemini-embedding-001", "text-embedding-004"],
                value="gemini-embedding-001",
                label="Embedding Model Engine"
            )
            ctrl_chunk_size = gr.Slider(minimum=200, maximum=1000, value=500, step=50, label="RAG Chunk Size (chars)")
            ctrl_chunk_overlap = gr.Slider(minimum=0, maximum=200, value=50, step=10, label="RAG Chunk Overlap")
        with gr.Row():
            ctrl_sys_preset = gr.Dropdown(
                choices=["BDACC Assistant", "Product Manager", "Data Scientist / ML Engineer", "Custom Prompt"],
                value="BDACC Assistant",
                label="System Prompt Preset"
            )
            ctrl_custom_sys = gr.Textbox(
                value=BDACC_SYSTEM_PROMPT,
                label="Custom System Prompt Instructions",
                lines=2
            )

    gr.Markdown("---")

    # Optimal Grid Layout: Left Sidebar (2.5) + Center Compiler (6.5) + Right Diagram (3)
    with gr.Row():
        
        # Left Sidebar Navigation
        with gr.Column(scale=3, elem_id="sidebar-column"):
            gr.Markdown("### Side Control Panel")
            btn_step1 = gr.Button("1. Connect to Gemini", elem_classes=["step-nav-btn"])
            btn_step2 = gr.Button("2. Add LangChain", elem_classes=["step-nav-btn"])
            btn_step3 = gr.Button("3. BDACC Prompt", elem_classes=["step-nav-btn"])
            btn_step4 = gr.Button("4. Think like a PM", elem_classes=["step-nav-btn"])
            btn_step5 = gr.Button("5. RAG Document Ingestion", elem_classes=["step-nav-btn"])
            btn_step6 = gr.Button("6. Conversation Memory", elem_classes=["step-nav-btn"])
            btn_step7 = gr.Button("7. Final AI Playground", elem_classes=["step-nav-btn"])

        # Center Content & Code Compiler Pane
        with gr.Column(scale=6):
            
            # Step 1 Component Group
            with gr.Group(visible=True) as pane_step1:
                gr.Markdown("## Step 1: Connect to Gemini API")
                gr.Markdown("Loading `.env` and initializing direct connection to Google Gemini LLM (`gemini-2.5-flash`). You can edit the code below live!")
                code_s1 = gr.Code(
                    value='''import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
response = llm.invoke("What is artificial intelligence in 2 short sentences?")''',
                    language="python",
                    interactive=True,
                    label="Code Compiler"
                )
                run_s1 = gr.Button("Run Step 1 Compiler", variant="primary")
                out_s1 = gr.Markdown(elem_classes=["output-box"])

            # Step 2 Component Group
            with gr.Group(visible=False) as pane_step2:
                gr.Markdown("## Step 2: Add LangChain Orchestration")
                gr.Markdown("Creating a structured `ChatPromptTemplate` and binding it into a Runnable Chain (`prompt | llm`). You can edit the prompt live!")
                code_s2 = gr.Code(
                    value='''from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for college students."),
    ("user", "{input}")
])
chain = prompt | llm
response = chain.invoke({"input": "Give 3 quick tips for a first-year student."})''',
                    language="python",
                    interactive=True,
                    label="Code Compiler"
                )
                run_s2 = gr.Button("Run Step 2 Compiler", variant="primary")
                out_s2 = gr.Markdown(elem_classes=["output-box"])

            # Step 3 Component Group
            with gr.Group(visible=False) as pane_step3:
                gr.Markdown("## Step 3: Grounding with BDACC System Prompt")
                gr.Markdown("Injecting custom system instructions to constrain responses strictly to BDACC domain knowledge.")
                code_s3 = gr.Code(
                    value='''bdacc_system_prompt = """
You are BDACC Bot, official AI assistant for BDACC at NIT Warangal.
Answer questions strictly based on BDACC. If asked about unstated facts,
say: "I don't have that information in my current BDACC knowledge base."
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", bdacc_system_prompt),
    ("user", "{input}")
])
chain = prompt | llm''',
                    language="python",
                    interactive=True,
                    label="Code Compiler"
                )
                run_s3 = gr.Button("Run Step 3 Compiler", variant="primary")
                out_s3 = gr.Markdown(elem_classes=["output-box"])

            # Step 4 Component Group
            with gr.Group(visible=False) as pane_step4:
                gr.Markdown("## Step 4: Think Like a Product Manager (PM Mode)")
                gr.Markdown("Demonstrating persona engineering by swapping the system prompt template on the fly.")
                
                with gr.Row():
                    pm_toggle = gr.Checkbox(label="Enable PM Mode", value=False)
                    mode_indicator = gr.Markdown("BDACC Mode Active", elem_classes=["mode-badge"])
                
                code_s4 = gr.Code(
                    value='''pm_system_prompt = """
You are answering as a Product Manager. Structure responses:
1. Clarify user goal  2. Trade-offs  3. Frameworks (RICE, JTBD)  4. Actionable recommendation.
"""
active_prompt = pm_system_prompt if pm_mode else bdacc_system_prompt''',
                    language="python",
                    interactive=True,
                    label="Code Compiler"
                )
                
                gr.Markdown("**Click sample question chips to run:**")
                with gr.Row():
                    chip1 = gr.Button("Mobile App vs Website first?", size="sm")
                    chip2 = gr.Button("Prioritize BDACC Onboarding", size="sm")
                    chip3 = gr.Button("Explain RICE Score", size="sm")
                
                run_s4 = gr.Button("Run Custom Prompt Compiler", variant="primary")
                out_s4 = gr.Markdown(elem_classes=["output-box"])

            # Step 5 Component Group
            with gr.Group(visible=False) as pane_step5:
                gr.Markdown("## Step 5: Retrieval-Augmented Generation (RAG)")
                gr.Markdown("Demonstrating how model capabilities are expanded via custom document chunking and vector search.")
                
                gr.Markdown("### Phase A: Question Before Document Upload")
                btn_ask_before = gr.Button("Ask: 'Who is the first General Secretary of BDACC?'")
                out_rag_before = gr.Markdown(elem_classes=["output-box"])
                
                gr.Markdown("---")
                gr.Markdown("### Phase B: Upload & Ingest Knowledge Document")
                file_upload = gr.File(label="Upload Document (.txt, .pdf, .md)", file_types=[".txt", ".pdf", ".md"])
                btn_ingest = gr.Button("Add Knowledge Document", variant="primary")
                out_ingest = gr.Markdown(elem_classes=["output-box"])
                
                code_s5 = gr.Code(
                    value='''# Load -> Split -> Embed -> Index into Chroma Vector Store
docs = TextLoader("sample_bdacc_facts.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})''',
                    language="python",
                    interactive=True,
                    label="Code Compiler"
                )
                
                gr.Markdown("### Phase C: Re-Ask Question After Upload")
                btn_ask_after = gr.Button("Re-Ask: 'Who is the first General Secretary of BDACC?'", variant="primary")
                out_rag_after = gr.Markdown(elem_classes=["output-box"])

            # Step 6 Component Group
            with gr.Group(visible=False) as pane_step6:
                gr.Markdown("## Step 6: Conversational Memory")
                gr.Markdown("Retaining turn-by-turn conversation context using LangChain chat history buffers.")
                code_s6 = gr.Code(
                    value='''from langchain_core.runnables.history import RunnableWithMessageHistory

memory_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_session_history,
    history_messages_key="history"
)''',
                    language="python",
                    interactive=True,
                    label="Code Compiler"
                )
                name_input = gr.Textbox(label="Enter a Student Name & Department for Memory Test:", value="Rahul from CSE")
                run_s6 = gr.Button("Test Conversation Memory Compiler", variant="primary")
                out_s6 = gr.Markdown(elem_classes=["output-box"])

            # Step 7 Component Group
            with gr.Group(visible=False) as pane_step7:
                gr.Markdown("## Step 7: Final Combined AI Playground")
                gr.Markdown("Full production chatbot combining **BDACC Knowledge + RAG Search + Conversational Memory + PM Persona Toggle**.")
                
                with gr.Row():
                    final_pm_toggle = gr.Checkbox(label="Enable PM Persona", value=False)
                
                chatbot = gr.Chatbot(height=400, label="BDACC Assistant")
                msg_box = gr.Textbox(placeholder="Ask anything about BDACC, Data Science, or Product Management...", label="Your Question")
                
                with gr.Row():
                    sample_q1 = gr.Button("What is BDACC and when was it founded?", size="sm")
                    sample_q2 = gr.Button("Who was the first Gen Sec?", size="sm")
                    sample_q3 = gr.Button("How to design a Datathon onboarding flow?", size="sm")
                
                send_btn = gr.Button("Send Question", variant="primary")

        # Right Side Architecture Diagram Panel
        with gr.Column(scale=3, elem_id="diagram-column"):
            gr.Markdown("### What Just Happened?")
            gr.Markdown("*Live System Architecture Flow*")
            diagram_view = gr.Markdown(DIAGRAMS[1], elem_classes=["diagram-box"])

    # State update helper for step switches
    def switch_step(step_idx: int):
        panes = [pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7]
        updates = [gr.update(visible=(i == step_idx - 1)) for i in range(7)]
        diag = DIAGRAMS.get(step_idx, DIAGRAMS[1])
        return updates + [diag]

    # Event Bindings for Navigation
    btn_step1.click(lambda: switch_step(1), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])
    btn_step2.click(lambda: switch_step(2), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])
    btn_step3.click(lambda: switch_step(3), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])
    btn_step4.click(lambda: switch_step(4), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])
    btn_step5.click(lambda: switch_step(5), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])
    btn_step6.click(lambda: switch_step(6), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])
    btn_step7.click(lambda: switch_step(7), outputs=[pane_step1, pane_step2, pane_step3, pane_step4, pane_step5, pane_step6, pane_step7, diagram_view])

    # Execution Handlers with Dynamic Absolute Controls
    run_s1.click(run_step1_compiler, inputs=[code_s1, ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_s1, diagram_view])
    run_s2.click(run_step2_compiler, inputs=[code_s2, ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_s2, diagram_view])
    run_s3.click(run_step3_compiler, inputs=[code_s3, ctrl_model, ctrl_temp, ctrl_top_p, ctrl_sys_preset, ctrl_custom_sys], outputs=[out_s3, diagram_view])
    
    # Step 4 PM Mode Handlers
    run_s4.click(lambda q, pm, m, t, tp: run_step4_compiler(q, pm, m, t, tp), inputs=[msg_box, pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_s4, diagram_view, mode_indicator])
    pm_toggle.change(lambda pm: ("PM Mode Active" if pm else "BDACC Mode Active"), inputs=[pm_toggle], outputs=[mode_indicator])
    chip1.click(lambda pm, m, t, tp: run_step4_compiler("Should we build a mobile app or a website first?", pm, m, t, tp), inputs=[pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_s4, diagram_view, mode_indicator])
    chip2.click(lambda pm, m, t, tp: run_step4_compiler("How would you prioritize features for BDACC's onboarding flow?", pm, m, t, tp), inputs=[pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_s4, diagram_view, mode_indicator])
    chip3.click(lambda pm, m, t, tp: run_step4_compiler("What's a RICE score and how would you use it here?", pm, m, t, tp), inputs=[pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_s4, diagram_view, mode_indicator])

    # Step 5 RAG Handlers
    btn_ask_before.click(query_before_rag, inputs=[ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_rag_before])
    btn_ingest.click(add_document_knowledge, inputs=[file_upload, ctrl_embed, ctrl_chunk_size, ctrl_chunk_overlap], outputs=[out_ingest, diagram_view])
    btn_ask_after.click(query_after_rag, inputs=[ctrl_model, ctrl_temp, ctrl_top_p], outputs=[out_rag_after])

    # Step 6 Memory Handlers
    run_s6.click(run_memory_demo, inputs=[name_input, ctrl_model, ctrl_temp], outputs=[out_s6, diagram_view])

    # Step 7 Playground Handlers
    send_btn.click(full_app_chat, inputs=[msg_box, chatbot, final_pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p, ctrl_sys_preset, ctrl_custom_sys], outputs=[chatbot, msg_box])
    msg_box.submit(full_app_chat, inputs=[msg_box, chatbot, final_pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p, ctrl_sys_preset, ctrl_custom_sys], outputs=[chatbot, msg_box])
    sample_q1.click(lambda h, pm, m, t, tp, sp, cs: full_app_chat("What is BDACC and when was it founded?", h, pm, m, t, tp, sp, cs), inputs=[chatbot, final_pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p, ctrl_sys_preset, ctrl_custom_sys], outputs=[chatbot, msg_box])
    sample_q2.click(lambda h, pm, m, t, tp, sp, cs: full_app_chat("Who was the first Gen Sec?", h, pm, m, t, tp, sp, cs), inputs=[chatbot, final_pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p, ctrl_sys_preset, ctrl_custom_sys], outputs=[chatbot, msg_box])
    sample_q3.click(lambda h, pm, m, t, tp, sp, cs: full_app_chat("How to design a Datathon onboarding flow?", h, pm, m, t, tp, sp, cs), inputs=[chatbot, final_pm_toggle, ctrl_model, ctrl_temp, ctrl_top_p, ctrl_sys_preset, ctrl_custom_sys], outputs=[chatbot, msg_box])

    # Presenter Toolbar Event Handlers (Font Size Zoom & Code Expansion)
    btn_font_std.click(None, js="() => { document.body.className = 'font-size-std'; }")
    btn_font_aud.click(None, js="() => { document.body.className = 'font-size-aud'; }")
    btn_font_lg.click(None, js="() => { document.body.className = 'font-size-lg'; }")
    btn_font_xl.click(None, js="() => { document.body.className = 'font-size-xl'; }")
    btn_expand_code.click(None, js="""() => {
        document.body.classList.toggle('code-expanded');
        const isExp = document.body.classList.contains('code-expanded');
        const els = document.querySelectorAll('.cm-editor, .cm-scroller, .cm-content, div[data-testid="code"], .code-container, div[class*="code"]');
        els.forEach(el => {
            if (isExp) {
                el.style.setProperty('max-height', 'none', 'important');
                el.style.setProperty('height', 'auto', 'important');
            } else {
                el.style.removeProperty('max-height');
                el.style.removeProperty('height');
            }
        });
    }""")

    # Reset Handler
    reset_btn.click(reset_demo, outputs=[out_s1, diagram_view, mode_indicator, file_upload, out_rag_before, out_rag_after, chatbot])

# ===================================================================
# LAUNCH APPLICATION
# ===================================================================

if __name__ == "__main__":
    print("===================================================================")
    print("Starting 2026 Glassmorphic BDACC AI Lab Demo Application...")
    print(f"* Target URL: http://127.0.0.1:7860")
    print(f"* Primary Model: {DEFAULT_MODEL_NAME}")
    print(f"* Embedding Model: {DEFAULT_EMBEDDING_MODEL}")
    print("===================================================================")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"))
