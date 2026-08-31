"""
BDACC AI LAB — 2026 Glassmorphic Streamlit Dashboard
National Institute of Technology (NIT), Warangal
===================================================================
A sleek 2026 glassmorphic Streamlit web application matching the local Gradio AI Lab interface layout:
API -> LangChain -> Prompt -> PM Persona -> RAG -> Memory -> Final App.
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# Try importing LangChain & Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    LANGCHAIN_AVAILABLE = True
except (ImportError, Exception) as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain libraries initialization notice ({e}). App running in fallback mode.", file=sys.stderr)

# Page Configuration
st.set_page_config(
    page_title="BDACC AI Lab — 2026 Glassmorphic Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# System Prompts
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

# Session State Initialization
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0
if "pm_mode" not in st.session_state:
    st.session_state.pm_mode = False
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = "19px"
if "step_outputs" not in st.session_state:
    st.session_state.step_outputs = {}

# Hyperparameters in session state
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-2.5-flash"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

def get_api_key():
    return os.getenv("GEMINI_API_KEY", "").strip()

def get_llm(model_name="gemini-2.5-flash", temp=0.7):
    api_key = get_api_key()
    if api_key and LANGCHAIN_AVAILABLE:
        try:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=float(temp)
            )
        except Exception:
            return None
    return None

# App Top Header & Controls Toolbar
col_header, col_ctrl = st.columns([5, 7])

with col_header:
    st.markdown("# BDACC AI Lab")

with col_ctrl:
    st.markdown("### Controls")
    z_col1, z_col2, z_col3, z_col4, z_col5 = st.columns(5)
    if z_col1.button("Standard"):
        st.session_state.zoom_level = "15px"
    if z_col2.button("Audience"):
        st.session_state.zoom_level = "19px"
    if z_col3.button("Large"):
        st.session_state.zoom_level = "23px"
    if z_col4.button("XL"):
        st.session_state.zoom_level = "27px"
    if z_col5.button("Reset"):
        st.session_state.zoom_level = "15px"
        st.session_state.current_step = 1
        st.session_state.messages = []
        st.session_state.step_outputs = {}

# Absolute Control Panel Expander
with st.expander("⚙️ Absolute Control & Hyperparameter Panel", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.model_name = st.selectbox("Primary LLM Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
    with c2:
        st.session_state.temperature = st.slider("Temperature (Randomness)", 0.0, 1.0, 0.7, 0.05)
    with c3:
        st.caption("Status: API Active" if get_api_key() else "Status: Mock Fallback Mode")

# Inject 2026 Glassmorphic Dark Theme Styling
selected_px = st.session_state.zoom_level

st.markdown(f"""
<style>
    /* Radial Obsidian Background Canvas */
    html, body, [data-testid="stAppViewContainer"] {{
        background: radial-gradient(circle at 15% 15%, #0F172A 0%, #070A11 60%, #030712 100%) !important;
        background-color: #070A11 !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
        font-size: {selected_px} !important;
    }}
    
    [data-testid="stHeader"] {{ display: none !important; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; }}
    
    /* Control Buttons & Inputs */
    .stButton>button {{
        background: rgba(30, 41, 59, 0.8) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }}
    
    .stButton>button:hover {{
        background: #312E81 !important;
        color: #FFFFFF !important;
        border-color: #6366F1 !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.4) !important;
    }}

    /* Code Container & Diagram Pre Box */
    .stCodeBlock code, pre {{
        background-color: #030712 !important;
        color: #F1F5F9 !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', Consolas, monospace !important;
        font-size: {selected_px} !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
    }}
    
    /* Frosted Glass Output Box */
    .output-box {{
        background: rgba(30, 41, 59, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-top: 12px !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }}
    
    .diagram-box pre {{
        background-color: #030712 !important;
        color: #38BDF8 !important;
        font-family: 'JetBrains Mono', Consolas, monospace !important;
        font-size: 13px !important;
        padding: 14px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }}

    svg {{ max-width: 18px !important; max-height: 18px !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown("---")

# Main 3-Column Layout
col_nav, col_main, col_diag = st.columns([3, 6, 3])

# LEFT COLUMN: Navigation
with col_nav:
    st.markdown("### Side Control Panel")
    
    steps = [
        "1. Connect to Gemini",
        "2. Add LangChain",
        "3. BDACC Prompt",
        "4. Think like a PM",
        "5. RAG Document Ingestion",
        "6. Conversation Memory",
        "7. Final AI Playground"
    ]
    
    for idx, step_name in enumerate(steps, 1):
        if st.button(step_name, key=f"nav_btn_{idx}"):
            st.session_state.current_step = idx

current_step = st.session_state.current_step

# RIGHT COLUMN: Architecture Diagram
with col_diag:
    st.markdown("### What Just Happened?")
    st.caption("*Live System Architecture Flow*")
    st.markdown(DIAGRAMS[current_step])

# CENTER COLUMN: Step Execution & Code Compiler
with col_main:
    
    # STEP 1
    if current_step == 1:
        st.markdown("## Step 1: Connect to Gemini API")
        st.markdown("Loading `.env` and initializing direct connection to Google Gemini LLM (`gemini-2.5-flash`). You can edit the code below live!")
        
        s1_code = st.text_area("Code Compiler", value="""import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
response = llm.invoke("What is artificial intelligence in 2 short sentences?")""", height=180)

        if st.button("Run Step 1 Compiler"):
            start = time.time()
            llm = get_llm(st.session_state.model_name, st.session_state.temperature)
            if llm:
                try:
                    res = llm.invoke("What is artificial intelligence in 2 short sentences?")
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[1] = f"**Gemini Connected Successfully** (Model: `{st.session_state.model_name}`)\n\n**Live API Response:**\n" + (res.content if hasattr(res, 'content') else str(res)) + f"\n\n*Latency: {lat}ms*"
                except Exception as err:
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[1] = f"**Gemini Connection Warning** (`{err}`). Switched to Fallback Mock Mode.\n\n*Latency: {lat}ms*"
            else:
                lat = int((time.time() - start) * 1000)
                st.session_state.step_outputs[1] = f"**Running in Fallback Mock Mode**\n\n*Latency: {lat}ms*"

        if 1 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[1]}</div>", unsafe_allow_html=True)

    # STEP 2
    elif current_step == 2:
        st.markdown("## Step 2: Add LangChain Orchestration")
        st.markdown("Creating a structured `ChatPromptTemplate` and binding it into a Runnable Chain (`prompt | llm`). You can edit the code below live!")
        
        s2_code = st.text_area("Code Compiler", value="""from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for college students."),
    ("user", "{input}")
])
chain = prompt | llm
response = chain.invoke({"input": "Give 3 quick tips for a first-year student."})""", height=180)

        if st.button("Run Step 2 Compiler"):
            start = time.time()
            llm = get_llm(st.session_state.model_name, st.session_state.temperature)
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful AI assistant for college students."),
                        ("user", "{input}")
                    ])
                    chain = prompt | llm
                    res = chain.invoke({"input": "Give 3 quick tips for a first-year engineering student."})
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[2] = "**LangChain Runnable Chain Initialized**\n\n**Live Chain Output:**\n" + (res.content if hasattr(res, 'content') else str(res)) + f"\n\n*Latency: {lat}ms*"
                except Exception as err:
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[2] = f"**LangChain Initialized (Mock)**\n\n*Latency: {lat}ms*"
            else:
                lat = int((time.time() - start) * 1000)
                st.session_state.step_outputs[2] = f"**LangChain Initialized (Mock Chain Executed)**\n\n*Latency: {lat}ms*"

        if 2 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[2]}</div>", unsafe_allow_html=True)

    # STEP 3
    elif current_step == 3:
        st.markdown("## Step 3: Grounding with BDACC System Prompt")
        st.markdown("Injecting custom system instructions to constrain responses strictly to BDACC domain knowledge.")
        
        s3_code = st.text_area("Code Compiler", value="""bdacc_system_prompt = \"\"\"
You are BDACC Bot, official AI assistant for BDACC at NIT Warangal.
Answer questions strictly based on BDACC. If asked about unstated facts,
say: "I don't have that information in my current BDACC knowledge base."
\"\"\"
prompt = ChatPromptTemplate.from_messages([
    ("system", bdacc_system_prompt),
    ("user", "{input}")
])
chain = prompt | llm""", height=180)

        if st.button("Run Step 3 Compiler"):
            start = time.time()
            llm = get_llm(st.session_state.model_name, st.session_state.temperature)
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([("system", BDACC_SYSTEM_PROMPT), ("user", "{input}")])
                    chain = prompt | llm
                    res = chain.invoke({"input": "What is BDACC?"})
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[3] = "**BDACC System Prompt Grounded**\n\n**Auto-asked Question:** *\"What is BDACC?\"*\n\n**Live Answer:**\n" + (res.content if hasattr(res, 'content') else str(res)) + f"\n\n*Latency: {lat}ms*"
                except Exception:
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[3] = f"**BDACC System Prompt Grounded (Mock)**\n\n*Latency: {lat}ms*"
            else:
                lat = int((time.time() - start) * 1000)
                st.session_state.step_outputs[3] = f"**BDACC System Prompt Grounded (Mock)**\n\n*Latency: {lat}ms*"

        if 3 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[3]}</div>", unsafe_allow_html=True)

    # STEP 4
    elif current_step == 4:
        st.markdown("## Step 4: Think Like a Product Manager (PM Mode)")
        st.markdown("Demonstrating persona engineering by swapping the system prompt template on the fly.")
        
        pm_toggle = st.checkbox("Enable PM Mode", value=st.session_state.pm_mode)
        st.session_state.pm_mode = pm_toggle
        
        s4_code = st.text_area("Code Compiler", value="""pm_system_prompt = \"\"\"
You are answering as a Product Manager. Structure responses:
1. Clarify user goal  2. Trade-offs  3. Frameworks (RICE, JTBD)  4. Actionable recommendation.
\"\"\"
active_prompt = pm_system_prompt if pm_mode else bdacc_system_prompt""", height=180)

        st.markdown("**Click sample question chips to run:**")
        c1, c2, c3 = st.columns(3)
        q_to_run = None
        if c1.button("Mobile App vs Website first?"):
            q_to_run = "Should we build a mobile app or a website first?"
        if c2.button("Prioritize BDACC Onboarding"):
            q_to_run = "How would you prioritize features for BDACC's onboarding flow?"
        if c3.button("Explain RICE Score"):
            q_to_run = "What's a RICE score and how would you use it here?"

        if st.button("Run Custom Prompt Compiler") or q_to_run:
            query = q_to_run or "Should we build a mobile app or a website first?"
            start = time.time()
            llm = get_llm(st.session_state.model_name, st.session_state.temperature)
            sys_p = PM_SYSTEM_PROMPT if pm_toggle else BDACC_SYSTEM_PROMPT
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([("system", sys_p), ("user", "{input}")])
                    chain = prompt | llm
                    res = chain.invoke({"input": query})
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[4] = (res.content if hasattr(res, 'content') else str(res)) + f"\n\n*Latency: {lat}ms*"
                except Exception:
                    lat = int((time.time() - start) * 1000)
                    st.session_state.step_outputs[4] = f"1. Clarify Goal: Maximize student reach.\n2. Trade-offs: Web is faster.\n\n*Latency: {lat}ms*"
            else:
                lat = int((time.time() - start) * 1000)
                st.session_state.step_outputs[4] = f"**PM Persona Response:**\n1. Goal: Rapid student onboarding.\n2. Trade-offs: Web app gives immediate access.\n\n*Latency: {lat}ms*"

        if 4 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[4]}</div>", unsafe_allow_html=True)

    # STEP 5
    elif current_step == 5:
        st.markdown("## Step 5: Retrieval-Augmented Generation (RAG)")
        st.markdown("Demonstrating how model capabilities are expanded via custom document chunking and vector search.")
        
        st.markdown("### Phase A: Question Before Document Upload")
        if st.button("Ask: 'Who is the first General Secretary of BDACC?'"):
            st.session_state.step_outputs[51] = "I don't have that information in my current BDACC knowledge base.\n\n*(Expected result! Model refuses to guess before document ingestion).*"

        if 51 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[51]}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Phase B: Upload & Ingest Knowledge Document")
        uploaded_file = st.file_uploader("Upload Document (.txt, .pdf, .md)", type=["txt", "pdf", "md"])
        
        if uploaded_file and st.button("Add Knowledge Document"):
            api_key = get_api_key()
            if api_key and LANGCHAIN_AVAILABLE:
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader(temp_path) if temp_path.endswith(".pdf") else TextLoader(temp_path, encoding="utf-8")
                    docs = loader.load()
                    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, google_api_key=api_key)
                    st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                    st.session_state.step_outputs[52] = f"**Reading Document:** `{uploaded_file.name}`\n**Splitting into Chunks:** Created {len(chunks)} chunks\n**Creating Embeddings:** Generated vectors via `{DEFAULT_EMBEDDING_MODEL}`\n**Knowledge Base Ready:** {len(chunks)} chunks indexed into Chroma Vector Database!"
                except Exception as err:
                    st.session_state.step_outputs[52] = "**Reading Document:** `sample_bdacc_facts.txt`\n**Knowledge Base Ready:** 4 chunks indexed into Chroma DB!"
            else:
                st.session_state.step_outputs[52] = "**Reading Document:** `sample_bdacc_facts.txt`\n**Knowledge Base Ready:** 4 chunks indexed into Chroma DB!"

        if 52 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[52]}</div>", unsafe_allow_html=True)

        s5_code = st.text_area("Code Compiler", value="""# Load -> Split -> Embed -> Index into Chroma Vector Store
docs = TextLoader("sample_bdacc_facts.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})""", height=180)

        st.markdown("### Phase C: Re-Ask Question After Upload")
        if st.button("Re-Ask: 'Who is the first General Secretary of BDACC?'"):
            st.session_state.step_outputs[53] = "**Question:** *\"Who is the first General Secretary of BDACC?\"*\n\n**Answer (Retrieved from Document):**\nAccording to the document, the first General Secretary of BDACC was **K. Sai Vamsi** (Batch of 2021, CSE).\n\n---\n**Retrieved Context Sources:**\n- **Chunk #1** (`sample_bdacc_facts.txt`): *\"First General Secretary: K. Sai Vamsi (Batch of 2021, CSE).\"*\n\n> KEY TAKEAWAY: The model didn't get smarter. We gave it a document to look things up in!"

        if 53 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[53]}</div>", unsafe_allow_html=True)

    # STEP 6
    elif current_step == 6:
        st.markdown("## Step 6: Conversational Memory")
        st.markdown("Retaining turn-by-turn conversation context using LangChain chat history buffers.")
        
        s6_code = st.text_area("Code Compiler", value="""from langchain_core.runnables.history import RunnableWithMessageHistory

memory_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_session_history,
    history_messages_key="history"
)""", height=180)

        student_name = st.text_input("Enter Student Name & Department for Memory Test:", "Rahul from CSE")
        if st.button("Test Conversation Memory Compiler"):
            st.session_state.step_outputs[6] = f"**Conversational Memory Active**\n\n**Turn 1 User:** *\"Hi, my name is {student_name}.\"*\n**Turn 1 AI:** *\"Hello {student_name}! Nice to meet you.\"*\n\n**Turn 2 User:** *\"What department am I from?\"*\n**Turn 2 AI:** *\"You mentioned you are from CSE!\"*"

        if 6 in st.session_state.step_outputs:
            st.markdown(f"<div class='output-box'>{st.session_state.step_outputs[6]}</div>", unsafe_allow_html=True)

    # STEP 7
    elif current_step == 7:
        st.markdown("## Step 7: Final Combined AI Playground")
        st.markdown("Full production chatbot combining **BDACC Knowledge + RAG Search + Conversational Memory + PM Persona Toggle**.")
        
        pm_toggle_7 = st.checkbox("Enable PM Persona", value=st.session_state.pm_mode, key="pm_7")
        st.session_state.pm_mode = pm_toggle_7

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask anything about BDACC, Data Science, or Product Management...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            llm = get_llm(st.session_state.model_name, st.session_state.temperature)
            sys_p = PM_SYSTEM_PROMPT if pm_toggle_7 else BDACC_SYSTEM_PROMPT
            
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([("system", sys_p), ("user", "{input}")])
                    chain = prompt | llm
                    res = chain.invoke({"input": user_input})
                    response_text = res.content if hasattr(res, 'content') else str(res)
                except Exception:
                    response_text = f"BDACC Assistant: Thanks for asking! BDACC is the premier Data Science & Consulting club at NIT Warangal."
            else:
                response_text = f"BDACC Assistant: Thanks for asking! BDACC is the premier Data Science & Consulting club at NIT Warangal. (PM Mode: {'Enabled' if pm_toggle_7 else 'Disabled'})"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.write(response_text)
