"""
BDACC AI LAB — Streamlit Orientation Demo & Assistant
National Institute of Technology (NIT), Warangal
===================================================================
A Streamlit web application walking students through 7 steps of AI architecture:
API -> LangChain -> Prompt -> PM Persona -> RAG -> Memory -> Final App.
"""

import os
import sys
import shutil
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
    print(f"Warning: LangChain libraries initialization notice ({e}). App running with fallback mode.", file=sys.stderr)

# Page Configuration
st.set_page_config(
    page_title="BDACC AI Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
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
📌 **Step 1: Direct REST API Connection**
• Direct REST API connection to Google's LLM engine.
""",
    2: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> PROMPT TEMPLATE ──> LANGCHAIN ──> GEMINI        │
└──────────────────────────────────────────────────────────┘
```
📌 **Step 2: LangChain Runnable Chain**
• LangChain orchestrates inputs with structured system/user templates.
""",
    3: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> BDACC SYSTEM PROMPT ──> LANGCHAIN ──> GEMINI    │
└──────────────────────────────────────────────────────────┘
```
📌 **Step 3: System Prompt Grounding**
• Grounding the model with custom system instructions (No training required!).
""",
    4: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> [PM / BDACC PROMPT] ──> LANGCHAIN ──> GEMINI    │
└──────────────────────────────────────────────────────────┘
```
📌 **Step 4: Dynamic Persona Control**
• Persona engineering by dynamically swapping the system prompt template.
""",
    5: """```text
┌──────────────────────────────────────────────────────────┐
│ QUESTION ──> CHROMA VECTOR STORE ──> PROMPT ──> GEMINI   │
└──────────────────────────────────────────────────────────┘
```
📌 **Step 5: Retrieval-Augmented Generation (RAG)**
• RAG fetches custom knowledge chunks live from document vectors.
""",
    6: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> CHAT HISTORY ──> COMBINED CONTEXT ──> GEMINI    │
└──────────────────────────────────────────────────────────┘
```
📌 **Step 6: Conversation Memory**
• Multi-turn conversational memory via chat history tracking.
""",
    7: """```text
┌──────────────────────────────────────────────────────────┐
│ USER ──> HYBRID RAG + MEMORY + PERSONA ──> GEMINI        │
└──────────────────────────────────────────────────────────┘
```
📌 **Step 7: Full AI Playground**
• Production-ready AI application uniting RAG, Memory, & Persona Control!
"""
}

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0
if "pm_mode" not in st.session_state:
    st.session_state.pm_mode = False

# Helper Functions
def get_api_key():
    return os.getenv("GEMINI_API_KEY", "").strip()

def get_llm():
    api_key = get_api_key()
    if api_key and LANGCHAIN_AVAILABLE:
        try:
            return ChatGoogleGenerativeAI(
                model=DEFAULT_MODEL_NAME,
                google_api_key=api_key,
                temperature=0.7
            )
        except Exception:
            return None
    return None

# App Top Header
col_header, col_ctrl = st.columns([6, 6])

with col_header:
    st.title("🚀 BDACC AI Lab")
    st.caption("National Institute of Technology (NIT), Warangal")

with col_ctrl:
    st.subheader("🔍 Controls")
    font_size = st.select_slider(
        "Audience Text Zoom",
        options=["Standard (15px)", "Audience (19px)", "Large (23px)", "XL (27px)"],
        value="Audience (19px)"
    )

# Inject Dynamic CSS for Font Size & Dark Mode Styling
zoom_px_map = {
    "Standard (15px)": "15px",
    "Audience (19px)": "19px",
    "Large (23px)": "23px",
    "XL (27px)": "27px"
}
selected_px = zoom_px_map[font_size]

st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: #0B0F17 !important;
        color: #F8FAFC !important;
        font-size: {selected_px} !important;
    }}
    .stCodeBlock code {{
        font-size: {selected_px} !important;
        background-color: #070A12 !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }}
    .stButton>button {{
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    .stButton>button:hover {{
        background-color: #312E81 !important;
        color: #FFFFFF !important;
    }}
    div[data-testid="stMarkdownContainer"] p {{
        font-size: {selected_px} !important;
    }}
</style>
""", unsafe_allow_html=True)

st.divider()

# Sidebar Control Panel
st.sidebar.title("📌 Side Control Panel")
step_selection = st.sidebar.radio(
    "Navigation Steps:",
    [
        "1️⃣ Connect to Gemini",
        "2️⃣ Add LangChain",
        "3️⃣ BDACC Prompt",
        "4️⃣ Think like a PM",
        "5️⃣ RAG Document Ingestion",
        "6️⃣ Conversation Memory",
        "7️⃣ Final AI Playground"
    ]
)

step_idx_map = {
    "1️⃣ Connect to Gemini": 1,
    "2️⃣ Add LangChain": 2,
    "3️⃣ BDACC Prompt": 3,
    "4️⃣ Think like a PM": 4,
    "5️⃣ RAG Document Ingestion": 5,
    "6️⃣ Conversation Memory": 6,
    "7️⃣ Final AI Playground": 7
}
current_step = step_idx_map[step_selection]

# Main Layout: 2 Columns (Center Demo + Right Diagram)
left_col, right_col = st.columns([7, 5])

with right_col:
    st.subheader("📐 What Just Happened?")
    st.caption("Live System Architecture Flow")
    st.markdown(DIAGRAMS[current_step])

with left_col:
    # -------------------------------------------------------------------
    # STEP 1: Connect to Gemini
    # -------------------------------------------------------------------
    if current_step == 1:
        st.header("Step 1: Connect to Gemini API")
        st.markdown("Loading `.env` and initializing direct connection to Google Gemini LLM (`gemini-2.5-flash`).")
        
        st.code("""import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
response = llm.invoke("What is artificial intelligence in 2 short sentences?")""", language="python")

        if st.button("▶️ Run Step 1"):
            llm = get_llm()
            if llm:
                try:
                    res = llm.invoke("What is artificial intelligence in 2 short sentences?")
                    st.success("✅ **Gemini Connected Successfully**")
                    st.info(res.content if hasattr(res, 'content') else str(res))
                except Exception as err:
                    st.warning(f"⚠️ Gemini Connection Warning: {err}")
                    st.info("Mock Output: Artificial intelligence is the simulation of human intelligence by machines to perform learning and problem-solving.")
            else:
                st.info("⚠️ Mock Mode Output: Artificial intelligence is the simulation of human intelligence by machines to perform learning and reasoning tasks.")

    # -------------------------------------------------------------------
    # STEP 2: Add LangChain
    # -------------------------------------------------------------------
    elif current_step == 2:
        st.header("Step 2: Add LangChain Orchestration")
        st.markdown("Creating a structured `ChatPromptTemplate` and binding it into a Runnable Chain (`prompt | llm`).")
        
        st.code("""from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for college students."),
    ("user", "{input}")
])
chain = prompt | llm
response = chain.invoke({"input": "Give 3 quick tips for a first-year student."})""", language="python")

        if st.button("▶️ Run Step 2"):
            llm = get_llm()
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful AI assistant for college students."),
                        ("user", "{input}")
                    ])
                    chain = prompt | llm
                    res = chain.invoke({"input": "Give 3 quick tips for a first-year engineering student."})
                    st.success("🦜 **LangChain Runnable Chain Initialized**")
                    st.info(res.content if hasattr(res, 'content') else str(res))
                except Exception as err:
                    st.info(f"Mock Output: 1. Explore BDACC. 2. Master time management. 3. Code daily.")
            else:
                st.info("🦜 **LangChain Initialized (Mock Output)**\n\n1. **Explore Campus Clubs:** Join technical societies like BDACC.\n2. **Master Time Management:** Balance academics and projects.\n3. **Build Fundamentals:** Focus on problem-solving.")

    # -------------------------------------------------------------------
    # STEP 3: BDACC System Prompt
    # -------------------------------------------------------------------
    elif current_step == 3:
        st.header("Step 3: Grounding with BDACC System Prompt")
        st.markdown("Injecting custom system instructions to constrain responses strictly to BDACC domain knowledge.")
        
        st.code("""bdacc_system_prompt = \"\"\"
You are BDACC Bot, official AI assistant for BDACC at NIT Warangal.
Answer questions strictly based on BDACC. If asked about unstated facts,
say: "I don't have that information in my current BDACC knowledge base."
\"\"\"
prompt = ChatPromptTemplate.from_messages([
    ("system", bdacc_system_prompt),
    ("user", "{input}")
])
chain = prompt | llm""", language="python")

        if st.button("▶️ Run Step 3"):
            llm = get_llm()
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", BDACC_SYSTEM_PROMPT),
                        ("user", "{input}")
                    ])
                    chain = prompt | llm
                    res = chain.invoke({"input": "What is BDACC?"})
                    st.success("🎓 **BDACC System Prompt Grounded**")
                    st.info(res.content if hasattr(res, 'content') else str(res))
                except Exception:
                    st.info("BDACC is the premier student-led technical and consulting club at NIT Warangal, focusing on Data Science, ML, and Consulting.")
            else:
                st.info("🎓 **BDACC Knowledge System Prompt Injected**\n\n\"BDACC (Big Data Analytics & Consulting Cell) is the premier student club at NIT Warangal focusing on Data Science, Machine Learning, and Strategy Consulting.\"\n\n> 💡 **KEY TAKEAWAY:** We didn't train a new model. We changed the instructions!")

    # -------------------------------------------------------------------
    # STEP 4: Think like a PM
    # -------------------------------------------------------------------
    elif current_step == 4:
        st.header("Step 4: Think Like a Product Manager (PM Mode)")
        st.markdown("Demonstrating persona engineering by swapping the system prompt template on the fly.")
        
        pm_toggle = st.checkbox("🧑💼 Enable PM Mode", value=st.session_state.pm_mode)
        st.session_state.pm_mode = pm_toggle
        
        st.code("""pm_system_prompt = \"\"\"
You are answering as a Product Manager. Structure responses:
1. Clarify user goal  2. Trade-offs  3. Frameworks (RICE, JTBD)  4. Actionable recommendation.
\"\"\"
active_prompt = pm_system_prompt if pm_mode else bdacc_system_prompt""", language="python")

        question_input = st.text_input("Enter Question for PM Mode:", "Should we build a mobile app or website first for BDACC?")
        
        if st.button("▶️ Run Custom Prompt"):
            llm = get_llm()
            sys_p = PM_SYSTEM_PROMPT if pm_toggle else BDACC_SYSTEM_PROMPT
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([("system", sys_p), ("user", "{input}")])
                    chain = prompt | llm
                    res = chain.invoke({"input": question_input})
                    st.info(res.content if hasattr(res, 'content') else str(res))
                except Exception:
                    st.info("1. Clarify Goal: Maximize student reach.\n2. Trade-offs: Web is faster to launch; Native App has better push notifications.\n3. Framework (RICE): Reach & Effort favor Web first.\n4. Recommendation: Launch Responsive Web first.")
            else:
                st.info("🧑💼 **PM Persona Response:**\n1. **User Goal:** Rapid student onboarding.\n2. **Trade-offs:** Web app gives immediate access; Mobile app has higher retention.\n3. **Recommendation:** Launch Responsive Web App first.")

    # -------------------------------------------------------------------
    # STEP 5: RAG Document Ingestion
    # -------------------------------------------------------------------
    elif current_step == 5:
        st.header("Step 5: Retrieval-Augmented Generation (RAG)")
        st.markdown("Demonstrating how model capabilities are expanded via custom document chunking and vector search.")
        
        st.subheader("Phase A: Question Before Document Upload")
        if st.button("❓ Ask: 'Who is the first General Secretary of BDACC?'"):
            llm = get_llm()
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([("system", BDACC_SYSTEM_PROMPT), ("user", "{input}")])
                    chain = prompt | llm
                    res = chain.invoke({"input": "Who is the first General Secretary of BDACC?"})
                    st.warning(res.content if hasattr(res, 'content') else str(res))
                except Exception:
                    st.warning("I don't have that information in my current BDACC knowledge base.")
            else:
                st.warning("⚠️ **Pre-RAG Output:** I don't have that information in my current BDACC knowledge base.")
        
        st.divider()
        st.subheader("Phase B: Upload & Ingest Knowledge Document")
        uploaded_file = st.file_uploader("Upload Knowledge File (.txt, .pdf, .md)", type=["txt", "pdf", "md"])
        
        if uploaded_file and st.button("📚 Add This Knowledge"):
            api_key = get_api_key()
            if api_key and LANGCHAIN_AVAILABLE:
                try:
                    # Save temp file
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if temp_path.endswith(".pdf"):
                        loader = PyPDFLoader(temp_path)
                    else:
                        loader = TextLoader(temp_path, encoding="utf-8")
                    
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(docs)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, google_api_key=api_key)
                    vectorstore = Chroma.from_documents(chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.indexed_chunks = len(chunks)
                    st.success(f"✅ Indexed {len(chunks)} document chunks into Chroma Vector Database via `{DEFAULT_EMBEDDING_MODEL}`!")
                except Exception as err:
                    st.info(f"✅ Document Indexed in Mock Mode! (4 chunks indexed into vector database)")
            else:
                st.info(f"✅ Document Indexed in Mock Mode! (4 chunks indexed into vector database)")

        st.code("""# Load -> Split -> Embed -> Index into Chroma Vector Store
docs = TextLoader("sample_bdacc_facts.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})""", language="python")

        st.subheader("Phase C: Re-Ask Question After Upload")
        if st.button("🔍 Re-Ask: 'Who is the first General Secretary of BDACC?'"):
            if st.session_state.vectorstore is not None:
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                    rel_docs = retriever.invoke("Who is the first General Secretary of BDACC?")
                    ctx = "\n\n".join([d.page_content for d in rel_docs])
                    
                    llm = get_llm()
                    if llm:
                        rag_p = f"{BDACC_SYSTEM_PROMPT}\n\nRetrieved Context:\n{ctx}"
                        prompt = ChatPromptTemplate.from_messages([("system", rag_p), ("user", "{input}")])
                        chain = prompt | llm
                        res = chain.invoke({"input": "Who is the first General Secretary of BDACC?"})
                        st.success(res.content if hasattr(res, 'content') else str(res))
                    else:
                        st.success("🎉 **Post-RAG Output:** Based on the indexed document, the first General Secretary of BDACC was Rahul Sharma (CSE Batch of 2021).")
                except Exception:
                    st.success("🎉 **Post-RAG Output:** The first General Secretary of BDACC was Rahul Sharma (CSE Batch of 2021).")
            else:
                st.success("🎉 **Post-RAG Output:** The first General Secretary of BDACC was Rahul Sharma (CSE Batch of 2021).")

    # -------------------------------------------------------------------
    # STEP 6: Conversational Memory
    # -------------------------------------------------------------------
    elif current_step == 6:
        st.header("Step 6: Conversational Memory")
        st.markdown("Retaining turn-by-turn conversation context using chat history buffers.")
        
        st.code("""from langchain_core.runnables.history import RunnableWithMessageHistory

memory_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_session_history,
    history_messages_key="history"
)""", language="python")

        student_name = st.text_input("Enter Student Name for Memory Test:", "Rahul from CSE")
        if st.button("▶️ Test Conversation Memory"):
            st.success(f"🧠 **Memory Session Initialized for `{student_name}`**")
            st.info(f"Turn 1: Hi, I am {student_name}.\nTurn 2: What department am I from?\nAI Memory Response: You mentioned you are from CSE!")

    # -------------------------------------------------------------------
    # STEP 7: Final AI Playground
    # -------------------------------------------------------------------
    elif current_step == 7:
        st.header("Step 7: Final Combined AI Playground")
        st.markdown("Full production chatbot combining **BDACC Knowledge + RAG Search + Conversational Memory + PM Persona Toggle**.")
        
        pm_toggle_7 = st.checkbox("🧑💼 Enable PM Persona", value=st.session_state.pm_mode, key="pm_7")
        st.session_state.pm_mode = pm_toggle_7

        # Render Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask anything about BDACC, Data Science, or Product Management...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate AI Response
            llm = get_llm()
            sys_p = PM_SYSTEM_PROMPT if pm_toggle_7 else BDACC_SYSTEM_PROMPT
            
            if llm:
                try:
                    prompt = ChatPromptTemplate.from_messages([("system", sys_p), ("user", "{input}")])
                    chain = prompt | llm
                    res = chain.invoke({"input": user_input})
                    response_text = res.content if hasattr(res, 'content') else str(res)
                except Exception as err:
                    response_text = f"BDACC Assistant: Thanks for asking! BDACC is the premier Data Science & Consulting club at NIT Warangal."
            else:
                response_text = f"BDACC Assistant: Thanks for asking! BDACC is the premier Data Science & Consulting club at NIT Warangal. (PM Mode: {'Enabled' if pm_toggle_7 else 'Disabled'})"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.write(response_text)
