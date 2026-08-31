# 🎤 BDACC AI Lab — Live Presenter Script (7–8 Minutes)
**Event:** BDACC Fresher Orientation, NIT Warangal  
**Audience:** First-Year B.Tech / M.Tech Students  
**Presenter Goal:** Demystify AI software engineering by revealing code and architecture step by step.

---

## ⏱️ Timeline & Talk Track

### 0:00 - 1:00 | Introduction & Set the Stage
- **Action:** Open `http://127.0.0.1:7860` projected on the big screen.
- **Presenter Says:**
  > *"Welcome freshers to NIT Warangal and to BDACC! Today, we aren't just going to show you a flashy chatbot. We are going to build one live, line by line, right in front of your eyes. By the end of 8 minutes, you'll understand API calls, Prompts, Product Management personas, RAG, and Memory."*

---

### 1:00 - 2:00 | Step 1: Connect to Gemini API
- **Action:** Click **Step 1** in the left sidebar nav. Highlight the 10-line Python code panel. Click **"▶️ Run Step 1"**.
- **Presenter Says:**
  > *"Every AI app starts with an API call. Here on screen, you see 10 lines of Python code loading an API key and invoking Google's `gemini-2.5-flash-lite`. When I click Run... look at the diagram on the right: User -> API -> Gemini -> Response. No magic, just standard HTTP requests!"*

---

### 2:00 - 3:00 | Step 2 & 3: LangChain & System Prompt Grounding
- **Action:** Click **Step 2**, hit Run. Then click **Step 3**, hit Run.
- **Presenter Says:**
  > *"In Step 2, we wrap our LLM in LangChain using a prompt template. In Step 3, we give it BDACC's identity.*  
  > *Look at the big bold text on screen: **'We didn't train a new model. We changed the instructions!'** System prompts dictate the rules of engagement."*

---

### 3:00 - 4:15 | Step 4: Think Like a PM (Persona Toggle)
- **Action:** Click **Step 4**. Check the **"Enable PM Mode"** box (point out badge changing to `🧑💼 PM Mode Active`). Click the **"📱 Mobile App vs Website first?"** chip.
- **Presenter Says:**
  > *"At BDACC, technical engineering meets strategy consulting. In Step 4, we switch our system prompt to act as a Senior Product Manager. Notice how it structures the response with frameworks like RICE and trade-off analysis. Prompt engineering allows you to pivot an AI's persona instantly."*

---

### 4:15 - 6:00 | Step 5: RAG Document Upload (The "Aha" Moment)
- **Action:**
  1. Click **Step 5**. Click **"❓ Ask: 'Who is the first General Secretary of BDACC?'"**.
  2. Point to the answer: *"I don't have that information in my knowledge base."*
  3. Upload `sample_bdacc_facts.txt` via `gr.File` and click **"📚 Add This Knowledge"**. Point out the status messages (*Chunking... Embedding... Indexing into Chroma*).
  4. Click **"🔍 Re-Ask: 'Who is the first General Secretary of BDACC?'"**.
- **Presenter Says:**
  > *"Watch this carefully! Before uploading our document, the AI correctly admitted it didn't know who our first Gen Sec was. Now, we upload our `sample_bdacc_facts.txt`. Watch the vector store chunk, embed, and index it into Chroma DB.  
  > Re-asking the exact same question... Boom! It answers **K. Sai Vamsi** and cites the exact text chunk!  
  > **The model didn't get smarter. We gave it a document to look things up in!** That is Retrieval-Augmented Generation (RAG)."*

---

### 6:00 - 7:00 | Step 6: Conversation Memory
- **Action:** Click **Step 6**. Type *"Ananya from ECE"* into the name field. Click **"▶️ Test Conversation Memory"**.
- **Presenter Says:**
  > *"LLMs are stateless by default. Step 6 shows how we pass conversation history back and forth so the AI remembers your name and context across multiple turns."*

---

### 7:00 - 8:00 | Step 7 & Wrap-Up
- **Action:** Click **Step 7**. Ask a question in the final playground (e.g. *"What is BDACC and how to join?"*).
- **Presenter Says:**
  > *"And here is our complete, production-ready AI application combining BDACC knowledge, RAG document search, memory, and PM persona control!  
  > If you want to build systems like this during your first year at NIT Warangal, BDACC is the place to be. Thank you!"*
