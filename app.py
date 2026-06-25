import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="CS173 RAG Tutor",
    page_icon="📐",
    layout="centered"
)
 
st.title("📐 CS173 Discrete Math Tutor")
st.caption(
    "Powered by RAG — answers grounded in *Building Blocks for Theoretical "
    "Computer Science* (Fleck, 2017), the official CS173 textbook at UIUC."
)

openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

@st.cache_resource(show_spinner="Loading textbook and building vector store — this takes ~60 seconds on first load...")
def load_vectorstore():
    pdf_path = hf_hub_download(
        repo_id="rishijagannath/cs173-textbook",   # ← update this
        filename="cs173textbook.pdf",
        repo_type="dataset"
    )
 
    docs = PyPDFLoader(pdf_path).load()
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
 
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )
 
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
 
vectorstore = load_vectorstore()
 
# ── Model ─────────────────────────────────────────────────────────────────────
model = ChatOpenAI(
    model="o3-mini",
    openai_api_key=openai_api_key
)

 
# ── RAG prompt ────────────────────────────────────────────────────────────────
def build_prompt(context: str, query: str, history:str) -> str:
    return f"""You are an expert CS173 discrete mathematics tutor.
 
Answer using only the provided context.
 
Rules:
- Use precise mathematical language.
- Make sure to use the same notation as the context (e.g. $W_n$)
- Base all definitions and claims strictly on the context.
- If the context is insufficient, explicitly say so.
- Justify all claims clearly and logically.
- Follow proof style consistent with the context when applicable.
 
Problem-specific guidance:
- For state machine creation: identify required state memory, define transitions, then check determinism and check if your solution can be optimized by merging states.
- Proofs: follow standard discrete math proof structure (direct, contradiction, induction as appropriate).
- For counting and set-theoretic questions, carefully distinguish finite, countably infinite, and uncountable sets.
 
Formatting:
- Present structured objects (trees, automata, diagrams) in clean readable formats (tables, dictionaries, or ASCII diagrams).
 
Context:
{context}

History: 
{history}
 
Question:
{query}"""
 
# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
if query := st.chat_input("Ask a CS173 question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
 
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            retrieved_docs = vectorstore.similarity_search(query, k=4)
            context = "\n\n".join(d.page_content for d in retrieved_docs)
            history = "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages[-6:])
            prompt = build_prompt(context, query, history)
            response = model.invoke(prompt)
            answer = response.content
 
        st.markdown(answer)
 
        with st.expander("📄 Retrieved context (pages used)"):
            for i, doc in enumerate(retrieved_docs):
                page_label = doc.metadata.get("page_label", "?")
                page = int(page_label) - 15
                st.text(page)
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
 
