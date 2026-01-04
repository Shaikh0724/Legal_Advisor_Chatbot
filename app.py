import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key load karein
load_dotenv()

# --- Configuration ---
DATA_PATH = "legal_data/"
DB_PATH = "chroma_db_legal"

# --- Automatic Legal Ingestion Logic ---
def setup_legal_knowledge():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        st.info(f"📁 '{DATA_PATH}' folder created. Please upload Legal PDFs to GitHub.")
        return False

    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    
    if len(raw_docs) == 0:
        return False

    # Legal documents need precise splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    return True

# --- Professional Navy & Slate UI ---
st.set_page_config(page_title="Legal AI Advisor", page_icon="⚖️", layout="centered")

st.markdown("""
    <style>
    /* Main Background - Professional Navy */
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
        color: #ffffff; 
    }
    
    /* Title Styling - Sky Blue & White */
    .main-title { 
        color: #38bdf8; 
        font-size: 40px; 
        font-weight: 800; 
        text-align: center; 
        margin-bottom: 0px;
    }

    /* Subtitle */
    .sub-title { 
        color: #94a3b8; 
        text-align: center; 
        font-size: 16px; 
        margin-bottom: 30px;
    }

    /* Chat Messages - Dark Slate Glassmorphism */
    .stChatMessage { 
        background: rgba(30, 41, 59, 0.7) !important; 
        border: 1px solid #334155;
        border-radius: 15px; 
        padding: 15px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }

    /* Force White Text for Readability */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #ffffff !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Chat Input Bar */
    .stChatInputContainer {
        background: rgba(15, 23, 42, 0.9) !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>⚖️ Legal AI Advisor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Professional Legal Guidance & Universal Intelligence</div>", unsafe_allow_html=True)

# --- Load Database ---
@st.cache_resource
def get_legal_retriever():
    if not os.path.exists(DB_PATH):
        with st.spinner("⏳ Analyzing legal statutes and documents..."):
            success = setup_legal_knowledge()
            if not success:
                return None
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={'k': 4})

retriever = get_legal_retriever()

# --- Chain Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3) # Low temperature for legal accuracy

template = """You are a highly intelligent and professional Legal Assistant.

1. DOCUMENT KNOWLEDGE: First, search the provided Document Context to answer any legal, contractual, or rights-related questions.
2. WORLD INTELLIGENCE: If the user asks about world news, history, current events, or any general topic not in the documents, provide a factual and detailed answer from your internal knowledge.
3. PERSONALITY: Always be formal, objective, and clear. Mention you are a RAG-based AI powered by GPT-4o-mini.

Document Context: {context}
User Question: {question}

Final Response:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs: return "No specific legal context found."
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
if retriever:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
else:
    # General mode if no PDFs
    rag_chain = (
        {"context": lambda x: "No PDFs found.", "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# --- Chat Interface ---
if "legal_messages" not in st.session_state:
    st.session_state.legal_messages = []

for msg in st.session_state.legal_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about consumer rights, labor laws, or world events..."):
    st.session_state.legal_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting Legal Base..."):
            response = rag_chain.invoke(query)
            st.markdown(response)
            st.session_state.legal_messages.append({"role": "assistant", "content": response})
            