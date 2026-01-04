import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Configuration
DATA_PATH = "legal_data/"
DB_PATH = "chroma_db_legal"

def create_legal_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created '{DATA_PATH}'. Add Legal PDFs here.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    
    # Legal documents need precise splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    print(f"✅ Legal Knowledge Base Ready with {len(documents)} chunks!")

if __name__ == "__main__":
    create_legal_db()
    