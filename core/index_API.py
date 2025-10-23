from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

os.environ["OPENAI_API_KEY"] = "your_api_key_here"  

def load_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def build_faiss_index(chunks, index_path="faiss_index"):
    docs = [Document(page_content=c) for c in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"save to {index_path}/ sucessfully")

if __name__ == "__main__":
    file_path = "notes.md"
    text = load_markdown(file_path)
    chunks = split_text(text)
    build_faiss_index(chunks)
