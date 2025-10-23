# build_index_pdf.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import glob, os

def load_pdfs(folder: str = ".", pattern: str = "*.pdf") -> str:
    text = ""
    for file_path in glob.glob(os.path.join(folder, pattern)):
        print(f"Loading PDF: {file_path}")
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def build_faiss_index(chunks, index_path="faiss_index_pdf"):
    docs = [Document(page_content=c) for c in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"save to {index_path}/ sucessfully")

if __name__ == "__main__":
    text = load_pdfs(folder=".", pattern="*.pdf")
    chunks = split_into_chunks(text)
    print(f"load succesfully, {len(chunks)} chunks")
    build_faiss_index(chunks)
