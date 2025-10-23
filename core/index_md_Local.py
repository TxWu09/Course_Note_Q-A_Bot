from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def load_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_multiple_markdown(folder: str = ".", pattern: str = "*.md") -> str:
    all_text = ""
    for file_path in glob.glob(os.path.join(folder, pattern)):
        print(f"Loading: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
    return all_text

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def build_faiss_index(chunks, index_path="faiss_index"):
    docs = [Document(page_content=c) for c in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"save to {index_path}/ sucessfully")

if __name__ == "__main__":
    file_path = "notes.md"
    text = load_markdown(file_path)
    # text = load_multiple_markdown(folder=".", pattern="*.md")
    chunks = split_into_chunks(text)
    print(f"{file_path} load succesfully, {len(chunks)} chunks")
    build_faiss_index(chunks)
