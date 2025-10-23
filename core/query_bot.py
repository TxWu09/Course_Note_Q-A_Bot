from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from logger import log_query

set_llm_cache(InMemoryCache())

def load_vectorstore(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def create_qa_chain():
    llm = Ollama(model="llama3")  
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

if __name__ == "__main__":
    qa = create_qa_chain()
    print( "Course Notes Q&A Bot (Local Ollama Mode)")
    print( "What's your question")

    while True:
        query = input("?")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa.run(query)
        print("Here is the answer:", answer, "\n")

        log_query(query, answer)