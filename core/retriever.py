# core/retriever.py
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from core.logger import log_query

set_llm_cache(InMemoryCache())

class RAGRetriever:
    def __init__(self, model="llama3", index_path="faiss_index", top_k=4):
        self.model = model
        self.index_path = index_path
        self.top_k = top_k
        self.qa_chain = self._create_qa_chain()

    def _load_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)

    def _create_qa_chain(self):
        llm = Ollama(model=self.model)
        vectorstore = self._load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

    def answer(self, query: str):
        result = self.qa_chain({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])
        log_query(query, answer, sources)
        return {"answer": answer, "sources": sources}

    def get_combined_text(self):
        db = self._load_vectorstore()
        docs = db.similarity_search(" ", k=50)
        return "\n".join(d.page_content for d in docs)
