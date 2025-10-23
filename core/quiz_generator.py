# core/quiz_generator.py
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

QUIZ_PROMPT = """
You are a helpful teaching assistant.
Based on the following course material, generate {num_questions} quiz questions.
Each question should test understanding of a key concept.
Provide each question followed by a short, direct answer.

Material:
{content}
"""

class QuizGenerator:
    """Quiz generation module (used by CLI and Web)"""

    def __init__(self, model="llama3", index_path="faiss_index"):
        self.model = model
        self.index_path = index_path
        self.llm = Ollama(model=model)
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)

    def generate(self, notes_text=None, num=5):
        """Generate quiz questions from index or provided text"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
        if not notes_text:
            docs = retriever.get_relevant_documents("core concepts")
            notes_text = "\n".join(d.page_content for d in docs[:8])

        prompt = QUIZ_PROMPT.format(num_questions=num, content=notes_text)
        print(f"Generating {num} quiz questions...\n")
        quiz = self.llm.invoke(prompt)
        return quiz


if __name__ == "__main__":
    qg = QuizGenerator()
    num = input("How many quiz questions do you want to generate? (default 5): ")
    num = int(num) if num.strip().isdigit() else 5
    print(qg.generate(num=num))
