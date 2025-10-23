from core.retriever import RAGRetriever
from core.quiz_generator import QuizGenerator
from app.config import OLLAMA_MODEL
import streamlit as st

def run_web():
    st.set_page_config(page_title="Course Notes Assistant", layout="wide")
    st.title("Course Notes Assistant")

    mode = st.sidebar.selectbox("Choose", ["Q&A", "Quiz"])
    retriever = RAGRetriever(model=OLLAMA_MODEL)
    quiz_gen = QuizGenerator(model=OLLAMA_MODEL)

    if mode == "Q&A":
        query = st.text_input("What is your question")
        if query:
            result = retriever.answer(query)
            st.subheader("Answer")
            st.write(result["answer"])
    else:
        num = st.slider("How many questions would you like?", 3, 15, 5)
        if st.button("Generate Quiz"):
            notes_text = retriever.get_combined_text()
            quiz = quiz_gen.generate(notes_text, num=num)
            st.text_area("Quiz", quiz, height=500)
