from core.logger import log_query
from core.quiz_generator import QuizGenerator
from core.retriever import RAGRetriever
from app.config import OLLAMA_MODEL

def run_cli():
    print("Course Notes Assistant")
    print("Enter 'quiz' to nevigate quiz mode\n")

    retriever = RAGRetriever(model=OLLAMA_MODEL)
    quiz_gen = QuizGenerator(model=OLLAMA_MODEL)

    while True:
        query = input("? Your question ")
        if query.lower() in ["exit", "quit"]:
            print("Thank you!")
            break
        elif query.lower().startswith("quiz"):
            num = int(query.split()[-1]) if query.split()[-1].isdigit() else 5
            quiz = quiz_gen.generate(retriever.get_combined_text(), num=num)
            print("\n Quiz start: \n", quiz, "\n")
        else:
            result = retriever.answer(query)
            print("Answer:", result["answer"], "\n")
            log_query(query, result["answer"])
