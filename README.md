# Course-Note-Q-A-Bot (Local RAG with Ollama)
A fully local **Retrieval-Augmented Generation (RAG)** system that allows you to chat with your own course notes  
and automatically generate quizzes for self-assessment — all **without uploading any private data** to the cloud.

---

##  Features

- **Q&A Mode:** Ask questions about your lecture notes and receive context-aware, citation-backed answers.  
- **Quiz Mode:** Automatically generate short quiz questions from your notes for review or exams.  
- **Privacy First:** Runs **completely offline** using a locally deployed [Ollama](https://ollama.ai) model (`llama3`).  
- **Multi-source Retrieval:** Works with pre-built FAISS indexes from Markdown, PDF, or merged content.  
- **Command-Line & Web Interface:** Choose between CLI interaction or a Streamlit web app.  
- **Fast Local Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for lightweight semantic search.

---

##  Installation

### Install dependencies
```
pip install -r requirements.txt
```
Install and run Ollama
Download and install Ollama,
then pull the Llama 3 model locally:

```
ollama pull llama3
```

### Building the Knowledge Index
Before chatting or generating quizzes, you must build your FAISS index from your notes.

For Markdown files:
```
python build_index_local.py
```
For PDFs:
```
python build_index_pdf.py
```

### Running the Assistant
Command-line mode
```
python main.py --mode cli
```
Web interface (Streamlit)
```
python main.py --mode web
Then open:
```
http://localhost:8501
```

## Privacy and Offline Usage
This project is designed for educational environments where data confidentiality matters.
All computation happens locally on your machine:
-Embeddings: computed locally with sentence-transformers.
-Language model: served by your local Ollama llama3 instance.
-No external API calls: no data is sent to OpenAI or the internet.

Ideal for schools, instructors, or students who want AI-powered assistance
without sharing sensitive lecture content or exam materials online.
