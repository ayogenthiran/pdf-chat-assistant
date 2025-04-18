# ChatPDF Pro: Conversational RAG PDF Q&A with Memory

ChatPDF Pro is a Streamlit-based chatbot that enables users to upload and interact with multiple PDF documents using natural language queries. Powered by **LangChain**, **Groq LLM (Gemma2-9b-It)**, **Hugging Face Embeddings**, and **ChromaDB**, the app supports contextual memory and session-based conversation history.

---

## Features

- Upload one or more PDF files
- Contextual Q&A using Retrieval-Augmented Generation (RAG)
- Session-based chat history to retain conversation context
- API support for Groq LLM (Gemma 2B)
- Fast semantic search powered by HuggingFace & Chroma vector store

---

## Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/ayogenthiran/pdf-chat-assistant.git
cd chatpdf-pro
```

### 2. Install required dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file in the root directory and add:
```env
HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

---

## Running the Application
```bash
streamlit run app.py
```

---

## Example Prompt

> "Summarize the key concepts covered in these slides/documents/research article."

---

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Gemma2-9b-It)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: Chroma
- **RAG Framework**: LangChain

---

## References

- [Groq API](https://groq.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)