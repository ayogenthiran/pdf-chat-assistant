# Monkey patch for Streamlit + torch.classes
import sys, types, os
torch_classes_patch = types.SimpleNamespace()
torch_classes_patch.__path__ = []
sys.modules['torch.classes'] = torch_classes_patch

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Streamlit UI
import streamlit as st

# LangChain & Components
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import glob


# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 Conversational RAG with PDF Uploads & Chat History")
st.write("Upload PDFs and have a conversation with their content. Supports contextual memory.")

# Input: Groq API Key
api_key = st.text_input("🔑 Enter your Groq API key:", type="password")

if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Chat session setup
    session_id = st.text_input("🧾 Session ID", value="default_session")

    # Initialize session history store
    if "store" not in st.session_state:
        st.session_state.store = {}

    # File upload
    uploaded_files = st.file_uploader("📎 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = "./temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split and embed documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        # Create history-aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Given a chat history and the latest user question "
             "which might reference context in the chat history, "
             "formulate a standalone question which can be understood "
             "without the chat history. Do NOT answer the question, "
             "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Create question-answering chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an assistant for question-answering tasks. "
             "Use the following pieces of retrieved context to answer "
             "the question. If you don't know the answer, say that you "
             "don't know. Use three sentences maximum and keep the answer concise."
             "\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Chat message history manager
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat input box
        user_input = st.text_input("💬 Ask a question about the uploaded PDF(s):")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("🤖 Assistant:", response["answer"])
            st.expander(" Chat History").write(session_history.messages)

else:
    st.warning("🚨 Please enter your Groq API key to begin.")