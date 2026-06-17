import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQCLOUD_API_KEY = os.getenv("GROQCLOUD_API_KEY")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _build_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_document(file_path, extension):
    """Loads the document based on its extension."""
    if extension == '.pdf':
        loader = PyMuPDFLoader(file_path)
    elif extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported document format.")
    
    return loader.load()


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """Splits the document data into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)


def create_embedding_vector_db(chunks, persist_path=None):
    """Creates an embedding vector database and optionally persists it."""
    embedding_model = _build_embedding_model()
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    if persist_path:
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        vector_store.save_local(persist_path)
    return vector_store


def load_vector_store(persist_path):
    """Load a persisted FAISS vector store from disk."""
    embedding_model = _build_embedding_model()
    return FAISS.load_local(
        persist_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


def ask_question(vector_store, question, model, k=3):
    if model == "google_gemini":
        llm = GoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=GOOGLE_API_KEY, temperature=0.7)
    elif model == "groq_llama":
        llm = ChatGroq(model='llama-3.1-8b-instant', api_key=GROQCLOUD_API_KEY, temperature=0.7)
    else:
        raise ValueError("Invalid model specified.")

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a document assistant. Use only the provided context to answer.\n"
                "If the user asks to summarize, provide a concise and structured summary of the context.\n"
                "If the user asks to search inside the document, respond with exact snippets or key points from the context and prioritize coverage of the requested terms.\n"
                "If context is insufficient to answer, say you don't know.\n"
                "Do not use any information outside the provided context."
            )
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
    ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

    answer = chain.invoke({"input": question})
    return answer["answer"]
