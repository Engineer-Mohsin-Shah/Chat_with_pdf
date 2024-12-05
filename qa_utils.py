import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq


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


def create_embedding_vector_db(chunks):
    """Creates an embedding vector database using FAISS."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vector_store


def ask_question(vector_store, question, model, k=3):
    GOOGLE_API_KEY = 'AIzaSyAAOvSrnt3nb6Cqf81ipDAgKInVG9tm62w'  
    GROQCLOUD_API_KEY = "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"

    if model == "google_gemini":
        llm = GoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY, temperature=0.7)
    elif model == "groq_llama":
        llm = ChatGroq(model='llama-3.1-8b-instant', api_key=GROQCLOUD_API_KEY, temperature=0.7)
    else:
        raise ValueError("Invalid model specified.")


    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.invoke(question)
    return answer['result']
    
