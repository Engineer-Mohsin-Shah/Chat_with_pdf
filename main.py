import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from qa_utils import load_document, chunk_data, create_embedding_vector_db, ask_question

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the QA Application API!"}


# Endpoint to upload and process a document
@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(256),
    k: int = Form(3)
):
    try:
        # Generate unique file path
        import uuid
        temp_file_path = os.path.join("./", f"{uuid.uuid4()}_{file.filename}")
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        
        # Load and chunk document
        name, extension = os.path.splitext(file.filename)
        data = load_document(temp_file_path, extension)
        if not data:
            raise ValueError("Document is empty or could not be read.")
        
        chunks = chunk_data(data, chunk_size)
        if not chunks:
            raise ValueError("No chunks created from document.")
        
        # Create embedding vector DB
        vector_store = create_embedding_vector_db(chunks)
        app.state.vector_store = vector_store

        # Clean up file
        os.remove(temp_file_path)
        return {"message": "File uploaded, chunked, and embedded successfully!"}
    except Exception as e:
        # Cleanup in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Endpoint to ask a question based on the uploaded document
@app.post("/ask/")
async def ask_question_endpoint(
    question: str = Form(...), 
    model: str = Form(...),  # Add model selection as a form parameter
    k: int = Form(3)
):
    # List of valid models
    valid_models = ["google_gemini", "groq_llama"]

    # Check if the provided model is valid
    if model not in valid_models:
        return JSONResponse(
            content={
                "error": f"Invalid model selected. Please choose one of the following models: {', '.join(valid_models)}."
            },
            status_code=400
        )

    # Ensure a document is uploaded and processed first
    if not hasattr(app.state, "vector_store"):
        return JSONResponse(content={"error": "Please upload and process a document first."}, status_code=400)

    try:
        # Retrieve vector store
        vector_store = app.state.vector_store

        # Ask the question using the selected model
        answer = ask_question(vector_store, question, model, k)
        return {"result": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


