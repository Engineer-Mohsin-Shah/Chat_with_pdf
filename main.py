import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from chat_with_pdf_utils import load_document, chunk_data, create_embedding_vector_db, load_vector_store, ask_question

app = FastAPI()
VECTOR_STORE_PATH = os.path.join("data", "faiss_index", "active")
app.mount("/static", StaticFiles(directory="static"), name="static")
MODEL_ALIAS = {
    "gemini": "google_gemini",
    "llama": "groq_llama",
}
MODEL_CHOICES = list(MODEL_ALIAS.keys())


def _vector_store_exists():
    return os.path.isdir(VECTOR_STORE_PATH) and bool(os.listdir(VECTOR_STORE_PATH))


def _reset_vector_store_path():
    if os.path.isdir(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    Path(VECTOR_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to the QA Application API!"}


@app.get("/agent")
async def agent_ui():
    return FileResponse("static/index.html")


# Endpoint to upload and process a document
@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(256),
    k: int = Form(3)
):
    try:
        _reset_vector_store_path()

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
        vector_store = create_embedding_vector_db(chunks, persist_path=VECTOR_STORE_PATH)
        app.state.vector_store = vector_store

        # Clean up file
        os.remove(temp_file_path)
        return {
            "message": "File uploaded, chunked, embedded, and saved successfully!",
            "vector_store_path": VECTOR_STORE_PATH,
            "vector_store_files": sorted(os.listdir(VECTOR_STORE_PATH))
        }
    except Exception as e:
        # Cleanup in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Endpoint to ask a question based on the uploaded document
@app.post("/ask/")
async def ask_question_endpoint(
    question: str = Form(...), 
    model: str = Form("gemini"),  # Pre-selected / default model name
    k: int = Form(3)
):
    model = model.strip().lower()
    normalized_model = MODEL_ALIAS.get(model)
    if normalized_model is None and model in MODEL_ALIAS.values():
        normalized_model = model

    if normalized_model is None:
        return JSONResponse(
            content={
                "error": f"Invalid model selected. Use one of: {', '.join(MODEL_CHOICES)}."
            },
            status_code=400
        )

    # Ensure a document is uploaded and processed first
    if hasattr(app.state, "vector_store"):
        vector_store = app.state.vector_store
    elif os.path.isdir(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        vector_store = load_vector_store(VECTOR_STORE_PATH)
        app.state.vector_store = vector_store
    else:
        return JSONResponse(content={"error": "Please upload and process a document first."}, status_code=400)

    try:
        # Retrieve vector store
        # Ask the question using the selected model
        answer = ask_question(vector_store, question, normalized_model, k)
        return {"result": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/models")
async def available_models():
    return {"models": MODEL_CHOICES}


@app.get("/vector-store/status")
async def vector_store_status():
    exists = _vector_store_exists()
    return {
        "exists": exists,
        "vector_store_path": VECTOR_STORE_PATH,
        "files": sorted(os.listdir(VECTOR_STORE_PATH)) if exists else []
    }


