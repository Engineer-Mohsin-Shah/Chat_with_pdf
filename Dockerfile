# Use a slim Python image as the base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Set environment variable for Python output
ENV PYTHONUNBUFFERED=1

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
