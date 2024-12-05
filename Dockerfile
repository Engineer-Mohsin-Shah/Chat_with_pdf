# Use a slim Python image as the base
FROM python:3.9.20-slim

# Set the working directory
WORKDIR /app

# Copy the required files to the container
COPY . /app

# Install Python dependencies from the requirements.txt with specific versions
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Set environment variable for Python output
#ENV PYTHONUNBUFFERED=1

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
