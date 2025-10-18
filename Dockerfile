# Base image: lightweight Python runtime
FROM python:3.11-slim

# Set work directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY src ./src
COPY artifacts ./artifacts

# Environment setup
ENV PYTHONPATH=/app/src \
    MODEL_DIR=/app/artifacts \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose port 8080
EXPOSE 8080

# Default command: run the FastAPI app using Uvicorn
CMD ["uvicorn", "diabetes_service.api:app", "--host", "0.0.0.0", "--port", "8080"]
