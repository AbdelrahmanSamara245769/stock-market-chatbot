# Use official Python base image
FROM python:3.9-slim

# Set environment variables
ENV POETRY_VERSION=1.8.2 \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y curl build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    apt-get clean

# Create and set working directory
WORKDIR /app

# Copy pyproject.toml and poetry.lock first for dependency installation
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (without venvs, directly to system)
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --only main

# Copy the rest of the project files
COPY . /app
COPY src/ /app

# Expose Gradio default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "run_chatbot.py"]
