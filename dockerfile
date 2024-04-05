# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask flask-asgi scikit-learn tensorflow-hub nltk

# Download NLTK resources
RUN python -m nltk.downloader punkt stopwords

# Expose the Flask port
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "python.py"]
