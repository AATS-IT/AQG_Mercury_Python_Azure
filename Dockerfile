
# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code and models to the container
COPY inference_api.py .
COPY models/ ./models/
# COPY temp_files/ ./temp_files/
# Create temp_files directory if it doesn't exist
RUN mkdir -p ./temp_files

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
