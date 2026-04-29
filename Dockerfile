FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code and artifacts
COPY . /code

# Set PYTHONPATH to include the src directory for custom modules
ENV PYTHONPATH=/code/src

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
