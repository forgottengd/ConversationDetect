FROM python:3.11-slim

# Set environment variables
ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file to the image
COPY ./ /app

RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose application port
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
