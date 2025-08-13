# Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install CPU-only PyTorch
RUN pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose app port
EXPOSE 8000

# Start the app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]