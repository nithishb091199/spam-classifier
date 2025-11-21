# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose port
EXPOSE 5000

# Install waitress (WSGI server)
RUN pip install --no-cache-dir waitress

# Start the Flask app
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "app.app:app"]
