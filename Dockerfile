# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the Flask app
CMD ["python", "app/main.py"]
