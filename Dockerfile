# Use an official Python runtime as a parent image
# We use 'slim' to keep the image size small and fast
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first 
# (This allows Docker to cache our libraries for faster builds)
COPY requirements.txt .

# Install the specific versions of our libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our code and the trained model
# This includes main.py and cancer_model_v1.pkl
COPY . .

# Tell the container which port to open
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "main.py"]