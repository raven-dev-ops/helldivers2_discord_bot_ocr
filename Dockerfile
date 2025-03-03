# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-eng

# Run bot.py when the container launches
CMD ["python", "bot.py"]
