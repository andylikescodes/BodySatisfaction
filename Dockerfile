# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./scripts /app
COPY ./requirements.txt /app
COPY ./data /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "scripts/prediction.py"]