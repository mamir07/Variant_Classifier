# Use an official Python 3.9 image as a starting point
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the files from your project folder into the container's /app directory
COPY . /app

# Install the packages using pip
RUN pip install pysam pandas scikit-learn matplotlib shap

# Set the default command to run when the container starts
CMD ["python", "1_data_preprocessing.py"]

#To start virtual enviornment, in terminal: docker build -t variant-classifier .
#Then:  docker run -it -v "${pwd}:/app" variant-classifier bash