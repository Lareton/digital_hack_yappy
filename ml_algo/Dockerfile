FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip

RUN git submodule update --init --recursive
# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory
WORKDIR /app

COPY . .

# Set the entrypoint
CMD ["python3", "./main.py"]