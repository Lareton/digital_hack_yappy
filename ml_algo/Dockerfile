FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip

RUN apt-get install -y python3 python3-pip libgl1-mesa-glx



# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install --upgrade opencv-python

# Set the working directory
WORKDIR /app

COPY . .

# Set the entrypoint
CMD ["python3", "./main.py"]