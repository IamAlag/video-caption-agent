# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install ffmpeg and clean up in one layer to keep the image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create the output directory expected by the harness
RUN mkdir -p /output

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]