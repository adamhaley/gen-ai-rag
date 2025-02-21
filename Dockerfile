FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential pkg-config python3-dev make && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --upgrade pymupdf  #upgrade flag passed to solve dependency issue 

EXPOSE 8501

RUN mkdir -p ~/.streamlit

COPY config.toml ~/.streamlit/







