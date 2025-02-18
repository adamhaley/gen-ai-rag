FROM python:3.12-slim

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

RUN mkdir -p ~/.streamlit

COPY config.toml ~/.streamlit/







