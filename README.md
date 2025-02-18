RAG / ollama / python

docker build . -t ragscraper
docker run -v $(pwd):/app -p 8501:8501 -it ragscraper /bin/bash
#mounts as a volume so you can edit ragscraper.py without re-building image every time, runs shell for control

streamlit run ragscraper.py                                                             #run this from bash prompt                     
