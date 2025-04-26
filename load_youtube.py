import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url = "https://www.youtube.com/watch?v=BJjsfNO5JTo"

save_dir="data/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),  # fetch from youtube
    #FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
    OpenAIWhisperParser()
)
docs = loader.load()

print(len(docs))

#print(docs[0].page_content[0:500])

