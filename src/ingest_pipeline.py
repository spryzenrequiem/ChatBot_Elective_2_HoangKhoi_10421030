from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from src.global_settings import STORAGE_PATH, FILES_PATH, CACHE_FILE
from src.prompts import CUSTOM_SUMMARY_EXTRACT_TEMPLATE
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import load_index_from_storage, Settings
import openai
import os

import nest_asyncio
nest_asyncio.apply()


#load environment variables
load_dotenv()

openai.api_key = os.getenv("OPEN_API_KEY")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens= 1024, streaming=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def ingest_documents():
   documents = SimpleDirectoryReader(
      input_files=FILES_PATH, 
      filename_as_id = True
   ).load_data()
   for doc in documents:
      print(doc.id_)
   
   try: 
      cached_hashes = IngestionCache.from_persist_path(
         CACHE_FILE
      )
      print("Cache file found. Running using cache...")
   except:
      cached_hashes = ""
      print("No cache file found. Running without cache...")
      
   pipeline = IngestionPipeline(
      transformations=[
         TokenTextSplitter(
            chunk_size=512, 
            chunk_overlap=20
         ),
         SummaryExtractor(summaries=['self'], prompt_template=CUSTOM_SUMMARY_EXTRACT_TEMPLATE),
         OpenAIEmbedding()
      ],
      cache=cached_hashes
   )

   nodes = pipeline.run(documents=documents)
   pipeline.cache.persist(CACHE_FILE)
   
   return nodes