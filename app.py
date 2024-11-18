from llama_index.core import Settings
import os
import openai
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
   Settings,
   StorageContext,
   VectorStoreIndex,
   SimpleDirectoryReader,
   load_index_from_storage,
)
from typing import Dict, Optional
from literalai import LiteralClient
import chainlit as cl
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from src.prompts import CUSTOM_AGENT_SYSTEM_TEMPLATE
from langchain.memory import ConversationBufferMemory
from chainlit.types import ThreadDict
import datetime
import nest_asyncio
nest_asyncio.apply()


#load environment variables
load_dotenv()


literal_ai_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))
openai.api_key = os.getenv("OPEN_API_KEY")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

try:
   # rebuild storage context
   storage_context = StorageContext.from_defaults(persist_dir="./data/index_storage")
   # load index
   index = load_index_from_storage(storage_context)
except:
   documents = SimpleDirectoryReader(
      "./data/ingestion_storage"
   ).load_data(show_progress=True)
   index = VectorStoreIndex.from_documents(documents)
   index.storage_context.persist()

query_engine = index.as_query_engine(
   similarity_top_k=3
)

def multiply(a: float, b: float) -> float:
   """Multiply two numbers and returns the product"""
   return a * b
multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
   """Add two numbers and returns the sum"""
   return a + b
add_tool = FunctionTool.from_defaults(fn=add)

def get_date_time()->str:
   now = datetime.datetime.now()
   return now.strftime ("%Y -%m -%d %H:%M:%S")
datetime_tool = FunctionTool.from_defaults(fn=get_date_time)

tool = QueryEngineTool.from_defaults(
   query_engine,
   name="book",
   description=(
      f"Provides information that you have read from the book 'Common diseases in Vietnam'."
      # f"Use detailed plain text questions as input for the tool."
   ),
)

agent = OpenAIAgent.from_tools(
   [multiply_tool, add_tool, datetime_tool, tool],
   system_prompt=CUSTOM_AGENT_SYSTEM_TEMPLATE,
   verbose=True
)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
   # Fetch the user matching username from your database
   # and compare the hashed password with the value stored in the database
   if (username, password) == ("hoangkhoi", "01012003"):
      return cl.User(
         identifier="admin", metadata={"role": "admin", "provider": "credentials"}
      )
   else:
      return None

@cl.set_starters
async def set_starters():
   return [
      cl.Starter(
         label="Common diseases of chilli",
         message="List me some common diseases of chilli",
         # icon="./public/idea.svg",
      ),

      cl.Starter(
         label="Common fungal diseases of onions",
         message="List me some common diseases of onions",
         # icon="./public/idea.svg",
      ),
   ]

@cl.on_chat_start
async def start():
   Settings.context_window = 4096

   Settings.callback_manager = CallbackManager(
      [
         cl.LlamaIndexCallbackHandler()
      ]
   )
   query_engine = index.as_query_engine(
      streaming=True, similarity_top_k=3
   )

   cl.user_session.set("query_engine", query_engine)

   app_user = cl.user_session.get("user")
   # await cl.Message(
   #     author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
   # ).send()

   # Memory for resume chat
   cl.user_session.set(
      "memory", ConversationBufferMemory(return_messages=True)
   )


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
   memory = ConversationBufferMemory(return_messages=True)
   # root_messages = [m for m in thread["steps"] if m["parentId"] == None]

   # for message in root_messages:
   for message in thread.get("steps", []):
      if message["type"] == "user_message":
         memory.chat_memory.add_user_message(message["output"])
      else:
         memory.chat_memory.add_ai_message(message["output"])

   cl.user_session.set("memory", memory)

@cl.on_message
async def run_conversation(message: cl.Message):
   # message_history = cl.user_session.get("message_history")
   # message_history.append({"name": "user", "role": "user", "content": message.content})
   memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
   res = cl.Message(content="", author="Answer")

   answer = agent.stream_chat(message.content)
   response_gen = answer.response_gen
   print("response_gen:", response_gen)

   for token in response_gen:
      await res.stream_token(str(token))

   await res.send()

   memory.chat_memory.add_user_message(message.content)
   memory.chat_memory.add_ai_message(res.content)


if __name__ == "__main__":
   cl.run()