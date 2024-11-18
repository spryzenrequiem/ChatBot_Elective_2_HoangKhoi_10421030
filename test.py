from src.prompts import CUSTOM_AGENT_SYSTEM_TEMPLATE
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
import os
import openai
import datetime
from dotenv import load_dotenv
load_dotenv()

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
      f"Cung cấp thông tin từ văn bản lưu trữ của cuốn sách 'Common diseases in Vietnam'."
      f"Sử dụng câu hỏi văn bản thuần túy chi tiết làm đầu vào cho công cụ."
   ),
)

agent = OpenAIAgent.from_tools(
   [multiply_tool, add_tool, datetime_tool, tool],
   system_prompt=CUSTOM_AGENT_SYSTEM_TEMPLATE,
   verbose=True
)


while True:
   text_input = input("User: ")
   if text_input == "exit":
      break
   response = agent.chat(text_input)
   print(f"Agent: {response}")