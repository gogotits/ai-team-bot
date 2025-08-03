# core/config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
if not all(key in os.environ for key in ["GOOGLE_API_KEY", "TELEGRAM_BOT_TOKEN", "TAVILY_API_KEY"]):
    raise ValueError("Один или несколько ключей API не найдены!")
print("✅ Все ключи API загружены.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")

persistent_storage_path = "/var/data/main_chroma_db"
db = Chroma(persist_directory=persistent_storage_path, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={'k': 5})
print(f"✅ Единая база знаний готова. Записей в базе: {db._collection.count()}")