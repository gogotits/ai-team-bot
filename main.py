# --- 1. Загрузка библиотек ---
import os
from dotenv import load_dotenv

# Новые, правильные импорты для LangChain v0.2+
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- 2. Настройка окружения ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY не найден в .env файле!")
print("✅ Ключ API загружен. Начинаем обработку документа...")

# --- 3. Обработка документа ---
pdf_path = "docs/document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"✅ Документ разбит на {len(texts)} частей.")

# --- 4. Создание векторной базы данных (памяти) ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="./chroma_db"
)
print("✅ Векторная база данных создана и сохранена.")

# --- 5. Создание цепочки (Chain) для ответов на вопросы ---
# ИЗМЕНЕНИЕ 1: Указываем более новую модель и убираем устаревший параметр
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever()
)
print("✅ Система готова к вашим вопросам. Введите 'exit' для выхода.")

# --- 6. Запуск цикла для общения ---
while True:
    query = input("\n> Ваш вопрос: ")
    if query.lower() == 'exit':
        print("До свидания!")
        break

    try:
        # ИЗМЕНЕНИЕ 2: Используем новый метод .invoke() вместо .call()
        result = qa_chain.invoke({"query": query})
        print("\nОтвет: ", result["result"])
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")