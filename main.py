# --- 1. Загрузка библиотек ---
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# --- 2. Настройка окружения ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY не найден в .env файле!")
print("✅ Ключ API загружен.")

# --- 3. Инициализация LLM и эмбеддингов (общие для всех) ---
# Создаем "мозг" и "переводчик в векторы" один раз, чтобы использовать для всех агентов
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")


# --- 4. Функция для создания Агента «Архивариуса» ---
def create_archivist_agent():
    """
    Создает агента, который отвечает на вопросы по PDF-документу.
    """
    print("Инициализация Агента «Архивариус»...")
    pdf_path = "docs/document.pdf"
    vector_store_archivist = Chroma(
        persist_directory="./chroma_db_archivist", 
        embedding_function=embeddings
    )
    
    # Загружаем PDF и создаем базу, только если ее еще нет
    if not vector_store_archivist.get()['documents']:
        print("База «Архивариуса» пуста. Загружаем документ...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vector_store_archivist = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory="./chroma_db_archivist"
        )
        print("✅ Документ загружен в базу «Архивариуса».")
    else:
        print("✅ База «Архивариуса» уже существует.")
        
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store_archivist.as_retriever()
    )
    return qa_chain

# --- 5. Функция для создания Агента «Аналитика» ---
# Стало:
def create_analyst_agent():
    """
    Создает простого агента-аналитика с заранее определенными знаниями.
    """
    print("Инициализация Агента «Аналитик»...")
    analyst_texts = [
        "Бычий рынок - это состояние рынка, когда цены на акции растут или ожидается их рост.",
        "Медвежий рынок - это состояние, когда цены на акции падают, и ожидается продолжение этого тренда.",
        "Диверсификация - это стратегия инвестирования, направленная на снижение рисков путем вложения средств в различные активы."
    ]
    # Используем from_texts, который сам управляет эмбеддингами
    vector_store_analyst = Chroma.from_texts(
        texts=analyst_texts,
        embedding=embeddings,  # Передаем эмбеддинги через аргумент embedding
        persist_directory="./chroma_db_analyst"
    )
    print("✅ База «Аналитика» создана.")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store_analyst.as_retriever()
    )
    return qa_chain

# --- 6. Создаем агентов ---
archivist_agent = create_archivist_agent()
analyst_agent = create_analyst_agent()

# --- 7. Запуск цикла для общения (простой Диспетчер) ---
print("\n✅ Система готова. Вы можете задавать вопросы Архивариусу или Аналитику.")
print("   Для Аналитика используйте слова 'рынок', 'акции', 'инвестиции'.")
print("   Для выхода введите 'exit'.")

while True:
    query = input("\n> Ваш вопрос: ")
    if query.lower() == 'exit':
        print("До свидания!")
        break

    try:
        # Простой Диспетчер: решает, к какому агенту обратиться
        if any(keyword in query.lower() for keyword in ['рынок', 'акции', 'инвестиции']):
            print("Перенаправляю к «Аналитику»...")
            result = analyst_agent.invoke({"query": query})
        else:
            print("Перенаправляю к «Архивариусу»...")
            result = archivist_agent.invoke({"query": query})
            
        print("\nОтвет: ", result["result"])
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")