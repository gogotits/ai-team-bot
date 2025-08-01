# --- 1. Загрузка библиотек ---
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 2. Настройка окружения ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY не найден в .env файле!")
print("✅ Ключ API загружен.")

# --- 3. Инициализация LLM и эмбеддингов (общие для всех) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")


# --- 4. Функции для создания Агентов (без изменений) ---
def create_archivist_agent():
    print("Инициализация Агента «Архивариус»...")
    vector_store_archivist = Chroma(
        persist_directory="./chroma_db_archivist", 
        embedding_function=embeddings
    )
    
    # Загружаем PDF и создаем базу, только если ее еще нет
    # Для этого примера мы предполагаем, что база уже создана в Фазе 2
    if not vector_store_archivist._collection.count():
         print("База «Архивариуса» пуста. Загружаем документ...")
         pdf_path = "docs/document.pdf"
         loader = PyPDFLoader(pdf_path)
         documents = loader.load()
         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
         texts = text_splitter.split_documents(documents)
         vector_store_archivist.add_documents(texts)
         print("✅ Документ загружен в базу «Архивариуса».")
    else:
        print("✅ База «Архивариуса» уже существует.")
        
    return RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store_archivist.as_retriever()
    )

def create_analyst_agent():
    print("Инициализация Агента «Аналитик»...")
    # Для этого примера мы предполагаем, что база уже создана в Фазе 2
    vector_store_analyst = Chroma(
        persist_directory="./chroma_db_analyst", 
        embedding_function=embeddings
    )
    if not vector_store_analyst._collection.count():
        print("База «Аналитика» пуста. Создаем знания...")
        analyst_texts = [
            "Бычий рынок - это состояние рынка, когда цены на акции растут или ожидается их рост.",
            "Медвежий рынок - это состояние, когда цены на акции падают, и ожидается продолжение этого тренда.",
            "Диверсификация - это стратегия инвестирования, направленная на снижение рисков путем вложения средств в различные активы."
        ]
        vector_store_analyst.add_texts(texts=analyst_texts)
        print("✅ База «Аналитика» создана.")
    else:
         print("✅ База «Аналитика» уже существует.")

    return RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store_analyst.as_retriever()
    )

# --- 5. Создание агентов ---
archivist_agent = create_archivist_agent()
analyst_agent = create_analyst_agent()

# --- 6. Создание Диспетчера (маршрутизатора) ---
# Улучшенный шаблон для более точного выбора агента
router_template = """Твоя задача - направить вопрос пользователя к одному из двух специалистов. Ответь ТОЛЬКО одним словом: 'Архивариус' или 'Аналитик'. Не добавляй ничего лишнего.

Вот описание специалистов:
- **Аналитик**: Специалист по общим вопросам о финансах, экономике и инвестициях. Выбирай его, если вопрос касается таких тем, как 'акции', 'рынок', 'инвестиции', 'диверсификация', 'бычий рынок', 'медвежий рынок'.
- **Архивариус**: Специалист по содержанию КОНКРЕТНОГО загруженного документа. Выбирай его, если вопрос явно ссылается на документ ("что в документе...", "расскажи из файла про...") или если вопрос не имеет отношения к финансам.

Примеры:
- Вопрос: "что такое диверсификация?" -> Ответ: Аналитик
- Вопрос: "какие ключевые показатели описаны в документе?" -> Ответ: Архивариус
- Вопрос: "как приготовить пиццу?" -> Ответ: Архивариус (потому что это не про финансы)

Вопрос пользователя: '{user_question}'

Выбранный специалист:"""

# Создаем промпт на основе шаблона
prompt = PromptTemplate(
    template=router_template,
    input_variables=["user_question"],
)

# Создаем цепочку для диспетчера.
# Мы используем отдельную LLM с температурой 0 для максимальной точности выбора.
router_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
router_chain = prompt | router_llm

# --- 7. Запуск цикла общения ---
print("\n✅ Система с умным диспетчером готова. Задавайте любой вопрос.")
print("   Для выхода введите 'exit'.")

while True:
    query = input("\n> Ваш вопрос: ")
    if query.lower() == 'exit':
        print("До свидания!")
        break

    try:
        # 1. Отправляем вопрос диспетчеру, чтобы он решил, кто будет отвечать
        # .content убирает лишние технические детали из ответа LLM
        chosen_agent_name = router_chain.invoke({"user_question": query}).content.strip()
        
        print(f"Диспетчер решил: '{chosen_agent_name}'")

        # 2. Вызываем нужного агента в зависимости от решения диспетчера
        if "Архивариус" in chosen_agent_name:
            result = archivist_agent.invoke({"query": query})
        elif "Аналитик" in chosen_agent_name:
            result = analyst_agent.invoke({"query": query})
        else:
            # Если LLM ответил что-то неожиданное, используем Архивариуса по умолчанию
            print("Не удалось определить агента, обращаюсь к Архивариусу...")
            result = archivist_agent.invoke({"query": query})

        print("\nОтвет: ", result["result"])
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")