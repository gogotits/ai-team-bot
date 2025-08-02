# --- 1. Загрузка библиотек ---
import os
import logging
import base64
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tempfile

# Библиотеки для документов
from docx import Document as WordDocument
from openpyxl import Workbook as ExcelWorkbook
from fpdf import FPDF

# Основные компоненты LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_tavily import TavilySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 2. Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
if not all(key in os.environ for key in ["GOOGLE_API_KEY", "TELEGRAM_BOT_TOKEN", "TAVILY_API_KEY"]):
    raise ValueError("Один или несколько ключей API не найдены в .env файле!")
print("✅ Все ключи API загружены.")

# --- 3. Инициализация ИИ-компонентов ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")

# --- 4. ЕДИНАЯ БАЗА ЗНАНИЙ ---
print("Инициализация единой базы знаний...")
persistent_storage_path = "/var/data/main_chroma_db"
main_db = Chroma(persist_directory=persistent_storage_path, embedding_function=embeddings)
retriever = main_db.as_retriever(search_kwargs={'k': 5})
print(f"✅ Единая база знаний готова. Записей в базе: {main_db._collection.count()}")

# --- 5. СОЗДАНИЕ АГЕНТОВ-ЭКСПЕРТОВ (Вспомогательных) ---

# 5.1 Эксперт: Исследователь (ищет и сохраняет)
def research_and_learn(topic: str) -> str:
    logger.info(f"Эксперт 'Researcher': Начинаю исследование по теме: {topic}")
    search = TavilySearch(max_results=3)
    try:
        search_results = search.invoke(topic)
        raw_text = "\n\n".join([result.get('content', '') for result in search_results])
        summarizer_prompt = f"""Проанализируй текст по теме '{topic}'. Создай качественное саммари на русском языке. Ответ должен содержать только саммари."""
        summary = llm.invoke(summarizer_prompt).content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([summary], metadatas=[{"source": f"Research on {topic}"}])
        main_db.add_documents(texts)
        return f"Информация по теме '{topic}' была успешно исследована и сохранена в моей памяти."
    except Exception as e:
        return f"Ошибка в работе Исследователя: {e}"

# 5.2 Эксперт: Помощник (быстрый поиск в интернете)
quick_search_tool = TavilySearch(max_results=3)

# 5.3 Эксперт: Архивариус (поиск в памяти)
def retrieve_from_memory(query: str) -> str:
    logger.info(f"Эксперт 'Archivist': Поиск в памяти по запросу: {query}")
    docs = retriever.invoke(query)
    if not docs:
        return "В моей базе знаний нет информации по этому вопросу."
    return "\n".join([doc.page_content for doc in docs])

# 5.4 Эксперт: Секретарь (создание документов)
def create_document(content: str, doc_type: str) -> str:
    logger.info(f"Эксперт 'Secretary': Создаю документ типа {doc_type}")
    try:
        if doc_type.lower() == 'word':
            doc = WordDocument()
            doc.add_paragraph(content)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx", prefix="report_")
            doc.save(temp_file.name)
            return f"Документ Word успешно создан: {temp_file.name}"
        # Можно добавить Excel и PDF по аналогии
        else:
            return "Неподдерживаемый тип документа."
    except Exception as e:
        return f"Ошибка при создании документа: {e}"

# 5.5 Эксперт: Аналитик (анализ и обновление знаний)
def analyze_and_update_memory(query: str) -> str:
    logger.info("Эксперт 'Analyst': Начинаю анализ базы знаний.")
    all_docs = main_db.get(include=["metadatas"])
    if not all_docs or not all_docs.get('metadatas'):
        return "База знаний пуста. Нечего анализировать."
    topics = list(set([meta['source'].replace("Research on ", "") for meta in all_docs['metadatas'] if 'source' in meta]))
    if not topics:
        return "В базе знаний нет тем для анализа."
    
    planner_prompt = f"""Вот список тем в моей базе знаний: {", ".join(topics)}. Какая из них наиболее вероятно могла устареть? Ответь только названием одной темы."""
    topic_to_update = llm.invoke(planner_prompt).content.strip()
    
    logger.info(f"Аналитик решил обновить тему: {topic_to_update}")
    return research_and_learn(topic_to_update)

# --- 6. СОЗДАНИЕ ГЛАВНОГО АГЕНТА (Руководителя) ---
print("Инициализация Главного Агента и его команды...")

main_tools = [
    Tool(
        name="QuickInternetSearch",
        func=quick_search_tool.invoke,
        description="Используй для быстрых, фактических вопросов о мире (погода, новости, столицы и т.д.), которые не нужно сохранять."
    ),
    Tool(
        name="DeepResearcher",
        func=research_and_learn,
        description="Используй, когда пользователь прямо просит 'исследуй', 'найди и сохрани', чтобы найти и сохранить в память обширную информацию."
    ),
    Tool(
        name="MemoryArchivist",
        func=retrieve_from_memory,
        description="Используй, чтобы найти ответ на вопрос в своей долгосрочной памяти. Всегда пробуй этот инструмент первым для вопросов о ранее исследованных темах."
    ),
    Tool(
        name="KnowledgeAnalyst",
        func=analyze_and_update_memory,
        description="Используй, когда пользователь просит 'актуализируй знания' или 'обнови информацию'."
    ),
    Tool(
        name="Secretary",
        func=lambda input_str: create_document(content=input_str.split('|')[0], doc_type=input_str.split('|')[1]),
        description="Используй для создания документов. Входные данные должны быть строкой в формате 'текст для документа|тип документа' (например, 'Привет, мир|word')."
    ),
]

system_prompt = """Ты — Главный Агент-Руководитель. Твоя задача — общаться с пользователем, помнить контекст диалога и делегировать задачи своей команде экспертов (инструментов).

Твоя команда:
- `QuickInternetSearch`: Для быстрых фактов (погода, новости).
- `DeepResearcher`: Для глубокого исследования и сохранения знаний по команде "исследуй".
- `MemoryArchivist`: Для поиска в твоей базе знаний. Используй его первым для вопросов по исследованным темам.
- `KnowledgeAnalyst`: Для анализа и обновления базы знаний по команде "обнови".
- `Secretary`: Для создания документов по команде "создай документ".

Твоя задача — понять истинную цель пользователя и выбрать ОДНОГО наиболее подходящего эксперта для ее выполнения.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, main_tools, prompt)
memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(
    agent=agent,
    tools=main_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
print("✅ Главный Агент (Руководитель) и его команда готовы к работе.")

# --- 7. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['chat_history'] = []
    await update.message.reply_text('Привет! Я ваш ИИ-ассистент с командой экспертов. О чем поговорим сегодня?')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи... Обращаюсь к команде экспертов.')
    
    try:
        chat_history = context.user_data.get('chat_history', [])
        result = agent_executor.invoke({"input": user_query, "chat_history": chat_history})
        
        context.user_data['chat_history'] = result['chat_history']

        response_text = result["output"]
        if response_text.startswith("Документ") and '.docx' in response_text:
            try:
                file_path = response_text.split(":")[-1].strip()
                if os.path.exists(file_path):
                    await context.bot.send_document(chat_id=update.effective_chat.id, document=open(file_path, 'rb'))
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Ошибка при отправке документа: {e}", exc_info=True)
                await update.message.reply_text(f"Не удалось отправить документ.")
        else:
            await update.message.reply_text(response_text)
            logger.info("Отправлен текстовый ответ.")
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла внутренняя ошибка.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Получено изображение.")
    await update.message.reply_text('Анализирую изображение...')
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        user_caption = update.message.caption or "Опиши это изображение подробно."
        
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
        
        message_payload = HumanMessage(content=[{"type": "text", "text": user_caption}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},])
        response = llm.invoke([message_payload])
        await update.message.reply_text(response.content)
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}", exc_info=True)
        await update.message.reply_text(f"Не удалось обработать изображение.")

# --- 8. Основная функция запуска бота ---
def main() -> None:
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    print("🚀 Запускаю иерархического Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()