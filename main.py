# --- 1. Загрузка библиотек ---
import os
from dotenv import load_dotenv
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Импортируем нашу ИИ-логику
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 2. Настройка логирования ---
# Чтобы видеть информацию о работе бота в терминале
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- 3. Настройка окружения ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ or "TELEGRAM_BOT_TOKEN" not in os.environ:
    raise ValueError("Не найдены необходимые ключи API в .env файле!")
print("✅ Ключи API загружены.")

# --- 4. Инициализация ИИ-компонентов ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
router_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")

# --- 5. Функции для создания Агентов ---
def create_archivist_agent():
    print("Инициализация Агента «Архивариус»...")
    vector_store_archivist = Chroma(persist_directory="./chroma_db_archivist", embedding_function=embeddings)
    if not vector_store_archivist._collection.count():
        print("База «Архивариуса» пуста. Загружаем документ...")
        loader = PyPDFLoader("docs/document.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vector_store_archivist.add_documents(texts)
        print("✅ Документ загружен в базу «Архивариуса».")
    else:
        print("✅ База «Архивариуса» уже существует.")
    return RetrievalQA.from_chain_type(llm, retriever=vector_store_archivist.as_retriever())

def create_analyst_agent():
    print("Инициализация Агента «Аналитик»...")
    vector_store_analyst = Chroma(persist_directory="./chroma_db_analyst", embedding_function=embeddings)
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
    return RetrievalQA.from_chain_type(llm, retriever=vector_store_analyst.as_retriever())

# --- 6. Логика Диспетчера ---
router_template = """Твоя задача - направить вопрос пользователя к одному из двух специалистов. Ответь ТОЛЬКО одним словом: 'Архивариус' или 'Аналитик'. Не добавляй ничего лишнего.
Специалисты:
- Аналитик: Специалист по общим вопросам о финансах, экономике и инвестициях. Выбирай его, если вопрос касается таких тем, как 'акции', 'рынок', 'инвестиции', 'диверсификация', 'бычий рынок', 'медвежий рынок'.
- Архивариус: Специалист по содержанию КОНКРЕТНОГО загруженного документа. Выбирай его, если вопрос явно ссылается на документ ("что в документе...", "расскажи из файла про...") или если вопрос не имеет отношения к финансам.
Вопрос пользователя: '{user_question}'
Выбранный специалист:"""
prompt = PromptTemplate(template=router_template, input_variables=["user_question"])
router_chain = prompt | router_llm

# --- 7. Создаем агентов при старте ---
print("--- Создание ИИ-агентов ---")
archivist_agent = create_archivist_agent()
analyst_agent = create_analyst_agent()
print("--- Все агенты готовы ---")

# --- 8. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    await update.message.reply_text('Привет! Я ваш ИИ-ассистент. Задайте мне любой вопрос.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения от пользователя."""
    user_query = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Получен вопрос от chat_id {chat_id}: '{user_query}'")

    await update.message.reply_text('Думаю...')

    try:
        # 1. Вызываем Диспетчера
        chosen_agent_name = router_chain.invoke({"user_question": user_query}).content.strip()
        logger.info(f"Диспетчер выбрал: '{chosen_agent_name}'")

        # 2. Вызываем нужного агента
        if "Архивариус" in chosen_agent_name:
            result = archivist_agent.invoke({"query": user_query})
        elif "Аналитик" in chosen_agent_name:
            result = analyst_agent.invoke({"query": user_query})
        else:
            logger.warning("Не удалось определить агента, обращаюсь к Архивариусу по умолчанию.")
            result = archivist_agent.invoke({"query": user_query})
        
        # 3. Отправляем ответ пользователю
        response_text = result["result"]
        await update.message.reply_text(response_text)
        logger.info(f"Отправлен ответ для chat_id {chat_id}.")

    except Exception as e:
        logger.error(f"Произошла ошибка при обработке запроса: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла внутренняя ошибка. Попробуйте еще раз позже.\nОшибка: {e}")

# --- 9. Основная функция запуска бота ---
def main() -> None:
    """Запускает Telegram-бота."""
    # Создаем приложение и передаем ему токен
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()

    # Добавляем обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота (он будет работать, пока вы не остановите его вручную)
    print("🚀 Запускаю Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()