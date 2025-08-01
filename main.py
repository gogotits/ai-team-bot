# --- 1. Загрузка библиотек ---
import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Основные компоненты для создания Агента
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool

# Инструменты
from langchain_tavily import TavilySearch
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- 2. Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ or "TELEGRAM_BOT_TOKEN" not in os.environ or "TAVILY_API_KEY" not in os.environ:
    raise ValueError("Не найдены необходимые ключи API в .env файле!")
print("✅ Ключи API загружены.")

# --- 3. Инициализация ИИ-компонентов ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")

# --- 4. Создание Инструментов ---
print("Инициализация инструментов...")
search_tool = TavilySearch(max_results=3)
search_tool.description = "Используй для всех вопросов о текущих событиях, фактах, погоде или любой информации из реального мира. Это твой инструмент по умолчанию."
archivist_db = Chroma(persist_directory="./chroma_db_archivist", embedding_function=embeddings)
archivist_chain = RetrievalQA.from_chain_type(llm, retriever=archivist_db.as_retriever())
archivist_tool = Tool(
    name="Archivist",
    func=archivist_chain.invoke,
    description="Используй ТОЛЬКО для ответов на вопросы, которые явно касаются содержания загруженного PDF документа. Например: 'что говорится в документе о...?'"
)
analyst_db = Chroma(persist_directory="./chroma_db_analyst", embedding_function=embeddings)
analyst_chain = RetrievalQA.from_chain_type(llm, retriever=analyst_db.as_retriever())
analyst_tool = Tool(
    name="Analyst",
    func=analyst_chain.invoke,
    description="Используй ТОЛЬКО для ответов на вопросы об общих финансовых и рыночных терминах. Например: 'что такое бычий рынок?'"
)
tools = [search_tool, archivist_tool, analyst_tool]
print(f"✅ Инструменты готовы: {[tool.name for tool in tools]}")

# --- 5. Создание Главного Агента ---
agent_prompt_template = """Ты — умный ИИ-ассистент... (весь ваш длинный промпт остается здесь без изменений)"""
# ... (весь остальной код для создания agent_executor остается здесь)
agent_prompt = PromptTemplate.from_template(agent_prompt_template)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
print("✅ Главный Агент создан с улучшенной логикой.")


# --- 6. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я ваш автономный ИИ-ассистент. Задайте мне любой вопрос или отправьте картинку.')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получен текстовый вопрос: '{user_query}'")
    await update.message.reply_text('Думаю над вашим текстом...')

    try:
        result = agent_executor.invoke({"input": user_query})
        response_text = result["output"]
        await update.message.reply_text(response_text)
        logger.info("Отправлен ответ на текстовый вопрос.")
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла внутренняя ошибка: {e}")

# НОВАЯ ФУНКЦИЯ для обработки фото
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Получено изображение.")
    await update.message.reply_text('Анализирую изображение...')
    
    try:
        # Получаем файл изображения с самым высоким разрешением
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # Получаем подпись к фото, если она есть. Если нет - задаем вопрос по умолчанию.
        user_caption = update.message.caption
        if not user_caption:
            user_caption = "Опиши это изображение подробно."
        
        # Создаем сообщение для мультимодальной модели
        message_payload = HumanMessage(
            content=[
                {"type": "text", "text": user_caption},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{photo_bytes}"},
            ]
        )
        
        # Напрямую вызываем LLM для анализа изображения
        response = llm.invoke([message_payload])
        
        await update.message.reply_text(response.content)
        logger.info("Отправлен ответ на изображение.")

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}", exc_info=True)
        await update.message.reply_text(f"Не удалось обработать изображение. Ошибка: {e}")


# --- 7. Основная функция запуска бота ---
def main() -> None:
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()

    # Добавляем обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    # ДОБАВЛЯЕМ новый обработчик для фото
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    print("🚀 Запускаю Telegram-бота с функцией зрения...")
    application.run_polling()

if __name__ == '__main__':
    main()