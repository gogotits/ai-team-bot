# --- 1. Загрузка библиотек ---
import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Основные компоненты для создания Агента
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool

# ИЗМЕНЕНИЕ 1: Правильный импорт из новой библиотеки
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
# ИЗМЕНЕНИЕ 2: Используем правильное имя класса
search_tool = TavilySearch(max_results=3)
search_tool.description = "Используй для всех вопросов о текущих событиях, фактах, погоде или любой информации из реального мира. Это твой инструмент по умолчанию."

# Инструмент 2: Архивариус (для работы с PDF)
archivist_db = Chroma(persist_directory="./chroma_db_archivist", embedding_function=embeddings)
archivist_chain = RetrievalQA.from_chain_type(llm, retriever=archivist_db.as_retriever())
archivist_tool = Tool(
    name="Archivist",
    func=archivist_chain.invoke,
    description="Используй ТОЛЬКО для ответов на вопросы, которые явно касаются содержания загруженного PDF документа. Например: 'что говорится в документе о...?'"
)

# Инструмент 3: Аналитик (для работы с финансами)
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
agent_prompt_template = """Ты — умный ИИ-ассистент. Твоя задача — ответить на вопрос пользователя, выбрав наиболее подходящий инструмент.
ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools}
ИСПОЛЬЗУЙ СЛЕДУЮЩИЙ ФОРМАТ ДЛЯ ОТВЕТА:
Question: вопрос, на который ты должен ответить
Thought: Мои размышления. Какой инструмент лучше всего подходит и почему?
Action: Название инструмента из списка [{tool_names}]
Action Input: Входные данные для инструмента (обычно это сам вопрос пользователя).
Observation: Результат выполнения инструмента (это поле заполняется автоматически).
Thought: Теперь у меня есть вся информация для ответа.
Final Answer: Финальный, полный и развернутый ответ на исходный вопрос пользователя.
Начинаем!
Question: {input}
Thought:{agent_scratchpad}"""
agent_prompt = PromptTemplate.from_template(agent_prompt_template)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
print("✅ Главный Агент создан с улучшенной логикой.")


# --- 6. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я ваш автономный ИИ-ассистент. Задайте мне любой вопрос.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получен вопрос: '{user_query}'")
    await update.message.reply_text('Думаю...')

    try:
        result = agent_executor.invoke({"input": user_query})
        response_text = result["output"]
        await update.message.reply_text(response_text)
        logger.info("Отправлен ответ.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла внутренняя ошибка: {e}")

# --- 7. Основная функция запуска бота ---
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