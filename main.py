# --- 1. Загрузка библиотек ---
import os
import logging
import base64
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tempfile
from langchain.agents.format_scratchpad.tools import render_text_description

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
if "GOOGLE_API_KEY" not in os.environ or "TELEGRAM_BOT_TOKEN" not in os.environ or "TAVILY_API_KEY" not in os.environ:
    raise ValueError("Не найдены необходимые ключи API в .env файле!")
print("✅ Ключи API загружены.")

# --- 3. Инициализация ИИ-компонентов ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")

# --- 4. ЕДИНАЯ БАЗА ЗНАНИЙ И ФУНКЦИИ-ИНСТРУМЕНТЫ ---
print("Инициализация единой базы знаний...")
persistent_storage_path = "/var/data/main_chroma_db"
main_db = Chroma(persist_directory=persistent_storage_path, embedding_function=embeddings)
retriever = main_db.as_retriever(search_kwargs={'k': 5})
print(f"✅ Единая база знаний готова. Записей в базе: {main_db._collection.count()}")

# --- Функции-инструменты ---
def create_word_document(content: str) -> str:
    doc = WordDocument()
    doc.add_paragraph(content)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx", prefix="report_")
    doc.save(temp_file.name)
    return f"Документ Word успешно создан: {temp_file.name}"

def retrieve_from_memory(query: str) -> str:
    logger.info(f"Поиск в памяти по запросу: {query}")
    docs = retriever.invoke(query)
    if not docs:
        return "В моей базе знаний нет информации по этому вопросу."
    return "\n".join([doc.page_content for doc in docs])

# --- 5. СОЗДАНИЕ СПЕЦИАЛИСТОВ (Вспомогательных Агентов) ---

def create_specialist_agent(persona: str, specialist_tools: list) -> AgentExecutor:
    """Фабрика для создания stateless-агентов-специалистов."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, specialist_tools, prompt)
    return AgentExecutor(agent=agent, tools=specialist_tools, verbose=True)

# Создаем Агента-Историка
historian_persona = "Ты — профессиональный историк. Твоя задача — предоставлять точные, подробные и объективные ответы на исторические вопросы, используя поиск в интернете."
historian_agent = create_specialist_agent(historian_persona, [TavilySearch(max_results=5)])

# --- 6. СОЗДАНИЕ ГЛАВНОГО АГЕНТА (Руководителя) ---
print("Инициализация Главного Агента...")
main_tools = [
    Tool(
        name="HistoryExpert",
        func=historian_agent.invoke,
        description="Используй этот инструмент для любых вопросов, связанных с историей, историческими личностями, событиями и датами."
    ),
    Tool(
        name="MemoryRetriever",
        func=retrieve_from_memory,
        description="Используй, чтобы найти ответ на вопрос в своей долгосрочной памяти. Всегда пробуй этот инструмент первым, если вопрос касается ранее исследованных тем."
    ),
    Tool(
        name="CreateWordDocument",
        func=create_word_document,
        description="Используй для создания документа Word (.docx)."
    ),
]

main_system_prompt = """Ты — Главный Агент-Руководитель. Твоя задача — общаться с пользователем, помнить контекст диалога и делегировать задачи своей команде экспертов.

ТВОЯ КОМАНДА ЭКСПЕРТОВ (ИНСТРУМЕНТЫ):
{tools}

ТВОЙ АЛГОРИТМ:
1.  Пойми запрос пользователя, учитывая историю диалога.
2.  Выбери наиболее подходящего эксперта (инструмент) для выполнения задачи.
3.  Четко сформулируй задачу для эксперта и передай ему.
4.  Получив ответ от эксперта, проанализируй его и дай финальный, развернутый и дружелюбный ответ пользователю в контексте вашего диалога."""

main_prompt = ChatPromptTemplate.from_messages([
    ("system", main_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Стало:
# Важное изменение: передаем инструменты напрямую в prompt.
main_prompt = main_prompt.partial(tools=render_text_description(main_tools))
main_agent = create_tool_calling_agent(llm, main_tools, main_prompt)
memory = ConversationBufferWindowMemory(k=8, memory_key="chat_history", return_messages=True)
main_agent_executor = AgentExecutor(
    agent=main_agent,
    tools=main_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
print("✅ Главный Агент (Руководитель) создан.")

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
        result = main_agent_executor.invoke({"input": user_query, "chat_history": chat_history})
        
        context.user_data['chat_history'] = result['chat_history']

        response_text = result["output"]
        if response_text.startswith("Документ") and '.docx' in response_text:
            try:
                file_path = response_text.split(":")[-1].strip()
                if os.path.exists(file_path):
                    await context.bot.send_document(chat_id=update.effective_chat.id, document=open(file_path, 'rb'))
                    os.remove(file_path)
                else:
                    await update.message.reply_text("Не удалось найти созданный файл для отправки.")
            except Exception as e:
                logger.error(f"Ошибка при отправке документа: {e}", exc_info=True)
                await update.message.reply_text(f"Не удалось отправить документ.")
        else:
            await update.message.reply_text(response_text)
            logger.info("Отправлен текстовый ответ.")
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла внутренняя ошибка.")

# --- 8. Основная функция запуска бота ---
def main() -> None:
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    # Мы убрали обработку фото в этой версии для упрощения
    print("🚀 Запускаю иерархического Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()