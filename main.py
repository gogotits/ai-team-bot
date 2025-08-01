# --- 1. Загрузка библиотек ---
import os
import logging
import base64
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tempfile

# Библиотеки для создания документов
from docx import Document as WordDocument
from openpyxl import Workbook as ExcelWorkbook
from fpdf import FPDF

# Основные компоненты для создания Агента
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferWindowMemory

# Инструменты
from langchain_tavily import TavilySearch
from langchain.chains import RetrievalQA
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

# --- 4. Инициализация Баз Знаний и Функций-Инструментов ---
print("Инициализация баз знаний и инструментов...")
archivist_db = Chroma(persist_directory="./chroma_db_archivist", embedding_function=embeddings)
analyst_db = Chroma(persist_directory="./chroma_db_analyst", embedding_function=embeddings)

def create_word_document(content: str) -> str:
    # ... (код функции без изменений)
    doc = WordDocument()
    if content.startswith("# "):
        title, _, text = content.partition('\n')
        doc.add_heading(title[2:], level=1)
        doc.add_paragraph(text)
    else:
        doc.add_paragraph(content)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx", prefix="report_")
    doc.save(temp_file.name)
    return temp_file.name

def create_excel_document(content: str) -> str:
    # ... (код функции без изменений)
    wb = ExcelWorkbook()
    ws = wb.active
    for line in content.split('\n'):
        ws.append(line.split(','))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", prefix="table_")
    wb.save(temp_file.name)
    return temp_file.name

def create_pdf_document(content: str) -> str:
    # ... (код функции без изменений)
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, content)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="document_")
    pdf.output(temp_file.name)
    return temp_file.name

# НОВАЯ ФУНКЦИЯ ДЛЯ САМООБУЧЕНИЯ
# Стало:
def research_and_learn(topic: str) -> str:
    """Ищет информацию в интернете по теме, анализирует ее и добавляет в базу знаний Архивариуса."""
    logger.info(f"Начинаю исследование по теме: {topic}")
    search = TavilySearch()
    # Устанавливаем include_raw_content=True, чтобы получить полные тексты
    search_results = search.invoke({"query": topic, "include_raw_content": True})
    
    if not search_results:
        return "Не удалось найти информацию по данной теме."

    # Собираем весь найденный контент в один большой текст
    full_text = f"Отчет по теме: {topic}\n\n"
    # Теперь search_results - это список словарей, как и ожидалось
    for result in search_results:
        full_text += result.get("content", "") + "\n\n"
        
    # Разбиваем текст на чанки и добавляем в базу Архивариуса
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents([full_text])
    archivist_db.add_documents(texts)
    
    logger.info(f"Информация по теме '{topic}' успешно найдена и добавлена в базу знаний Архивариуса.")
    return f"Информация по теме '{topic}' была успешно исследована и сохранена в моей памяти."
    
# --- 5. Определение Инструментов ---
archivist_chain = RetrievalQA.from_chain_type(llm, retriever=archivist_db.as_retriever())
analyst_chain = RetrievalQA.from_chain_type(llm, retriever=analyst_db.as_retriever())
tools = [
    Tool(
        name="Researcher",
        func=research_and_learn,
        description="Используй, чтобы исследовать новую тему, найти по ней информацию и сохранить ее в долгосрочную память. Принимает на вход тему для исследования."
    ),
    Tool(
        name="Archivist",
        func=archivist_chain.invoke,
        description="Используй для ответов на вопросы по информации, которая УЖЕ ЕСТЬ в памяти (включая ту, что была найдена с помощью Researcher)."
    ),
    Tool(name="CreateWordDocument", func=create_word_document, description="Используй для создания документа Word (.docx)."),
    # ... (остальные инструменты без изменений)
    Tool(name="Analyst", func=analyst_chain.invoke, description="Используй ТОЛЬКО для ответов на вопросы об общих финансовых и рыночных терминах."),
    Tool(name="CreateExcelDocument", func=create_excel_document, description="Используй для создания документа Excel (.xlsx)."),
    Tool(name="CreatePdfDocument", func=create_pdf_document, description="Используй для создания PDF документа (.pdf).")
]
print(f"✅ Инструменты готовы: {[tool.name for tool in tools]}")


# --- 6. Создание Главного Агента с Памятью ---
memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", input_key="input")
agent_prompt_template = """Ты — автономный ИИ-ассистент. Твоя главная задача — выполнять цели, поставленные пользователем.
Ты можешь использовать инструменты для поиска информации, ее сохранения, анализа и создания документов.

ИСТОРИЯ ДИАЛОГА:
{chat_history}

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools}

ВАЖНОЕ ПРАВИЛО: Если ты используешь инструмент для создания документа, твой 'Final Answer' должен быть ТОЛЬКО путем к файлу.

ИСПОЛЬЗУЙ СЛЕДУЮЩИЙ ФОРМАТ ДЛЯ ОТВЕТА:
Question: текущая цель или вопрос пользователя.
Thought: Мои размышления. Какой мой следующий шаг? Какой инструмент использовать?
Action: Название инструмента из списка [{tool_names}]
Action Input: Входные данные для инструмента.
Observation: Результат выполнения инструмента.
... (этот цикл может повторяться)
Thought: Я достиг цели.
Final Answer: Финальный ответ пользователю, подтверждающий выполнение задачи.

Начинаем!

Question: {input}
Thought:{agent_scratchpad}"""
agent_prompt = ChatPromptTemplate.from_template(agent_prompt_template)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)
print("✅ Автономный Агент c памятью и самообучением создан.")

# --- 7. Функции-обработчики для Telegram (остаются без изменений) ---
# ... (весь код для start, handle_text_message, handle_photo_message) ...
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я ваш автономный ИИ-ассистент. Поставьте мне задачу для исследования.')
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи...')
    try:
        result = agent_executor.invoke({"input": user_query})
        response_text = result["output"]
        if os.path.exists(response_text) and response_text.endswith(('.docx', '.xlsx', '.pdf')):
            try:
                await context.bot.send_document(chat_id=update.effective_chat.id, document=open(response_text, 'rb'))
                os.remove(response_text)
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
    print("🚀 Запускаю автономного Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()