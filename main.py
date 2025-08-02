# --- 1. Загрузка библиотек ---
import os
import logging
import base64
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tempfile

from docx import Document as WordDocument
from openpyxl import Workbook as ExcelWorkbook
from fpdf import FPDF

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_tavily import TavilySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# НОВЫЕ ИМПОРТЫ для создания умной цепочки
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
    wb = ExcelWorkbook()
    ws = wb.active
    for line in content.split('\n'):
        ws.append(line.split(','))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", prefix="table_")
    wb.save(temp_file.name)
    return temp_file.name
def create_pdf_document(content: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, content)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="document_")
    pdf.output(temp_file.name)
    return temp_file.name

def research_and_learn(topic: str) -> str:
    logger.info(f"Начинаю исследование по теме: {topic}")
    search = TavilySearch(max_results=5)
    search_results = search.invoke(topic)
    if not search_results:
        return "Не удалось найти информацию по данной теме."
    raw_text = ""
    for result in search_results:
        raw_text += result + "\n\n"
    summarizer_prompt = f"""Проанализируй следующий текст, найденный в интернете по теме '{topic}'.
    Твоя задача — создать структурированное и подробное саммари. Выдели ключевые преимущества, недостатки, важные даты и факты.
    Ответь только текстом саммари, без лишних вступлений.
    ТЕКСТ ДЛЯ АНАЛИЗА:\n{raw_text}"""
    summary = llm.invoke(summarizer_prompt).content
    logger.info("Создано саммари найденной информации.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents([summary], metadatas=[{"source": f"Research on {topic}"}])
    archivist_db.add_documents(texts)
    logger.info(f"Саммари по теме '{topic}' успешно добавлено в базу знаний Архивариуса.")
    return f"Информация по теме '{topic}' была успешно исследована, проанализирована и сохранена в моей памяти."

# --- 5. Определение Инструментов ---
# СОЗДАЕМ УМНЫЙ АРХИВАРИУС
# 1. Создаем ретривер, который будет доставать документы из базы
archivist_retriever = archivist_db.as_retriever()
# 2. Создаем промпт, который говорит LLM, как отвечать на основе найденных документов
archivist_prompt = ChatPromptTemplate.from_template("""Твоя задача — ответить на вопрос пользователя, основываясь ТОЛЬКО на предоставленном ниже контексте.
Будь подробным и структурированным. Если в контексте нет ответа, скажи "В моей базе знаний нет информации по этому вопросу".

Контекст:
{context}

Вопрос: {input}""")
# 3. Создаем цепочку, которая объединяет документы в один промпт
document_chain = create_stuff_documents_chain(llm, archivist_prompt)
# 4. Создаем финальную цепочку, которая сначала находит документы, а потом передает их в document_chain
smart_archivist_chain = create_retrieval_chain(archivist_retriever, document_chain)

# (Остальные инструменты остаются прежними)
analyst_chain = RetrievalQA.from_chain_type(llm, retriever=analyst_db.as_retriever())
tools = [
    Tool(
        name="Researcher",
        func=research_and_learn,
        description="Используй, чтобы исследовать новую тему, найти по ней информацию, проанализировать и сохранить ее в долгосрочную память."
    ),
    Tool(
        name="Archivist",
        # Теперь инструмент вызывает нашу новую умную цепочку
        func=lambda d: smart_archivist_chain.invoke(d).get("answer", "Не удалось извлечь ответ."),
        description="Используй для ответов на вопросы по информации, которая УЖЕ ЕСТЬ в памяти."
    ),
    Tool(name="CreateWordDocument", func=create_word_document, description="Используй для создания документа Word (.docx)."),
    Tool(name="Analyst", func=analyst_chain.invoke, description="Используй ТОЛЬКО для ответов на вопросы об общих финансовых и рыночных терминах."),
    Tool(name="CreateExcelDocument", func=create_excel_document, description="Используй для создания документа Excel (.xlsx)."),
    Tool(name="CreatePdfDocument", func=create_pdf_document, description="Используй для создания PDF документа (.pdf).")
]
print(f"✅ Инструменты готовы: {[tool.name for tool in tools]}")

# --- 6. Создание Главного Агента с Памятью ---
# (Этот блок остается без изменений)
memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", input_key="input")
agent_prompt_template = """Ты — автономный ИИ-ассистент... (весь ваш длинный промпт со строгими правилами)"""
agent_prompt = ChatPromptTemplate.from_template(agent_prompt_template)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)
print("✅ Автономный Агент c памятью и улучшенной логикой создан.")

# --- 7. Функции-обработчики для Telegram ---
# (Этот блок остается без изменений)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я ваш автономный ИИ-ассистент. Поставьте мне задачу для исследования.')
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи...')
    try:
        # Передаем в invoke словарь, как и ожидает агент
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
# (Этот блок остается без изменений)
def main() -> None:
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    print("🚀 Запускаю автономного Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()