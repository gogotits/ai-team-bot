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
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool
# НОВЫЕ ИМПОРТЫ для управления историей
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

from langchain_tavily import TavilySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# НОВЫЕ ИМПОРТЫ для создания умной цепочки
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
print(f"✅ Единая база знаний готова. Записей в базе: {main_db._collection.count()}")

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
    summarizer_prompt = f"""Проанализируй следующий текст по теме '{topic}'. Создай структурированное саммари. Ответь только текстом саммари."""
    summary = llm.invoke(summarizer_prompt).content
    logger.info("Создано саммари найденной информации.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents([summary], metadatas=[{"source": f"Research on {topic}"}])
    main_db.add_documents(texts)
    logger.info(f"Саммари по теме '{topic}' успешно добавлено в единую базу знаний.")
    return f"Информация по теме '{topic}' была успешно исследована и сохранена в моей памяти."

# --- 5. Определение Инструментов ---
# СОЗДАЕМ САМЫЙ УМНЫЙ АРХИВАРИУС, ПОНИМАЮЩИЙ КОНТЕКСТ
retriever = main_db.as_retriever(search_kwargs={'k': 5})
contextualize_q_system_prompt = """Учитывая историю чата и последний вопрос пользователя, который может ссылаться на контекст в истории чата, сформулируй независимый вопрос, который можно понять без истории чата. НЕ отвечай на вопрос, просто переформулируй его, если это необходимо, в противном случае верни его как есть."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_system_prompt = """Ты ассистент, который отвечает на вопросы. Используй приведенные ниже фрагменты информации, чтобы ответить на вопрос. Если ты не знаешь ответа, просто скажи, что не знаешь. Будь краток, но содержателен.\n\n{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
smart_archivist_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

# СОЗДАЕМ ДВА НАБОРА ИНСТРУМЕНТОВ
query_tools = [
    Tool(
        name="Archivist", 
        func=lambda d: smart_archivist_chain.invoke(d).get("answer", "Не удалось извлечь ответ."),
        description="Используй для ответов на вопросы по информации, которая УЖЕ ЕСТЬ в памяти."
    ),
    Tool(name="CreateWordDocument", func=create_word_document, description="Используй для создания документа Word (.docx)."),
    Tool(name="CreateExcelDocument", func=create_excel_document, description="Используй для создания документа Excel (.xlsx)."),
    Tool(name="CreatePdfDocument", func=create_pdf_document, description="Используй для создания PDF документа (.pdf).")
]
research_tools = [
    Tool(name="Researcher", func=research_and_learn, description="Используй, чтобы исследовать новую тему и сохранить ее в долгосрочную память.")
]
print(f"✅ Инструменты для ответов готовы: {[tool.name for tool in query_tools]}")
print(f"✅ Инструменты для исследования готовы: {[tool.name for tool in research_tools]}")

# --- 6. Создание ДВУХ Главных Агентов ---
agent_prompt_template = """Ты — автономный ИИ-ассистент. Твоя главная задача — выполнять цели, поставленные пользователем.
Ты должен строго следовать инструкциям и использовать инструменты максимально эффективно.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools}

ВАЖНЫЕ ПРАВИЛА:
1.  **Проверка Памяти:** Прежде чем искать новую информацию ('Researcher'), всегда сначала проверь, нет ли ответа в памяти ('Archivist').
2.  **Отчет об Ошибке:** Если 'Archivist' не находит информации, твой 'Final Answer' должен быть: "В моей базе знаний нет информации по этому вопросу. Пожалуйста, дайте команду на исследование."
3.  **Создание Файлов:** Используй инструменты для создания документов ТОЛЬКО когда пользователь ЯВНО попросил об этом.
4.  **Вывод Файла:** Если ты создаешь документ, твой 'Final Answer' должен быть ТОЛЬКО путем к файлу.

ИСПОЛЬЗУЙ СЛЕДУЮЩИЙ ФОРМАТ:
Question: вопрос пользователя.
Thought: Мои размышления и план действий.
Action: Название инструмента из списка [{tool_names}]
Action Input: Входные данные для инструмента.
Observation: Результат выполнения инструмента.
Thought: Я достиг цели.
Final Answer: Финальный ответ пользователю.

Начинаем!

Question: {input}
Thought:{agent_scratchpad}"""
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", agent_prompt_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Агент для ОТВЕТОВ (без доступа к Researcher)
query_agent = create_react_agent(llm, query_tools, agent_prompt)
query_agent_executor = AgentExecutor(agent=query_agent, tools=query_tools, verbose=True, handle_parsing_errors=True)
# Агент для ИССЛЕДОВАНИЯ (только с Researcher)
research_agent = create_react_agent(llm, research_tools, agent_prompt)
research_agent_executor = AgentExecutor(agent=research_agent, tools=research_tools, verbose=True, handle_parsing_errors=True)
print("✅ Два типа агентов (Query и Research) созданы.")

# --- 7. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['chat_history'] = [] # Очищаем историю при старте
    await update.message.reply_text('Привет! Я ваш автономный ИИ-ассистент. Задайте мне вопрос или дайте команду на исследование (начните с "исследуй", "найди" или "сохрани").')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи...')
    
    try:
        # Получаем историю чата для этого пользователя
        chat_history = context.user_data.get('chat_history', [])
        
        if user_query.lower().startswith(('исследуй', 'найди', 'сохрани')):
            logger.info("Активирован РЕЖИМ ИССЛЕДОВАНИЯ")
            result = research_agent_executor.invoke({"input": user_query, "chat_history": chat_history})
        else:
            logger.info("Активирован РЕЖИМ ОТВЕТОВ")
            result = query_agent_executor.invoke({"input": user_query, "chat_history": chat_history})
        
        # Обновляем историю чата
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(HumanMessage(content=result['output'])) # Используем HumanMessage для простоты
        context.user_data['chat_history'] = chat_history[-8:] # Ограничиваем историю

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
        
        message_payload = HumanMessage(
            content=[
                {"type": "text", "text": user_caption},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
            ]
        )
        response = llm.invoke([message_payload])
        await update.message.reply_text(response.content)
        logger.info("Отправлен ответ на изображение.")
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