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
from langchain_core.messages import HumanMessage, AIMessage
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

# --- 5. ФУНКЦИИ-ЭКСПЕРТЫ ---

def research_and_learn(topic: str) -> str:
    logger.info(f"Эксперт 'DeepResearcher': Начинаю исследование по теме: {topic}")
    try:
        # ИСПРАВЛЕНИЕ: Явно передаем API-ключ для надежности
        search = TavilySearch(max_results=3, api_key=os.environ.get("TAVILY_API_KEY"))
        search_results = search.invoke(topic)
        raw_text = "\n\n".join([result.get('content', '') for result in search_results])
        if not raw_text.strip(): return "Поиск в интернете не дал результатов."
        
        summarizer_prompt = f"""Проанализируй текст по теме '{topic}'. Создай качественное саммари на русском языке. Ответ должен содержать только саммари."""
        summary = llm.invoke(summarizer_prompt).content
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([summary], metadatas=[{"source": f"Research on {topic}"}])
        main_db.add_documents(texts)
        
        return f"Информация по теме '{topic}' была успешно исследована и сохранена в моей памяти."
    except Exception as e:
        logger.error(f"Ошибка в работе 'DeepResearcher': {e}", exc_info=True)
        return f"В процессе исследования произошла ошибка: {e}"

def retrieve_from_memory(query: str) -> str:
    logger.info(f"Эксперт 'MemoryArchivist': Поиск в памяти по запросу: {query}")
    docs = retriever.invoke(query)
    if not docs:
        return "В моей базе знаний нет информации по этому вопросу."
    return "\n".join([doc.page_content for doc in docs])

def quick_internet_search(query: str) -> str:
    logger.info(f"Эксперт 'FactChecker': Быстрый поиск по запросу: {query}")
    try:
        # ИСПРАВЛЕНИЕ: Явно передаем API-ключ для надежности
        search = TavilySearch(max_results=1, api_key=os.environ.get("TAVILY_API_KEY"))
        results = search.invoke(query)
        answer = results[0].get('answer')
        if answer:
            return answer
        return results[0].get('content', 'Не удалось извлечь информацию.')
    except Exception as e:
        return f"Ошибка при быстром поиске: {e}"

def create_document(input_str: str) -> str:
    logger.info(f"Эксперт 'Secretary': Получена задача на создание документа.")
    try:
        parts = input_str.split('|')
        if len(parts) != 2:
            return "Ошибка: неверный формат. Используйте 'текст|тип документа' (word, excel, pdf)."
        content, doc_type = parts[0].strip(), parts[1].strip().lower()

        if doc_type == 'word':
            doc = WordDocument()
            doc.add_paragraph(content)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx", prefix="report_")
            doc.save(temp_file.name)
            return f"Документ Word успешно создан: {temp_file.name}"
        elif doc_type == 'excel':
            wb = ExcelWorkbook()
            ws = wb.active
            for line in content.split('\n'):
                ws.append(line.split(','))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", prefix="table_")
            wb.save(temp_file.name)
            return f"Документ Excel успешно создан: {temp_file.name}"
        elif doc_type == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', '', 12)
            pdf.multi_cell(0, 10, content)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="document_")
            pdf.output(temp_file.name)
            return f"PDF документ успешно создан: {temp_file.name}"
        else:
            return "Неподдерживаемый тип документа. Доступные типы: word, excel, pdf."
    except Exception as e:
        return f"Ошибка при создании документа: {e}"

# --- 6. СОЗДАНИЕ АГЕНТА-СОБЕСЕДНИКА ---
conversational_tools = [
    Tool(
        name="MemoryArchivist",
        func=retrieve_from_memory,
        description="Используй, чтобы найти ответ на вопрос в своей долгосрочной памяти."
    ),
]
system_prompt_conv = """Ты — дружелюбный ИИ-ассистент. Твоя задача — поддерживать беседу с пользователем.
Если ты можешь ответить на вопрос, используя свою память (`MemoryArchivist`), сделай это.
Если пользователь задает общий вопрос, на который ты не знаешь ответа, вежливо скажи, что не знаешь, и предложи ему использовать команды "исследуй" или "посмотри в интернете"."""

prompt_conv = ChatPromptTemplate.from_messages([
    ("system", system_prompt_conv),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
conversational_agent_executor = AgentExecutor(
    agent=create_tool_calling_agent(llm, conversational_tools, prompt_conv),
    tools=conversational_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
print("✅ Агент-Собеседник готов к работе.")

# --- 7. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if 'chat_history' in context.user_data:
        del context.user_data['chat_history']
    memory.clear()
    await update.message.reply_text('Привет! Я ваш ИИ-ассистент. Память очищена. Задайте мне вопрос или дайте команду.')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи...')
    
    response_text = ""
    
    try:
        # ЖЕСТКИЙ ДИСПЕТЧЕР НА ОСНОВЕ КОМАНД
        if user_query.lower().startswith("исследуй"):
            topic = user_query[len("исследуй"):].strip()
            response_text = research_and_learn(topic)
        elif user_query.lower().startswith("посмотри в интернете"):
            topic = user_query[len("посмотри в интернете"):].strip()
            response_text = quick_internet_search(topic)
        elif user_query.lower().startswith("предоставь информацию из памяти"):
            topic = user_query[len("предоставь информацию из памяти"):].strip()
            response_text = retrieve_from_memory(topic)
        elif user_query.lower().startswith("создай документ"):
            topic = user_query[len("создай документ"):].strip()
            response_text = create_document(topic)
        # Удаляем 'аналитика' для упрощения и стабильности
        # elif user_query.lower().startswith("актуализируй"):
        #     topic = user_query[len("актуализируй"):].strip()
        #     response_text = analyze_and_update_memory(topic)
        else:
            # Если нет команды, вызываем агента-собеседника
            chat_history = context.user_data.get('chat_history', [])
            result = conversational_agent_executor.invoke({"input": user_query, "chat_history": chat_history})
            response_text = result["output"]
            context.user_data['chat_history'] = result['chat_history']

        if response_text.startswith("Документ") and ('.docx' in response_text or '.xlsx' in response_text or '.pdf' in response_text):
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
    print("🚀 Запускаю Telegram-бота с жестким диспетчером...")
    application.run_polling()

if __name__ == '__main__':
    main()