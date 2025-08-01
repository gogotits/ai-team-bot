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
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory

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

# --- 4. Функции для создания документов ---
def create_word_document(content: str) -> str:
    doc = WordDocument()
    # Простой способ добавить заголовок, если он есть
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

# --- 5. Создание Инструментов ---
print("Инициализация инструментов...")
search_tool = TavilySearch(max_results=3)
search_tool.description = "Используй для всех вопросов о текущих событиях, фактах, погоде или любой информации из реального мира. Это твой инструмент по умолчанию."
archivist_db = Chroma(persist_directory="./chroma_db_archivist", embedding_function=embeddings)
archivist_chain = RetrievalQA.from_chain_type(llm, retriever=archivist_db.as_retriever())
archivist_tool = Tool(name="Archivist", func=archivist_chain.invoke, description="Используй ТОЛЬКО для ответов на вопросы, которые явно касаются содержания загруженного PDF документа.")
analyst_db = Chroma(persist_directory="./chroma_db_analyst", embedding_function=embeddings)
analyst_chain = RetrievalQA.from_chain_type(llm, retriever=analyst_db.as_retriever())
analyst_tool = Tool(name="Analyst", func=analyst_chain.invoke, description="Используй ТОЛЬКО для ответов на вопросы об общих финансовых и рыночных терминах.")
create_word_tool = Tool(name="CreateWordDocument", func=create_word_document, description="Используй для создания документа Word (.docx) с текстом.")
create_excel_tool = Tool(name="CreateExcelDocument", func=create_excel_document, description="Используй для создания документа Excel (.xlsx). Входные данные: строка, ячейки разделены запятыми, строки - переносом строки.")
create_pdf_tool = Tool(name="CreatePdfDocument", func=create_pdf_document, description="Используй для создания PDF документа (.pdf) с текстом.")
tools = [search_tool, archivist_tool, analyst_tool, create_word_tool, create_excel_tool, create_pdf_tool]
print(f"✅ Инструменты готовы: {[tool.name for tool in tools]}")

# --- 6. Создание Главного Агента с Памятью ---
# Создаем память, которая будет хранить 4 последних сообщения
memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", input_key="input")

# Обновляем промпт, добавляя в него переменную `chat_history`
agent_prompt_template = """Ты — умный и дружелюбный ИИ-ассистент. Твоя задача — ответить на вопрос пользователя, выбрав наиболее подходящий инструмент.
Ты имеешь доступ к истории предыдущего диалога.

ИСТОРИЯ ДИАЛОГА:
{chat_history}

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools}

ИСПОЛЬЗУЙ СЛЕДУЮЩИЙ ФОРМАТ ДЛЯ ОТВЕТА:
Question: вопрос, на который ты должен ответить
Thought: Мои размышления. Какой инструмент лучше всего подходит и почему? Учитываю ли я историю диалога?
Action: Название инструмента из списка [{tool_names}]
Action Input: Входные данные для инструмента.
Observation: Результат выполнения инструмента.
Thought: Теперь у меня есть вся информация для ответа.
Final Answer: Финальный, полный и развернутый ответ на исходный вопрос пользователя.

Начинаем!

Question: {input}
Thought:{agent_scratchpad}"""
agent_prompt = ChatPromptTemplate.from_template(agent_prompt_template)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory, # <-- ДОБАВЛЯЕМ ПАМЯТЬ
    verbose=True,
    handle_parsing_errors=True
)
print("✅ Главный Агент c памятью создан.")

# --- 7. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я ваш автономный ИИ-ассистент. Задайте мне любой вопрос.')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получен текстовый вопрос: '{user_query}'")
    await update.message.reply_text('Думаю...')
    try:
        # Теперь invoke будет учитывать и историю чата
        result = agent_executor.invoke({"input": user_query})
        response_text = result["output"]
        if os.path.exists(response_text) and response_text.endswith(('.docx', '.xlsx', '.pdf')):
            try:
                await context.bot.send_document(chat_id=update.effective_chat.id, document=open(response_text, 'rb'))
                os.remove(response_text)
                logger.info(f"Документ отправлен: {response_text}")
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
    # ... (эта функция остается без изменений) ...
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
    print("🚀 Запускаю Telegram-бота с памятью...")
    application.run_polling()

if __name__ == '__main__':
    main()