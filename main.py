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

# --- 4. ЕДИНАЯ БАЗА ЗНАНИЙ И ФУНКЦИИ-ИНСТРУМЕНТЫ ---
print("Инициализация единой базы знаний...")
persistent_storage_path = "/var/data/main_chroma_db"
main_db = Chroma(persist_directory=persistent_storage_path, embedding_function=embeddings)
retriever = main_db.as_retriever(search_kwargs={'k': 5})
print(f"✅ Единая база знаний готова. Записей в базе: {main_db._collection.count()}")

# --- Функции-инструменты ---
def research_and_learn(topic: str) -> str:
    logger.info(f"Инструмент 'research_and_learn': Начинаю исследование по теме: {topic}")
    search = TavilySearch(max_results=3)
    try:
        search_results = search.invoke(topic)
    except Exception as e:
        logger.error(f"Ошибка при поиске в Tavily: {e}")
        return "Произошла ошибка при доступе к поисковой системе."
    if not search_results:
        return "Не удалось найти информацию по данной теме в интернете."
    
    raw_text = "\n\n".join([result.get('content', '') for result in search_results])
    
    summarizer_prompt = f"""Проанализируй следующий текст по теме '{topic}'. Создай качественное, структурированное саммари на русском языке. Твой ответ должен содержать только саммари. ТЕКСТ:\n{raw_text}"""
    summary = llm.invoke(summarizer_prompt).content
    logger.info("Создано саммари найденной информации.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents([summary], metadatas=[{"source": f"Research on {topic}"}])
    
    main_db.add_documents(texts)
    logger.info(f"Саммари по теме '{topic}' успешно добавлено в единую базу знаний.")
    return f"Информация по теме '{topic}' была успешно исследована и сохранена в моей памяти."

def answer_question_tool(query: str) -> str:
    """Универсальный инструмент для ответа на вопросы. Сначала ищет в памяти, потом в интернете."""
    logger.info(f"Универсальный инструмент: Поиск ответа на вопрос: {query}")
    # Шаг 1: Поиск в памяти
    memory_docs = retriever.invoke(query)
    memory_context = "\n".join([doc.page_content for doc in memory_docs])
    
    # Шаг 2: Создаем промпт для LLM, чтобы она решила, достаточно ли информации
    decision_prompt = f"""Проанализируй следующий вопрос пользователя и найденную в моей памяти информацию.
    Вопрос: "{query}"
    Найденная информация: "{memory_context}"

    Если найденная информация полностью и исчерпывающе отвечает на вопрос, скажи только "ДОСТАТОЧНО".
    В противном случае, скажи только "ИСКАТЬ В ИНТЕРНЕТЕ"."""

    decision = llm.invoke(decision_prompt).content
    
    if "ДОСТАТОЧНО" in decision:
        logger.info("Принято решение: информации в памяти достаточно.")
        return memory_context
    else:
        logger.info("Принято решение: нужно искать в интернете.")
        search = TavilySearch(max_results=3)
        try:
            search_results = search.invoke(query)
            if not search_results:
                return "Не удалось найти информацию в интернете."
            internet_context = "\n\n".join([res.get('content', '') for res in search_results])
            return internet_context
        except Exception as e:
            return f"Произошла ошибка при поиске в интернете: {e}"

def create_word_document(content: str) -> str:
    doc = WordDocument()
    doc.add_paragraph(content)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx", prefix="report_")
    doc.save(temp_file.name)
    logger.info(f"Создан Word документ: {temp_file.name}")
    return f"Документ Word успешно создан и доступен по пути: {temp_file.name}"

# --- 5. Создание ЕДИНОГО АГЕНТА И ЕГО ИНСТРУМЕНТОВ ---
tools = [
    Tool(
        name="answer_question",
        func=answer_question_tool,
        description="Используй этот универсальный инструмент для ответа на ЛЮБОЙ вопрос пользователя (например, о погоде, фактах, или по ранее исследованным темам)."
    ),
    Tool(
        name="research_and_learn",
        func=research_and_learn,
        description="Используй этот инструмент, только когда пользователь прямо просит 'исследуй' или 'найди и сохрани', чтобы найти и сохранить в память обширную информацию."
    ),
    Tool(
        name="create_word_document",
        func=create_word_document,
        description="Используй для создания документа Microsoft Word (.docx), когда пользователь просит об этом."
    ),
]
print(f"✅ Инструменты готовы: {[tool.name for tool in tools]}")

system_prompt = """Ты — умный и дружелюбный ИИ-ассистент. Твоя главная задача — помогать пользователю.
У тебя есть инструменты. Ты должен сам решать, какой инструмент лучше всего подходит.

Твои принципы:
- Для ответа на любой обычный вопрос используй `answer_question`.
- Для больших исследовательских задач по команде 'исследуй' используй `research_and_learn`.
- Для создания файлов используй `create_word_document`.
- Всегда будь вежлив и старайся помочь.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
memory = ConversationBufferWindowMemory(k=8, memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
print("✅ Единый универсальный агент создан.")

# --- 6. Функции-обработчики для Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['chat_history'] = []
    await update.message.reply_text('Привет! Я ваш универсальный ИИ-ассистент. Я помню наш диалог. Задайте мне вопрос.')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи...')
    
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

# --- 7. Основная функция запуска бота ---
def main() -> None:
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    print("🚀 Запускаю Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()