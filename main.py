# --- 1. Загрузка библиотек ---
import os
from dotenv import load_dotenv
import logging

# Основные компоненты для создания Агента
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool

# Инструменты
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- 2. Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
    raise ValueError("Не найдены необходимые ключи API в .env файле!")
print("✅ Ключи API загружены.")

# --- 3. Инициализация LLM и Эмбеддингов ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✅ LLM и модель эмбеддингов инициализированы.")

# --- 4. Создание Инструментов ---

# Инструмент 1: Поиск в интернете
# max_results=3 означает, что он будет возвращать 3 самых релевантных результата
search_tool = TavilySearchResults(max_results=3)

# Инструмент 2: Архивариус (для работы с PDF)
# Мы "оборачиваем" нашего старого агента в новый инструмент
archivist_db = Chroma(persist_directory="./chroma_db_archivist", embedding_function=embeddings)
archivist_chain = RetrievalQA.from_chain_type(llm, retriever=archivist_db.as_retriever())
archivist_tool = Tool(
    name="Archivist",
    func=archivist_chain.invoke,
    description="Используй этот инструмент для ответов на вопросы о содержании конкретного загруженного документа."
)

# Инструмент 3: Аналитик (для работы с финансами)
analyst_db = Chroma(persist_directory="./chroma_db_analyst", embedding_function=embeddings)
analyst_chain = RetrievalQA.from_chain_type(llm, retriever=analyst_db.as_retriever())
analyst_tool = Tool(
    name="Analyst",
    func=analyst_chain.invoke,
    description="Используй этот инструмент для ответов на вопросы об общих финансовых и рыночных терминах (акции, рынок, инвестиции)."
)

# Собираем все инструменты в один список
tools = [search_tool, archivist_tool, analyst_tool]
print(f"✅ Инструменты готовы: {[tool.name for tool in tools]}")


# --- 5. Создание Главного Агента ---

# Улучшенный промпт с более четкими инструкциями для агента
agent_prompt_template = """Ты — умный ИИ-ассистент. Твоя задача — ответить на вопрос пользователя, выбрав наиболее подходящий инструмент.

ТЫ ДОЛЖЕН СЛЕДОВАТЬ ЭТОМУ АЛГОРИТМУ:
1.  Проанализируй вопрос пользователя.
2.  Посмотри на список доступных инструментов и их описания.
3.  Выбери ОДИН инструмент, который лучше всего подходит для ответа.
4.  Если сомневаешься, используй инструмент "tavily_search_results_json".

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools}

ИСПОЛЬЗУЙ СЛЕДУЮЩИЙ ФОРМАТ ДЛЯ ОТВЕТА:

Question: вопрос, на который ты должен ответить
Thought: Мои размышления. Какой инструмент лучше всего подходит и почему?
Action: Название инструмента из списка [{tool_names}]
Action Input: Входные данные для инструмента (обычно это сам вопрос пользователя).
Observation: Результат выполнения инструмента (это поле заполняется автоматически).
... (этот цикл Thought/Action/Action Input/Observation может повторяться, если нужно использовать несколько инструментов последовательно)
Thought: Теперь у меня есть вся информация для ответа.
Final Answer: Финальный, полный и развернутый ответ на исходный вопрос пользователя.

Начинаем!

Question: {input}
Thought:{agent_scratchpad}"""

# Создаем промпт на основе шаблона
agent_prompt = PromptTemplate.from_template(agent_prompt_template)

# Также давайте улучшим описания самих инструментов
archivist_tool.description = "Используй ТОЛЬКО для ответов на вопросы, которые явно касаются содержания загруженного PDF документа. Например: 'что говорится в документе о...?'"
analyst_tool.description = "Используй ТОЛЬКО для ответов на вопросы об общих финансовых и рыночных терминах. Например: 'что такое бычий рынок?'"
search_tool.description = "Используй для всех остальных вопросов, особенно если они касаются текущих событий, фактов, погоды или любой информации из реального мира. Это твой инструмент по умолчанию."


# Создаем Агента
agent = create_react_agent(llm, tools, agent_prompt)

# Создаем Исполнителя Агента (Agent Executor)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
print("✅ Главный Агент создан с улучшенной логикой.")


# --- 6. Запуск цикла общения ---
print("\n✅ Агент с доступом в интернет готов. Задавайте любой вопрос.")
print("   Для выхода введите 'exit'.")

while True:
    query = input("\n> Ваш вопрос: ")
    if query.lower() == 'exit':
        print("До свидания!")
        break

    try:
        # Вызываем исполнителя агента
        result = agent_executor.invoke({"input": query})
        print("\nОтвет: ", result['output'])
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")