# core/agent.py
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from core.config import llm

# Импортируем всех наших экспертов и отделы
from tools.tool_researcher import researcher_tool
from tools.tool_archivist import archivist_tool
from tools.tool_secretary import secretary_tool
# Импортируем нашего нового "начальника отдела"
from tools.fact_checker_department.agent import fact_checker_agent_executor

print("Инициализация Главного Агента и его команды...")

# Собираем команду экспертов (начальников отделов)
main_tools = [
    Tool(
        name="FactCheckerDepartment",
        func=fact_checker_agent_executor.invoke,
        description="Используй этот отдел для получения быстрых, фактических ответов на вопросы о мире (погода, новости, столицы, курсы валют и т.д.)."
    ),
    researcher_tool,
    archivist_tool,
    secretary_tool,
]

# Промпт для Главного Агента
system_prompt = """Ты — Главный Агент-Руководитель. Твоя задача — общаться с пользователем, помнить контекст диалога и делегировать задачи своей команде экспертов (инструментов).

Твоя команда:
- `FactCheckerDepartment`: Отдел быстрых фактов (погода, новости).
- `DeepResearcher`: Отдел глубоких исследований и сохранения знаний.
- `MemoryArchivist`: Отдел по работе с базой знаний.
- `Secretary`: Отдел по созданию документов.

Твоя задача — понять истинную цель пользователя и выбрать ОДИН наиболее подходящий отдел для ее выполнения.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Создаем Главного Агента и его Исполнителя
agent = create_tool_calling_agent(llm, main_tools, prompt)
memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(
    agent=agent,
    tools=main_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
print("✅ Главный Агент (Руководитель) и его команда готовы к работе.")