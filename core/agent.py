# core/agent.py
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from core.config import llm

# Импортируем всех наших экспертов НАПРЯМУЮ
from tools.tool_researcher import researcher_tool
from tools.tool_archivist import archivist_tool
from tools.tool_secretary import secretary_tool
# ИСПРАВЛЕНИЕ: Импортируем не "отдел", а самого "сотрудника"
from tools.fact_checker_department.tools import fact_checker_tool

print("Инициализация Главного Агента и его команды...")

# Собираем команду из ПРЯМЫХ исполнителей
main_tools = [
    fact_checker_tool, # <-- Теперь это прямой инструмент
    researcher_tool,
    archivist_tool,
    secretary_tool,
]

# Промпт для Главного Агента
system_prompt = """Ты — Главный Агент-Руководитель. Твоя задача — общаться с пользователем, помнить контекст диалога и делегировать задачи своей команде экспертов (инструментов).

Твоя команда:
- `FactSearcher`: Эксперт по быстрым фактам из интернета (погода, новости).
- `DeepResearcher`: Эксперт по глубокому исследованию и сохранению знаний.
- `MemoryArchivist`: Эксперт по работе с базой знаний.
- `Secretary`: Эксперт по созданию документов.

Твоя задача — понять истинную цель пользователя и выбрать ОДНОГО наиболее подходящего эксперта для ее выполнения.
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