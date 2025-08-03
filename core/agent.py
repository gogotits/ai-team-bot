# core/agent.py
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from core.config import llm

# Импортируем всех наших экспертов НАПРЯМУЮ
from tools.tool_researcher import researcher_tool
from tools.tool_archivist import archivist_tool
from tools.tool_secretary import secretary_tool
from tools.tool_fact_checker import fact_checker_tool

print("Инициализация Главного Агента и его команды...")

# Собираем команду из ПРЯМЫХ исполнителей
main_tools = [
    fact_checker_tool,
    researcher_tool,
    archivist_tool,
    secretary_tool,
]

# Промпт для Главного Агента
system_prompt = """Ты — Главный Агент-Руководитель. Твоя задача — общаться с пользователем, помнить контекст диалога и делегировать задачи своей команде экспертов (инструментов).

Твоя команда:
- `FactChecker`: Для быстрых фактов из интернета (погода, новости).
- `DeepResearcher`: Для глубокого исследования и сохранения знаний.
- `MemoryArchivist`: Для поиска в твоей базе знаний.
- `Secretary`: Для создания документов.

Твоя задача — понять истинную цель пользователя и выбрать ОДИН наиболее подходящий инструмент для ее выполнения.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

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