# core/agent.py
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from core.config import llm

# Импортируем всех наших экспертов и отделы
from tools.tool_researcher import researcher_tool
from tools.tool_archivist import archivist_tool
from tools.tool_secretary import secretary_tool
from tools.fact_checker_department.agent import fact_checker_agent_executor

print("Инициализация Главного Агента и его команды...")

main_tools = [
    Tool(
        name="FactCheckerDepartment",
        func=lambda user_input_str: fact_checker_agent_executor.invoke({"input": user_input_str}),
        description="Используй этот отдел для получения быстрых, фактических ответов на вопросы о мире (погода, новости, столицы, курсы валют и т.д.)."
    ),
    researcher_tool,
    archivist_tool,
    secretary_tool,
]

# ФИНАЛЬНЫЙ ПРОМПТ ДЛЯ ОТЛАДКИ ("ДУМАЙ ВСЛУХ")
system_prompt = """Ты — Главный Агент-Руководитель. Твоя задача — проанализировать запрос пользователя и делегировать его ОДНОМУ эксперту из твоей команды.

Твоя команда:
- `FactCheckerDepartment`: Для быстрых фактов (погода, новости).
- `DeepResearcher`: Для глубокого исследования и сохранения знаний.
- `MemoryArchivist`: Для поиска в базе знаний.
- `Secretary`: Для создания документов.

Твой финальный ответ ДОЛЖЕН быть в следующем формате:

**План:**
1. [Здесь напиши свои размышления: какой эксперт нужен и почему]
2. [Здесь напиши название эксперта, которого ты выбрал]

**Результат от эксперта:**
[Здесь дословно вставь результат, который вернул эксперт]

**Финальный ответ:**
[Здесь напиши итоговый, дружелюбный ответ для пользователя на основе результата]
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
print("✅ Главный Агент (Руководитель) с режимом отладки готов к работе.")