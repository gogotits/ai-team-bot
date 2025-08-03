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

# ФИНАЛЬНЫЙ ПРОМПТ ДЛЯ ОТЛАДКИ ("ПРОЗРАЧНЫЙ РЕЖИМ")
system_prompt = """Ты — Агент-Диспетчер. Твоя единственная задача — проанализировать запрос пользователя, выбрать ОДИН из доступных инструментов-экспертов и вызвать его.
Твой финальный ответ (`Final Answer`) ДОЛЖЕН БЫТЬ ТОЛЬКО тем результатом, который вернул вызванный тобой инструмент. Ничего не добавляй и не изменяй. Просто передай ответ как есть.

Твоя команда:
- `FactCheckerDepartment`: Отдел быстрых фактов.
- `DeepResearcher`: Отдел глубоких исследований и сохранения знаний.
- `MemoryArchivist`: Отдел по работе с базой знаний.
- `Secretary`: Отдел по созданию документов.
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
print("✅ Главный Агент (Руководитель) в режиме отладки готов к работе.")