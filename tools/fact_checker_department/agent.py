# tools/fact_checker_department/agent.py
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core.config import llm
from .tools import fact_checker_tools

system_prompt = """Ты — Агент, Начальник отдела проверки фактов. Твоя задача — получить запрос и делегировать его одному из твоих подчиненных специалистов.

Твои специалисты:
- `WeatherTool`: Специалист по погоде.
- `GeneralSearch`: Специалист по всем остальным вопросам.

Проанализируй запрос и выбери ОДНОГО специалиста для выполнения задачи.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

fact_checker_agent_runnable = create_tool_calling_agent(llm, fact_checker_tools, prompt)
fact_checker_agent_executor = AgentExecutor(
    agent=fact_checker_agent_runnable, 
    tools=fact_checker_tools, 
    verbose=True,
    handle_parsing_errors=True
)