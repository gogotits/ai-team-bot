# core/agent.py
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from core.config import llm

# Импортируем всех наших экспертов
from tools.tool_fact_checker import fact_checker_tool
from tools.tool_researcher import researcher_tool
from tools.tool_archivist import archivist_tool
from tools.tool_secretary import secretary_tool

print("Инициализация Главного Агента и его команды...")

# Собираем команду экспертов
main_tools = [
    fact_checker_tool,
    researcher_tool,
    archivist_tool,
    secretary_tool,
]

# Классический и надежный шаблон для ReAct агента
template = """Отвечай на следующие вопросы как можно лучше. У тебя есть доступ к следующим инструментам:

{tools}

Используй следующий формат:

Question: вопрос, на который ты должен ответить
Thought: ты должен подумать, что делать.
Action: действие, которое нужно предпринять. Должно быть одним из [{tool_names}]
Action Input: входные данные для действия.
Observation: результат действия.
... (этот цикл Thought/Action/Action Input/Observation может повторяться)
Thought: Теперь я знаю финальный ответ.
Final Answer: финальный ответ на исходный вопрос.

Начинаем! Помни отвечать на языке пользователя.

Previous conversation history:
{chat_history}

New question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, main_tools, prompt)
memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=False)
agent_executor = AgentExecutor(
    agent=agent,
    tools=main_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True # <-- ВКЛЮЧАЕМ РЕЖИМ ПОЛНОЙ ПРОЗРАЧНОСТИ
)
print("✅ Единый универсальный ReAct агент с режимом отладки создан.")