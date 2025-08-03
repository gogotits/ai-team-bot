# tools/fact_checker_department/tools.py
import os
import logging
from langchain.agents import Tool
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    if not TAVILY_API_KEY: return "Ошибка: API-ключ для Tavily не найден."
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: '{query}'")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results = search.invoke(query)
        if not results: return "Поиск в интернете не дал результатов."
        return results[0].get('content', 'Не удалось извлечь контент.')
    except Exception as e:
        return f"Ошибка при поиске факта: {e}"

fact_checker_tool = Tool(
    name="InternetFactSearcher",
    func=get_fact,
    description="Используй для поиска в интернете любых быстрых фактов: погода, новости, и т.д."
)
fact_checker_tools = [fact_checker_tool]