# tools/tool_fact_checker.py
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

def quick_internet_search(query: str) -> str:
    """Для быстрых вопросов, не требующих сохранения в память."""
    logger.info(f"Эксперт 'FactChecker': Быстрый поиск по запросу: {query}")
    try:
        search = TavilySearch(max_results=1)
        results = search.invoke(query)
        answer = results[0].get('answer')
        if answer:
            return answer
        return results[0].get('content', 'Не удалось извлечь информацию.')
    except Exception as e:
        return f"Ошибка при быстром поиске: {e}"

fact_checker_tool = Tool(
    name="FactChecker",
    func=quick_internet_search,
    description="Используй для быстрых, фактических вопросов о мире (погода, новости, столицы, курсы валют и т.д.), которые не нужно сохранять."
)