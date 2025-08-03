# tools/tool_fact_checker.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

def quick_internet_search(query: str) -> str:
    """Для быстрых вопросов, не требующих сохранения в память."""
    logger.info(f"Эксперт 'FactChecker': Быстрый поиск по запросу: {query}")
    try:
        # Явно передаем API-ключ для надежности
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Ошибка: API-ключ для Tavily не найден."
            
        search = TavilySearch(max_results=1, api_key=api_key)
        results = search.invoke(query)
        
        # Обрабатываем результат как список словарей
        if results and isinstance(results, list) and isinstance(results[0], dict):
            answer = results[0].get('answer')
            if answer:
                return answer
            return results[0].get('content', 'Не удалось извлечь информацию из результата.')
        else:
            # Если формат другой, возвращаем как есть, чтобы увидеть, что пришло
            return str(results)

    except Exception as e:
        logger.error(f"Ошибка при быстром поиске: {e}", exc_info=True)
        return f"Ошибка при быстром поиске: {e}"

fact_checker_tool = Tool(
    name="FactChecker",
    func=quick_internet_search,
    description="Используй для быстрых, фактических вопросов о мире (погода, новости, столицы, курсы валют и т.д.), которые не нужно сохранять."
)