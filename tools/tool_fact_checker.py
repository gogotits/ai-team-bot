# tools/tool_fact_checker.py
import os
import logging
from langchain.agents import Tool
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def quick_internet_search(query: str) -> str:
    """Ищет в интернете быстрые факты (погода, новости, столицы и т.д.)."""
    if not TAVILY_API_KEY: 
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    logger.info(f"Эксперт 'FactChecker': Ищу в интернете по запросу: '{query}'")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results = search.invoke(query)
        
        if not results:
            return "Поиск в интернете не дал результатов по этому запросу."

        # Tavily возвращает список словарей. Берем контент из первого.
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0].get('content', 'Не удалось извлечь контент из результата.')
        else:
            logger.warning(f"Получен неожиданный формат от Tavily: {results}")
            return "Получен неожиданный формат ответа от поисковой системы."

    except Exception as e:
        logger.error(f"Критическая ошибка в FactChecker: {e}", exc_info=True)
        return f"Произошла критическая ошибка при поиске факта: {e}"

fact_checker_tool = Tool(
    name="FactChecker",
    func=quick_internet_search,
    description="Используй для получения быстрых, фактических ответов на вопросы о мире (погода, новости, столицы, курсы валют и т.д.)."
)