# tools/fact_checker_department/tools.py
import os
import logging
from langchain.agents import Tool
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    """Универсальная и надежная функция для поиска фактов в интернете."""
    if not TAVILY_API_KEY: 
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: '{query}'")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results_data = search.invoke(query)
        
        logger.info(f"Получен сырой ответ от Tavily: {results_data}")

        if not results_data:
            return "Поиск в интернете не дал результатов."

        if isinstance(results_data, list) and results_data and isinstance(results_data[0], dict):
            answer = results_data[0].get('answer')
            if answer:
                return answer
            return results_data[0].get('content', 'Не удалось извлечь контент из результата.')
        else:
            logger.warning(f"Получен неизвестный формат от Tavily: {results_data}")
            return "Получен неожиданный формат ответа от поисковой системы."

    except Exception as e:
        logger.error(f"Критическая ошибка в FactSearcher: {e}", exc_info=True)
        return f"Произошла критическая ошибка при поиске факта: {e}"

fact_checker_tool = Tool(
    name="InternetFactSearcher",
    func=get_fact,
    description="Используй для поиска в интернете любых быстрых фактов: погода, новости, курсы валют, столицы и т.д."
)

fact_checker_tools = [fact_checker_tool]