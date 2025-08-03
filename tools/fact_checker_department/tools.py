# tools/fact_checker_department/tools.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    """Универсальная функция для поиска фактов, включая погоду."""
    if not TAVILY_API_KEY: 
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: '{query}'")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        # ИСПРАВЛЕНИЕ: Мы больше не меняем запрос. Ищем то, что нам передали.
        results = search.invoke(query)
        
        if not results:
            return "Поиск в интернете не дал результатов по этому запросу."

        # ИСПРАВЛЕНИЕ: LangChain TavilySearch возвращает list[dict].
        # Правильно извлекаем 'content'.
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0].get('content', 'Не удалось извлечь информацию из результата.')
        else:
            logger.warning(f"Получен неожиданный формат от Tavily: {results}")
            return "Получен неожиданный формат ответа от поисковой системы."

    except Exception as e:
        logger.error(f"Ошибка в FactSearcher: {e}", exc_info=True)
        return f"Произошла ошибка при поиске факта: {e}"

# Теперь у нас один, но очень умный и надежный сотрудник в отделе фактов
fact_checker_tool = Tool(
    name="InternetFactSearcher",
    func=get_fact,
    description="Используй для поиска в интернете любых быстрых фактов: погода, новости, курсы валют, столицы и т.д."
)

fact_checker_tools = [fact_checker_tool]