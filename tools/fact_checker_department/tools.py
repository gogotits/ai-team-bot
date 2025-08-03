# tools/fact_checker_department/tools.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

# Получаем ключ API один раз, чтобы использовать для всех инструментов
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def quick_internet_search(query: str) -> str:
    """Для быстрых вопросов, не требующих сохранения в память."""
    logger.info(f"Эксперт 'FactChecker': Быстрый поиск по запросу: {query}")
    if not TAVILY_API_KEY:
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results = search.invoke(query)
        # Обрабатываем ответ как список словарей
        if results and isinstance(results, list) and isinstance(results[0], dict):
            answer = results[0].get('answer')
            if answer:
                return answer
            return results[0].get('content', 'Не удалось извлечь информацию из результата.')
        else:
            return "Получен неожиданный формат ответа от поисковой системы."
    except Exception as e:
        logger.error(f"Ошибка при быстром поиске: {e}", exc_info=True)
        return f"Ошибка при быстром поиске: {e}"

# Инструменты-специалисты для Факт-Чекера
weather_tool = Tool(
    name="WeatherTool",
    func=quick_internet_search,
    description="Используй для получения информации о погоде в конкретном городе."
)
currency_tool = Tool(
    name="CurrencyExchangeTool",
    func=quick_internet_search,
    description="Используй для получения информации о курсах валют."
)
general_search_tool = Tool(
    name="GeneralSearch",
    func=quick_internet_search,
    description="Используй для поиска любых других общих фактов, которые не касаются погоды или курсов валют."
)
fact_checker_tools = [weather_tool, currency_tool, general_search_tool]