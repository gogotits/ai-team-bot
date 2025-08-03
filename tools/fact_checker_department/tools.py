# tools/fact_checker_department/tools.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_weather(query: str) -> str:
    """Ищет погоду в указанном месте."""
    if not TAVILY_API_KEY: return "Ошибка: API-ключ для Tavily не найден на сервере."
    logger.info(f"Сотрудник 'WeatherTool': Ищу погоду для: {query}")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        # LangChain wrapper для Tavily возвращает list[str]
        results = search.invoke(f"погода в {query}")
        
        if not results:
            return "Поиск погоды в интернете не дал результатов."
        
        # ИСПРАВЛЕНИЕ: Обрабатываем результат как простую строку
        return results[0]
    except Exception as e:
        logger.error(f"Ошибка в WeatherTool: {e}", exc_info=True)
        return f"Произошла ошибка при поиске погоды: {e}"

def get_general_fact(query: str) -> str:
    """Ищет любой другой быстрый факт."""
    if not TAVILY_API_KEY: return "Ошибка: API-ключ для Tavily не найден на сервере."
    logger.info(f"Сотрудник 'GeneralSearch': Ищу факт по запросу: {query}")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results = search.invoke(query)

        if not results:
            return "Поиск в интернете не дал результатов."

        # ИСПРАВЛЕНИЕ: Обрабатываем результат как простую строку
        return results[0]
    except Exception as e:
        logger.error(f"Ошибка в GeneralSearch: {e}", exc_info=True)
        return f"Ошибка при поиске факта: {e}"

weather_tool = Tool(
    name="WeatherTool",
    func=get_weather,
    description="Используй для получения информации о погоде в конкретном городе."
)

general_search_tool = Tool(
    name="GeneralSearch",
    func=get_general_fact,
    description="Используй для поиска любых других общих фактов (новости, столицы, курсы валют)."
)

fact_checker_tools = [weather_tool, general_search_tool]