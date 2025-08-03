# tools/fact_checker_department/tools.py
import os
from langchain.agents import Tool
from langchain_tavily import TavilySearch

# Получаем ключ API один раз
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_weather(query: str) -> str:
    """Ищет погоду в указанном месте."""
    if not TAVILY_API_KEY: return "Ошибка: API-ключ для Tavily не найден на сервере."
    search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
    results = search.invoke(f"погода в {query}")
    return results[0].get('content', 'Не удалось найти информацию о погоде.')

def get_general_fact(query: str) -> str:
    """Ищет любой другой быстрый факт."""
    if not TAVILY_API_KEY: return "Ошибка: API-ключ для Tavily не найден на сервере."
    search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
    results = search.invoke(query)
    answer = results[0].get('answer')
    if answer: return answer
    return results[0].get('content', 'Не удалось найти информацию.')

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