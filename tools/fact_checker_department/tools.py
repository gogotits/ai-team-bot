# tools/fact_checker_department/tools.py
import os
from langchain.agents import Tool
from langchain_tavily import TavilySearch

# Получаем ключ API один раз, чтобы использовать для всех инструментов
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Сотрудник, отвечающий за погоду
weather_tool = Tool(
    name="WeatherTool",
    # ИСПРАВЛЕНИЕ: Явно передаем API-ключ
    func=TavilySearch(max_results=1, api_key=TAVILY_API_KEY).invoke,
    description="Используй для получения информации о погоде в конкретном городе."
)

# Сотрудник, отвечающий за курсы валют
currency_tool = Tool(
    name="CurrencyExchangeTool",
    # ИСПРАВЛЕНИЕ: Явно передаем API-ключ
    func=TavilySearch(max_results=1, api_key=TAVILY_API_KEY).invoke,
    description="Используй для получения информации о курсах валют."
)

# Сотрудник для всех остальных фактов
general_search_tool = Tool(
    name="GeneralSearch",
    # ИСПРАВЛЕНИЕ: Явно передаем API-ключ
    func=TavilySearch(max_results=1, api_key=TAVILY_API_KEY).invoke,
    description="Используй для поиска любых других общих фактов, которые не касаются погоды или курсов валют."
)

# Собираем всех сотрудников в одну команду
fact_checker_tools = [weather_tool, currency_tool, general_search_tool]