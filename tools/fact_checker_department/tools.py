# tools/fact_checker_department/tools.py
from langchain.agents import Tool
from langchain_tavily import TavilySearch

# Сотрудник, отвечающий за погоду
weather_tool = Tool(
    name="WeatherTool",
    func=TavilySearch(max_results=1).invoke,
    description="Используй для получения информации о погоде в конкретном городе."
)

# Сотрудник, отвечающий за курсы валют
currency_tool = Tool(
    name="CurrencyExchangeTool",
    func=TavilySearch(max_results=1).invoke,
    description="Используй для получения информации о курсах валют."
)

# Сотрудник для всех остальных фактов
general_search_tool = Tool(
    name="GeneralSearch",
    func=TavilySearch(max_results=1).invoke,
    description="Используй для поиска любых других общих фактов, которые не касаются погоды или курсов валют."
)

# Собираем всех сотрудников в одну команду
fact_checker_tools = [weather_tool, currency_tool, general_search_tool]