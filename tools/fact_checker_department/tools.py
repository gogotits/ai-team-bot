# tools/fact_checker_department/tools.py
import os
import logging
from langchain.agents import Tool
# ИСПРАВЛЕНИЕ: Импортируем прямую библиотеку Tavily
from tavily import TavilyClient

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    """Универсальная и надежная функция для поиска фактов в интернете."""
    if not TAVILY_API_KEY: 
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: '{query}'")
    try:
        # ИСПРАВЛЕНИЕ: Используем прямой клиент TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response_dict = client.search(query=query, search_depth="basic")
        
        logger.info(f"Получен сырой ответ от Tavily: {response_dict}")

        if not response_dict:
            return "Поиск в интернете не дал результатов."

        # Теперь мы знаем точную структуру и можем ее надежно обработать
        results_list = response_dict.get('results', [])
        if not results_list:
            direct_answer = response_dict.get('answer')
            if direct_answer: return direct_answer
            return "Поиск не дал конкретных результатов."

        first_result = results_list[0]
        return first_result.get('content', 'Не удалось извлечь контент из результата.')

    except Exception as e:
        logger.error(f"Критическая ошибка в FactSearcher: {e}", exc_info=True)
        return f"Произошла критическая ошибка при поиске факта: {e}"

fact_checker_tool = Tool(
    name="InternetFactSearcher",
    func=get_fact,
    description="Используй для поиска в интернете любых быстрых фактов: погода, новости, курсы валют, столицы и т.д."
)

fact_checker_tools = [fact_checker_tool]