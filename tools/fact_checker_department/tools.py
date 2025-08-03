# tools/fact_checker_department/tools.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    """Универсальная и надежная функция для поиска фактов в интернете."""
    if not TAVILY_API_KEY: 
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: '{query}'")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        # Выполняем поиск
        results_data = search.invoke(query)
        
        logger.info(f"Получен ответ от Tavily: {results_data}")

        # --- НАДЕЖНАЯ ЛОГИКА ОБРАБОТКИ ОТВЕТА ---
        if not results_data:
            return "Поиск в интернете не дал результатов."

        # Проверяем, это список словарей (стандартный ответ)
        if isinstance(results_data, list) and results_data and isinstance(results_data[0], dict):
            first_result = results_data[0]
            answer = first_result.get('answer')
            if answer:
                return answer
            return first_result.get('content', 'Не удалось извлечь контент из результата.')
        
        # Проверяем, это список строк (иногда бывает в старых версиях)
        elif isinstance(results_data, list) and results_data and isinstance(results_data[0], str):
            return results_data[0]
            
        else:
            logger.warning(f"Получен неизвестный формат от Tavily: {results_data}")
            return "Получен неожиданный формат ответа от поисковой системы."

    except Exception as e:
        logger.error(f"Критическая ошибка в FactSearcher: {e}", exc_info=True)
        return f"Произошла критическая ошибка при поиске факта: {e}"

# У нас один, универсальный сотрудник в отделе фактов
fact_checker_tool = Tool(
    name="InternetFactSearcher",
    func=get_fact,
    description="Используй для поиска в интернете любых быстрых фактов: погода, новости, курсы валют, столицы и т.д."
)

fact_checker_tools = [fact_checker_tool]