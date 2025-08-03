# tools/fact_checker_department/tools.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    """Универсальная функция для поиска фактов, включая погоду."""
    if not TAVILY_API_KEY: return "Ошибка: API-ключ для Tavily не найден на сервере."
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: {query}")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results = search.invoke(query)
        
        if not results:
            return "Поиск в интернете не дал результатов по этому запросу."

        # Tavily часто дает прямой ответ в поле 'answer'
        answer = results[0].get('answer')
        if answer:
            logger.info(f"Найден прямой ответ от Tavily: {answer}")
            return answer
            
        # Если прямого ответа нет, возвращаем самое релевантное содержание
        content = results[0].get('content', 'Не удалось извлечь информацию из результата.')
        logger.info(f"Возвращаю контент из первого источника.")
        return content

    except Exception as e:
        logger.error(f"Ошибка в FactSearcher: {e}", exc_info=True)
        return f"Произошла ошибка при поиске факта: {e}"

# Теперь у нас один, но очень умный сотрудник в отделе фактов
fact_checker_tool = Tool(
    name="FactSearcher",
    func=get_fact,
    description="Используй для поиска в интернете любых быстрых фактов: погода, новости, курсы валют, столицы и т.д."
)

fact_checker_tools = [fact_checker_tool]