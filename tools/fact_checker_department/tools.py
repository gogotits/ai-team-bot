# tools/fact_checker_department/tools.py
import os
import logging
from langchain_tavily import TavilySearch
from langchain.agents import Tool
# Импортируем нашу основную LLM, чтобы сотрудник мог думать
from core.config import llm

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def get_fact(query: str) -> str:
    """Универсальная и надежная функция для поиска и анализа фактов."""
    if not TAVILY_API_KEY: 
        return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    logger.info(f"Сотрудник 'FactSearcher': Ищу факт по запросу: '{query}'")
    try:
        search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
        results = search.invoke(query)
        
        if not results or not isinstance(results, list) or not results[0]:
            return "Поиск в интернете не дал результатов."

        raw_content = results[0].get('content', '')
        if not raw_content:
            return "Не удалось извлечь контент из результата поиска."

        # ШАГ АНАЛИЗА: Просим LLM извлечь суть из найденного текста
        analysis_prompt = f"""Проанализируй следующий текст, найденный в интернете, и дай краткий и точный ответ на вопрос пользователя.
        Вопрос пользователя: "{query}"
        Найденный текст: "{raw_content}"
        
        Твой финальный ответ должен быть только сутью, без лишних фраз.
        """
        
        clean_answer = llm.invoke(analysis_prompt).content
        logger.info(f"Сотрудник 'FactSearcher': Сформулирован чистый ответ: {clean_answer}")
        return clean_answer

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