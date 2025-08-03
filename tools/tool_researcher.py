# tools/tool_researcher.py
import logging
import os
from langchain_tavily import TavilySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool
from core.config import llm, db

logger = logging.getLogger(__name__)

def research_and_learn(topic: str) -> str:
    logger.info(f"Эксперт 'DeepResearcher': Начинаю исследование по теме: {topic}")
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key: return "Ошибка: API-ключ для Tavily не найден на сервере."
    
    search = TavilySearch(max_results=3, api_key=api_key)
    try:
        search_results = search.invoke(topic)
        raw_text = "\n\n".join([result.get('content', '') for result in search_results])
        if not raw_text.strip(): return "Поиск в интернете не дал результатов."
        
        summarizer_prompt = f"""Проанализируй текст по теме '{topic}'. Создай качественное саммари. Ответ должен содержать только саммари."""
        summary = llm.invoke(summarizer_prompt).content
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([summary], metadatas=[{"source": f"Research on {topic}"}])
        db.add_documents(texts)
        
        return f"Информация по теме '{topic}' была успешно исследована и сохранена в моей памяти."
    except Exception as e:
        logger.error(f"Ошибка в работе 'DeepResearcher': {e}", exc_info=True)
        return f"В процессе исследования произошла ошибка: {e}"

researcher_tool = Tool(
    name="DeepResearcher",
    func=research_and_learn,
    description="Используй, когда пользователь прямо просит 'исследуй', 'найди и сохрани', чтобы найти и сохранить в память обширную информацию по сложной теме."
)