# tools/tool_archivist.py
import logging
from langchain.agents import Tool
from core.config import retriever

logger = logging.getLogger(__name__)

def retrieve_from_memory(query: str) -> str:
    logger.info(f"Эксперт 'MemoryArchivist': Поиск в памяти по запросу: {query}")
    docs = retriever.invoke(query)
    if not docs:
        return "В моей базе знаний нет информации по этому вопросу."
    return "\n".join([doc.page_content for doc in docs])

archivist_tool = Tool(
    name="MemoryArchivist",
    func=retrieve_from_memory,
    description="Используй, чтобы найти ответ на вопрос в своей долгосрочной памяти. Всегда пробуй этот инструмент первым для вопросов о ранее исследованных темах."
)