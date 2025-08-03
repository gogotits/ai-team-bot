# tools/tool_analytics.py
from langchain.agents import Tool
from core.analytics import get_usage_stats

analytics_tool = Tool(
    name="UsageAnalytics",
    func=get_usage_stats,
    description="Используй, когда пользователь просит предоставить статистику использования, например 'сколько раз я задавал вопросы' или 'какая статистика'."
)