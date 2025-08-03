# core/analytics.py
import sqlite3
import datetime
import logging

logger = logging.getLogger(__name__)
DB_PATH = "/var/data/analytics.db"

def init_analytics_db():
    """Создает таблицу для логов, если она не существует."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                query TEXT NOT NULL
            )
        ''')
        con.commit()
        con.close()
        logger.info("База данных для аналитики успешно инициализирована.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы аналитики: {e}", exc_info=True)

def log_action(agent_name: str, query: str):
    """Записывает действие в базу данных статистики."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cur.execute("INSERT INTO usage_logs (timestamp, agent_name, query) VALUES (?, ?, ?)",
                    (timestamp, agent_name, query))
        con.commit()
        con.close()
        logger.info(f"Действие для агента '{agent_name}' залогировано.")
    except Exception as e:
        logger.error(f"Ошибка при логировании действия: {e}", exc_info=True)

def get_usage_stats(query: str) -> str:
    """Извлекает статистику из базы данных."""
    logger.info("Эксперт 'Analytics': Запрос на получение статистики.")
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        # Простой пример: считаем, сколько раз вызывался каждый агент
        cur.execute("SELECT agent_name, COUNT(*) FROM usage_logs GROUP BY agent_name")
        rows = cur.fetchall()
        con.close()
        
        if not rows:
            return "Статистика использования пока пуста."
            
        stats_report = "Статистика использования экспертов:\n"
        for row in rows:
            stats_report += f"- {row[0]}: {row[1]} раз(а)\n"
            
        return stats_report
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}", exc_info=True)
        return "Не удалось получить статистику."