# main.py
import os
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters
# Убедимся, что config импортируется первым для загрузки .env
import core.config
from telegram_bot.handlers import start, handle_text_message, handle_photo_message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main() -> None:
    """Основная функция запуска бота."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Не найден TELEGRAM_BOT_TOKEN!")

    application = Application.builder().token(token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    
    print("🚀 Запускаю иерархического Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()