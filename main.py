# main.py
import os
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram_bot.handlers import start, handle_text_message, handle_photo_message

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main() -> None:
    """Основная функция запуска бота."""
    # Получаем токен из config, который загрузил его из .env
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    
    application = Application.builder().token(token).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    
    print("🚀 Запускаю иерархического Telegram-бота...")
    application.run_polling()

if __name__ == '__main__':
    main()