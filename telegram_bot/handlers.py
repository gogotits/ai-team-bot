# telegram_bot/handlers.py
import os
import logging
import base64
from telegram import Update
from telegram.ext import ContextTypes
from langchain_core.messages import HumanMessage
from core.agent import agent_executor
from core.config import llm

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agent_executor.memory.clear()
    await update.message.reply_text('Привет! Я ваш ИИ-ассистент. Память очищена.')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    logger.info(f"Получена задача: '{user_query}'")
    await update.message.reply_text('Приступаю к выполнению задачи...')
    
    try:
        # ReAct агент сам управляет историей через объект memory
        result = agent_executor.invoke({"input": user_query})
        
        # --- ФОРМИРУЕМ ПОЛНЫЙ ОТЧЕТ ДЛЯ ОТЛАДКИ ---
        final_answer = result.get("output", "Агент не дал финального ответа.")
        intermediate_steps = result.get("intermediate_steps", [])
        
        debug_report = "📝 **Ход Рассуждений Агента** 📝\n\n"
        for action, observation in intermediate_steps:
            # action.log содержит полный блок "Thought:"
            debug_report += f"🤔 **Мысль:** {action.log.strip().split('Action:')[0].strip()}\n"
            debug_report += f"🛠️ **Действие:** `{action.tool}`\n"
            debug_report += f"📥 **Входные данные:** `{action.tool_input}`\n"
            debug_report += f"🔎 **Результат:** `{observation}`\n\n"
        
        debug_report += f"✅ **Финальный Ответ:**\n{final_answer}"

        # Отправляем полный отчет
        await update.message.reply_text(debug_report, parse_mode='Markdown')
        
        # Если финальный ответ - это создание документа, отправляем файл
        if final_answer.startswith("Документ") and ('.docx' in final_answer):
            try:
                file_path = final_answer.split(":")[-1].strip()
                if os.path.exists(file_path):
                    await context.bot.send_document(chat_id=update.effective_chat.id, document=open(file_path, 'rb'))
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Ошибка при отправке документа: {e}", exc_info=True)
                await update.message.reply_text(f"Не удалось отправить документ.")
        
        logger.info("Отправлен полный отчет.")
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла внутренняя ошибка: {e}")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Получено изображение.")
    await update.message.reply_text('Анализирую изображение...')
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        user_caption = update.message.caption or "Опиши это изображение подробно."
        
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
        
        message_payload = HumanMessage(content=[{"type": "text", "text": user_caption}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},])
        response = llm.invoke([message_payload])
        await update.message.reply_text(response.content)
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}", exc_info=True)
        await update.message.reply_text(f"Не удалось обработать изображение.")