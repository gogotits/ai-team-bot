# --- Диагностический скрипт для проверки соединения с Tavily ---
import os
from dotenv import load_dotenv

# Пытаемся импортировать TavilyClient. Если не получится, будет ошибка.
try:
    from tavily import TavilyClient
except ImportError:
    print("ОШИБКА: Библиотека 'tavily-python' не установлена. Выполните 'pip install tavily-python'")
    exit()

# Загружаем переменные окружения (важно для локального теста)
load_dotenv()

print("--- НАЧАЛО ДИАГНОСТИКИ TAVILY API ---")

# 1. Пытаемся прочитать ключ из переменных окружения сервера
api_key = os.environ.get("TAVILY_API_KEY")

if not api_key:
    print("\n[ РЕЗУЛЬТАТ ]")
    print("❌ ПРОВАЛ: Переменная окружения TAVILY_API_KEY не найдена на сервере.")
    print("-> Убедитесь, что вы правильно ввели имя и значение ключа во вкладке 'Environment' на Render.")
else:
    print("✅ УСПЕХ: Переменная TAVILY_API_KEY найдена на сервере.")
    
    # 2. Пытаемся использовать ключ для реального запроса
    print("\nПопытка соединения с Tavily...")
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query="какая погода в москве")
        
        print("\n[ РЕЗУЛЬТАТ ]")
        print("✅✅✅ ПОЛНЫЙ УСПЕХ! Соединение с Tavily работает.")
        print("-> Ответ от API получен:")
        print(response)

    except Exception as e:
        print("\n[ РЕЗУЛЬТАТ ]")
        print(f"❌ ПРОВАЛ: Произошла ошибка при соединении с Tavily.")
        print(f"-> Текст ошибки: {e}")
        print("-> Это обычно означает, что сам ключ неверный или заблокирован.")

print("\n--- КОНЕЦ ДИАГНОСТИКИ ---")