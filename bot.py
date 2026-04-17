"""
VK Community Bot с RAG-функциональностью на основе ProxyAPI.

Этот бот использует HTTP запросы вместо OpenAI SDK, что позволяет
работать с любыми OpenAI-совместимыми API endpoint'ами.
"""

import asyncio
import logging
import time
import re
from pathlib import Path
from typing import List
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id

from config import VK_API_TOKEN, VK_GROUP_ID, DOCS_PATH, LOG_LEVEL, LOG_FORMAT
from rag.pipeline import RAGPipeline

# ========== НАСТРОЙКА ЛОГИРОВАНИЯ ==========
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== ОЧИСТКА ТЕКСТА ОТ РАЗМЕТКИ ==========
def clean_html(text: str) -> str:
    """
    Жёстко удаляет HTML и Markdown-разметку, оставляя чистый текст.
    Используется перед отправкой любых сообщений в VK API.
    
    Args:
        text: Исходный текст с возможной разметкой
        
    Returns:
        Очищенный текст без тегов и символов форматирования
    """
    if not text:
        return ""
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)
    # Удаляем Markdown: **жирный**, *курсив*, __жирный__, _курсив_
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Убираем лишние переносы строк (более 2 подряд)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ========== ИНИЦИАЛИЗАЦИЯ БОТА ==========
vk_session = vk_api.VkApi(token=VK_API_TOKEN)
vk = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, VK_GROUP_ID)

# Инициализируем RAG-пайплайн
logger.info("Инициализация RAG Pipeline (ProxyAPI)...")
rag_pipeline = RAGPipeline()
logger.info("RAG Pipeline готов к работе")

# ========== ПАМЯТЬ РАЗГОВОРОВ ==========
# Словарь для хранения истории сообщений каждого пользователя
# Структура: {user_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
conversation_history = {}
MAX_HISTORY_LENGTH = 10  # Максимум последних сообщений для контекста


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """
    Разбивает текст на части заданного размера.
    
    Args:
        text: Текст для разбиения
        max_chars: Максимальный размер одной части в символах
        
    Returns:
        Список частей текста
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Разбиваем по параграфам (двойной перенос строки)
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph_length = len(paragraph) + 2  # +2 для \n\n
        
        # Если параграф сам больше лимита, разбиваем его по предложениям
        if paragraph_length > max_chars:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                sentence_length = len(sentence) + 2
                if current_length + sentence_length > max_chars and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        
        # Если добавление параграфа превысит лимит, сохраняем текущий chunk
        elif current_length + paragraph_length > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length
    
    # Добавляем последний chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def load_documents_from_directory(directory: Path) -> tuple[List[str], List[str]]:
    """
    Загружает все текстовые документы из директории.
    Большие документы разбиваются на части (chunks).
    
    Args:
        directory: Путь к директории с документами
        
    Returns:
        Кортеж (тексты документов, имена файлов)
    """
    documents = []
    sources = []
    
    if not directory.exists():
        logger.warning(f"Директория {directory} не существует")
        return documents, sources
    
    # Ищем все .txt файлы
    for file_path in directory.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
                # Разбиваем большие документы на части
                chunks = chunk_text(text, max_chars=6000)
                
                if len(chunks) > 1:
                    logger.info(f"Загружен документ: {file_path.name} ({len(text)} символов) - разбит на {len(chunks)} частей")
                    for i, chunk in enumerate(chunks, 1):
                        documents.append(chunk)
                        sources.append(f"{file_path.name} (часть {i}/{len(chunks)})")
                else:
                    logger.info(f"Загружен документ: {file_path.name} ({len(text)} символов)")
                    documents.append(text)
                    sources.append(file_path.name)
                    
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
    
    logger.info(f"Всего загружено документов/частей: {len(documents)}")
    return documents, sources


def send_long_message(peer_id: int, text: str, max_length: int = 4000):
    """
    Отправляет длинное сообщение, разбивая его на части если нужно.
    Автоматически очищает текст от HTML/Markdown перед отправкой.
    
    Args:
        peer_id: ID чата/пользователя для отправки
        text: Текст для отправки
        max_length: Максимальная длина одного сообщения
    """
    # 🔥 Очищаем текст от разметки до отправки
    clean_text = clean_html(text)
    
    if len(clean_text) <= max_length:
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=clean_text)
        return
    
    # Разбиваем на части
    parts = [clean_text[i:i+max_length] for i in range(0, len(clean_text), max_length)]
    for part in parts:
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=part)
        time.sleep(0.5)


# ========== ОБРАБОТЧИКИ КОМАНД ==========

def cmd_start(peer_id: int):
    """
    Обработчик команды /start - приветствие и информация о боте.
    """
    logger.info(f"Пользователь {peer_id} запустил бота")
    
    welcome_text = """
🤖 Добро пожаловать в RAG-бота (ProxyAPI)!
Я интеллектуальный ассистент с доступом к базе знаний.
Работаю через ProxyAPI - совместимый с OpenAI endpoint.

Мои возможности:
📚 Поиск информации в базе знаний (RAG)
🖼 Обработка изображений и извлечение текста
💬 Ответы на вопросы с использованием контекста

Доступные команды:
/start - Показать это сообщение
/help - Подробная справка
/ask <вопрос> - Задать вопрос с поиском в базе знаний
/ingest - Перезагрузить базу знаний (только для администраторов)
/stats - Показать статистику системы
/test - Проверить подключение к ProxyAPI

Как пользоваться:
• Просто напишите вопрос, и я найду ответ в базе знаний
• Отправьте изображение с текстом, и я его обработаю
• Используйте /ask для явного RAG-запроса

Powered by ProxyAPI + FAISS
"""
    # Очищаем текст перед отправкой (VK не парсит HTML)
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=clean_html(welcome_text))


def cmd_help(peer_id: int):
    """
    Обработчик команды /help - подробная справка.
    """
    logger.info(f"Пользователь {peer_id} запросил справку")
    
    help_text = """
📖 Подробная справка по использованию бота (ProxyAPI версия)

1️⃣ RAG-запросы (Retrieval-Augmented Generation)
RAG - это технология, которая позволяет мне находить релевантную информацию в базе знаний и использовать её для формирования ответа.

Примеры запросов:
• "Что такое RAG и как он работает?"
• "Как применяется RAG в поддержке клиентов?"
• "Расскажи о преимуществах RAG в HR"

2️⃣ Обработка изображений
Отправьте мне изображение (фото, скриншот, скан документа), и я:
• Извлеку текст с изображения
• Могу ответить на вопросы по этому тексту
• Найду связанную информацию в базе знаний

Пример: отправьте фото документа с подписью "Что это значит?"

3️⃣ Команды
/ask <вопрос> - Явный RAG-запрос
Пример: /ask Как работает векторный поиск?

/ingest - Перезагрузка базы знаний
Эта команда переиндексирует все документы.

/stats - Статистика системы
Показывает информацию о системе и API endpoint.

/test - Проверка подключения
Проверяет доступность ProxyAPI.

4️⃣ Технические детали
🔹 API: ProxyAPI (OpenAI-совместимый)
🔹 Модель чата: GPT-4o-mini
🔹 Модель vision: GPT-4o-mini
🔹 Эмбеддинги: text-embedding-3-small
🔹 Векторная БД: FAISS
🔹 HTTP клиент: requests

Преимущества ProxyAPI версии:
✅ Работа с любыми OpenAI-совместимыми API
✅ Полный контроль над запросами
✅ Легкая отладка и мониторинг
✅ Возможность использования локальных моделей

Примеры эффективного использования:
✅ "Объясни как RAG помогает в HR-процессах"
✅ "Какие метрики эффективности RAG в поддержке?"
✅ "В чем преимущества использования RAG?"

Если у вас есть вопросы - просто задайте их! 😊
"""
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=clean_html(help_text))


def cmd_ask(peer_id: int, query: str):
    """
    Обработчик команды /ask - RAG-запрос.
    """
    if not query:
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="❌ Пожалуйста, укажите вопрос после команды.\nПример: /ask Что такое RAG?")
        return
    
    logger.info(f"RAG-запрос от пользователя {peer_id}: {query}")
    
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="🔍 Ищу информацию в базе знаний...")
    
    try:
        result = rag_pipeline.query(query)
        
        # Формируем ответ БЕЗ источников
        response_text = f"💡 Ответ:\n{result.get('answer', 'Нет ответа')}"
        
        send_long_message(peer_id, response_text)
        
        logger.info(f"Ответ отправлен пользователю {peer_id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Произошла ошибка: {str(e)}")


def cmd_ingest(peer_id: int):
    """
    Обработчик команды /ingest - индексация документов.
    """
    logger.info(f"Запрос на индексацию от пользователя {peer_id}")
    
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="📥 Начинаю индексацию документов через ProxyAPI...")
    
    try:
        documents, sources = load_documents_from_directory(DOCS_PATH)
        
        if not documents:
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Документы не найдены в директории {DOCS_PATH}\nПожалуйста, добавьте .txt файлы с документами.")
            return
        
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"📄 Найдено документов: {len(documents)}\nНачинаю обработку...")
        
        success = rag_pipeline.index_documents(documents, sources)
        
        if success:
            stats = rag_pipeline.get_stats()
            response = (
                "✅ Индексация завершена успешно!\n\n"
                f"📊 Статистика:\n"
                f"• Документов: {stats['total_documents']}\n"
                f"• Векторов: {stats['total_vectors']}\n"
                f"• Размерность: {stats['dimension']}\n"
                f"• API: {stats['api_url']}\n\n"
                f"Бот готов к работе! Задавайте вопросы. 💬"
            )
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=clean_html(response))
            logger.info("Индексация завершена успешно")
        else:
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="❌ Ошибка при индексации документов. См. логи для деталей.")
            
    except Exception as e:
        logger.error(f"Ошибка при индексации: {e}")
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Произошла ошибка при индексации: {str(e)}")


def cmd_stats(peer_id: int):
    """
    Обработчик команды /stats - статистика системы.
    """
    logger.info(f"Запрос статистики от пользователя {peer_id}")
    
    stats = rag_pipeline.get_stats()
    
    status_emoji = "✅" if stats['is_loaded'] else "❌"
    status_text = "Загружена" if stats['is_loaded'] else "Не загружена"
    
    # Статистика истории для пользователя
    history_count = len(conversation_history.get(peer_id, [])) // 2
    
    stats_text = f"""
📊 Статистика RAG-системы (ProxyAPI)

Состояние базы знаний:
{status_emoji} {status_text}

Данные:
• Документов: {stats['total_documents']}
• Векторов: {stats['total_vectors']}
• Размерность: {stats['dimension']}

Модели:
• Чат: {stats['chat_model']}
• Vision: {stats['vision_model']}
• Эмбеддинги: {stats['embed_model']}

API:
• URL: {stats['api_url']}

Файлы:
• Индекс: {'✅' if stats['index_exists'] else '❌'}
• Метаданные: {'✅' if stats['metadata_exists'] else '❌'}

Ваш контекст:
• Сообщений в истории: {history_count}
"""
    
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=clean_html(stats_text))


def cmd_clear(peer_id: int):
    """
    Обработчик команды /clear - очищает историю разговора пользователя.
    """
    logger.info(f"Очистка истории для пользователя {peer_id}")
    
    if peer_id in conversation_history:
        messages_count = len(conversation_history[peer_id]) // 2
        conversation_history[peer_id] = []
        response = f"🧹 История очищена!\n\nУдалено {messages_count} сообщений из контекста.\nНачинаем разговор с чистого листа!"
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=clean_html(response))
    else:
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="ℹ️ История пустая.\n\nУ вас еще нет сохраненных сообщений в контексте.")


def cmd_test(peer_id: int):
    """
    Обработчик команды /test - проверка подключения к ProxyAPI.
    """
    logger.info(f"Проверка подключения от пользователя {peer_id}")
    
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="🔄 Проверяю подключение к ProxyAPI...")
    
    try:
        success = rag_pipeline.test_connection()
        
        if success:
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="✅ Подключение к ProxyAPI работает корректно!")
        else:
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="❌ Проблемы с подключением к ProxyAPI. См. логи.")
            
    except Exception as e:
        logger.error(f"Ошибка при проверке подключения: {e}")
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Ошибка: {str(e)}")


# ========== ОБРАБОТЧИКИ СООБЩЕНИЙ ==========

def handle_photo(peer_id: int, attachments: list, message_text: str = ""):
    """
    Обработчик фотографий - извлекает текст с изображения.
    """
    logger.info(f"Получено изображение от пользователя {peer_id}")
    
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="🖼 Обрабатываю изображение через ProxyAPI...")
    
    try:
        # Извлекаем URL фото из вложений
        photo_url = None
        for attachment in attachments:
            if attachment.get('type') == 'photo':
                # Получаем URL фото с максимальным качеством
                photo = attachment['photo']
                sizes = photo.get('sizes', [])
                if sizes:
                    # Сортируем по размеру и берем последний (максимальный)
                    sizes.sort(key=lambda x: x.get('width', 0) * x.get('height', 0), reverse=True)
                    photo_url = sizes[0].get('url')
                    break
        
        if not photo_url:
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="❌ Не удалось получить URL изображения")
            return
        
        user_query = message_text if message_text else None
        
        result = rag_pipeline.process_image(photo_url, user_query)
        
        if result.get('error'):
            vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Ошибка при обработке изображения: {result['error']}")
            return
        
        # Формируем ответ БЕЗ источников, сразу очищая от разметки
        response_text = "📄 Текст с изображения:\n"
        response_text += clean_html(result.get('extracted_text', '')) + "\n\n"
        
        if result.get('rag_answer'):
            response_text += f"💡 Ответ на ваш вопрос:\n{clean_html(result['rag_answer'])}"
        
        send_long_message(peer_id, response_text)
        
        logger.info(f"Изображение обработано для пользователя {peer_id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Произошла ошибка: {str(e)}")


def handle_text(peer_id: int, message_text: str):
    """
    Обработчик текстовых сообщений - выполняет RAG-запрос с учетом истории.
    """
    query = message_text
    
    if query.startswith('/'):
        return
    
    logger.info(f"Текстовый запрос от пользователя {peer_id}: {query}")
    
    vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="🔍 Ищу информацию...")
    
    try:
        # Инициализируем историю для нового пользователя
        if peer_id not in conversation_history:
            conversation_history[peer_id] = []
        
        # Выполняем RAG-запрос с историей
        result = rag_pipeline.query_with_history(query, conversation_history[peer_id])
        
        # Сохраняем в историю ЧИСТЫЕ вопросы и ответы (без RAG промптов)
        conversation_history[peer_id].append({"role": "user", "content": query})
        conversation_history[peer_id].append({"role": "assistant", "content": result['answer']})
        
        # Ограничиваем размер истории
        if len(conversation_history[peer_id]) > MAX_HISTORY_LENGTH * 2:
            conversation_history[peer_id] = conversation_history[peer_id][-(MAX_HISTORY_LENGTH * 2):]
            logger.info(f"История обрезана до {MAX_HISTORY_LENGTH} пар сообщений")
        
        # Формируем ответ БЕЗ источников и очищаем от разметки
        response_text = clean_html(result.get('answer', ''))
        
        send_long_message(peer_id, response_text)
        
        logger.info(f"Ответ отправлен пользователю {peer_id} (история: {len(conversation_history[peer_id])//2} сообщений)")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message=f"❌ Произошла ошибка: {str(e)}")


# ========== ЗАПУСК БОТА ==========

def main():
    """
    Главная функция запуска бота.
    """
    logger.info("=" * 50)
    logger.info("Запуск VK Community Bot с RAG (ProxyAPI)")
    logger.info("=" * 50)
    
    stats = rag_pipeline.get_stats()
    logger.info(f"API URL: {stats['api_url']}")
    
    if stats['is_loaded']:
        logger.info(f"✅ База знаний загружена: {stats['total_documents']} документов")
    else:
        logger.warning("⚠️ База знаний не загружена. Выполните команду /ingest")
    
    logger.info("Бот запущен и готов к работе!")
    
    # Основной цикл обработки событий
    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:
            message = event.obj.message
            peer_id = message['peer_id']
            user_id = message['from_id']
            message_text = message.get('text', '')
            attachments = message.get('attachments', [])
            
            logger.info(f"Получено сообщение от пользователя {user_id}: {message_text}")
            
            # Обрабатываем команды
            if message_text.startswith('/start'):
                cmd_start(peer_id)
            elif message_text.startswith('/help'):
                cmd_help(peer_id)
            elif message_text.startswith('/ask'):
                query = message_text[5:].strip()
                cmd_ask(peer_id, query)
            elif message_text.startswith('/ingest'):
                cmd_ingest(peer_id)
            elif message_text.startswith('/stats'):
                cmd_stats(peer_id)
            elif message_text.startswith('/clear'):
                cmd_clear(peer_id)
            elif message_text.startswith('/test'):
                cmd_test(peer_id)
            elif message_text.startswith('/'):
                # Неизвестная команда
                vk.messages.send(peer_id=peer_id, random_id=get_random_id(), message="❌ Неизвестная команда. Используйте /help для получения списка команд.")
            elif attachments and any(attachment.get('type') == 'photo' for attachment in attachments):
                # Обработка фото
                handle_photo(peer_id, attachments, message_text)
            else:
                # Обработка обычного текста
                handle_text(peer_id, message_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")