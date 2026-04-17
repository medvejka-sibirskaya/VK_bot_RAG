# VK Community Bot с RAG-функциональностью

VK-бот с интеллектуальной поддержкой на основе RAG (Retrieval-Augmented Generation) и ProxyAPI.

## Особенности

- **RAG-функциональность**: Поиск информации в базе знаний и генерация ответов с использованием контекста
- **Поддержка изображений**: Извлечение текста с изображений и ответы на вопросы по ним
- **История разговоров**: Сохранение контекста диалога для более точных ответов
- **ProxyAPI интеграция**: Работа с любыми OpenAI-совместимыми API endpoint'ами
- **FAISS векторная база данных**: Быстрый поиск по векторным представлениям

## Команды бота

- `/start` - Приветствие и информация о боте
- `/help` - Подробная справка
- `/ask <вопрос>` - Явный RAG-запрос
- `/ingest` - Перезагрузить базу знаний
- `/stats` - Показать статистику системы
- `/clear` - Очистить историю разговора
- `/test` - Проверить подключение к ProxyAPI

## Обработка изображений

Бот может обрабатывать изображения и извлекать текст с них:
1. Отправьте изображение в чат
2. Бот извлечет текст с изображения
3. При необходимости можно задать вопрос по изображению

## Требования

- Python 3.11+
- VK Community Token
- ProxyAPI endpoint
- FAISS векторная база данных

## Установка

1. Склонируйте репозиторий
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Создайте файл `.env` с настройками (см. `.env.example`)
4. Запустите бота:
   ```bash
   python bot.py
   ```

## Настройка

Создайте файл `.env` в корне проекта с переменными окружения:

```env
# VK Bot Configuration
VK_API_TOKEN=your_vk_group_token
VK_GROUP_ID=your_vk_group_id

# ProxyAPI Configuration
PROXY_API_URL=https://api.proxyapi.ru/openai/v1
PROXY_API_KEY=your_proxy_api_key

# FAISS Configuration
FAISS_INDEX_PATH=index.faiss
FAISS_METADATA_PATH=metadata.json

# Document Directory = сложите в папку базу знаний.
DOCS_PATH=data/docs

# RAG Configuration
TOP_K_RESULTS=3
MAX_CONTEXT_LENGTH=3000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Структура проекта

```
bot_vk/
├── bot.py              # Основной файл бота
├── config.py           # Конфигурация
├── .env.example        # Пример файла конфигурации
├── requirements.txt    # Зависимости
├── data/               # Документы для индексации
│   └── docs/
├── rag/                # RAG компоненты 
│   ├── __init__.py
│   ├── embedder.py
│   ├── pipeline.py
│   ├── retriever.py
│   └── vectorstore.py
└── README.md
```

## Разработка

Для разработки используйте:
- `bot.py` - основной файл бота
- `rag/` - все RAG-компоненты остаются без изменений
- `data/docs/` - директория с документами для индексации

## Лицензия

MIT
