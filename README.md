# Люба (@asluba_bot) — OpenClaw architecture

Живой собеседник в Telegram — общается в личке и активно в группах/чатах,
ставит реакции, комментирует новости с инфой из интернета. В каналах — только
реакции. Работает **в среде OpenClaw**, развёрнутого в GitHub Actions 24/7.

## ✨ Возможности (10 функций)

| Функция | Описание |
|---|---|
| 💬 Текст | Общение в личке и группах, память диалога и фактов о людях |
| 📷 Фото (Vision) | Понимает фото через Gemini/GPT-4o |
| 🎤 Голосовые | Транскрипция через Whisper + ответ |
| 😀 Стикеры | Реагирует на эмодзи, комментирует |
| 🎬 GIF/Видео/Кружочки | Реагирует и комментирует |
| 🔍 Новости | Развёрнуто дополняет новости инфой из интернета |
| 🗣 Proactive topics | Сам начинает беседу в тихих/активных группах |
| 📝 Память бесед | 30-мин суммаризация обсуждений |
| 🏷 Inline | `@asluba_bot <вопрос>` в любом чате |
| 🤝 Партнёры | Контекстные ссылки на sochiautoparts.ru + рекомендации каналов |

**Покрытие 100% типов сообщений** — ни одно сообщение не игнорируется.

## 🏗 Архитектура

OpenClaw Gateway (Node.js) запускается как subprocess в GitHub Actions,
отдаёт OpenAI-совместимый API на localhost:18789. Python-бот (aiogram)
обрабатывает Telegram и все AI-запросы направляет через OpenClaw.

## 🚀 Запуск

### GitHub Actions (24/7 бесплатно)
1. Секреты: `BOT_TOKEN`, `OWNER_ID`, `GH_PAT_TOKEN` (обязательные)
2. Опциональные AI ключи: `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, etc.
3. Бот работает на Pollinations free даже без ключей.

### Локально
```bash
pip install -r requirements.txt
npm install -g openclaw@latest
cp .env.example .env  # заполнить BOT_TOKEN
python -m bot.main
```

## ⚙️ Настройки @BotFather
1. **Group Privacy → OFF** (видеть все сообщения в группах)
2. **Inline Mode → ON** (`@asluba_bot <вопрос>` в любом чате)
3. В каналы добавлять как админа (для реакций)

## 🔄 Надёжность 24/7
- Cancel conflicting runs + 60s wait
- Unlimited auto-restart loop
- DB cache + git commit
- Re-dispatch с 5 ретраями
- Cron каждые 2 часа (fallback)

## 📄 Лицензия
MIT
