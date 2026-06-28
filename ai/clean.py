"""
Очистка ответов AI — единая точка защиты от утечек промтов.

Модели (особенно маленькие и free) иногда эхируют фрагменты system/user
промта в ответ. Здесь вырезаются:
  - <think> теги и stray ChatML токены
  - Префиксы имён (Люба:, Assistant:, Собеседник:) — в начале и в строках
  - Markdown (жирный, курсив, заголовки) и HTML теги
  - Эхо инструкций/контекста (ДЛИНА:, ГДЕ:, КТО ПИШЕТ:, РЕКОМЕНДАЦИИ, 🔴 …)
  - Жадное поглощение буллетов после leak-заголовков

Легитимные ответы (включая списки и «сейчас» без «Москвы») сохраняются.
"""

import re


# Паттерны строк, которые модель НИКОГДА не должна писать в ответе.
# Это метки из промта/контекста — их вырезаем целиком со следующим буллетами.
_LEAK_LINE_PATTERNS = [
    # Длина-подсказки (из _adaptive_length_hint)
    r'^ДЛИНА\s*[:：]',
    # Время/контекст (из get_time_context)
    r'^Сейчас\s.+\sпо\sМоскве',
    r'^День\s+недели\s*[:：]',
    r'^Время\s+суток\s*[:：]',
    r'^Сезон\s*[:：]',
    r'^Настроение\s*[:：]',
    r'^Текущее\s+настроение',
    # Контекстные метки (из build_group_context / build_private_context)
    r'^ГДЕ\s*[:：]',
    r'^КТО\s+ПИШЕТ\s*[:：]',
    r'^НА\s+ЧТО\s+ОТВЕЧАЮТ',
    r'^ПРОТЕЖКА\s*[:：]',
    r'^ОБРАЩЕНИЕ\s*[:：]',
    r'^НЕДАВНИЕ\s+СООБЩЕНИЯ',
    r'^ЧТО\s+ТЫ\s+ПОМНИШЬ',
    r'^ЗАДАЧА\s*[:：]',
    r'^АВТОР\s+ПОСТА\s*[:：]',
    # Блоки рекомендаций/партнёров (из handlers)
    r'^РЕКОМЕНДАЦИИ',
    r'^ПАРТН[ЁЕ]РСКИЕ\s+ССЫЛКИ',
    r'^ТОВАР\s+ИЗ\s+МАГАЗИНА',
    r'^СВЕЖИЙ\s+ПОСТ',
    r'^РЕЗУЛЬТАТЫ\s+ВЕБ[-\s]ПОИСКА',
    # Секции персоны (из старого промта — на случай если модель видит кэш)
    r'^КЛЮЧЕВЫЕ\s+ЗНАНИЯ',
    r'^О\s+ТЕБЕ\b',
    r'^ЧЕЛОВЕЧЕСКИЕ\s+РЕАКЦИИ',
    r'^СТИЛЬ\s*[:：]',
    r'^ЗАПРЕТ\s*[:：]',
    r'^ССЫЛКИ\s+И\s+ПАРТН[ЁЕ]РЫ',
    r'^ВРЕМЯ\s*[:：]',
    r'^ПАМЯТЬ\s+И\s+КОНТЕКСТ',
    r'^ФОТО\s+И\s+ИЗОБРАЖЕНИЯ',
    # Эхо групповых инструкций (groups.py добавляет их в user msg)
    r'^В\s+группе\s+поделились',
    r'^Вступ[аиій]\s+в\s+беседу',
    r'^Тебе\s+пишут\s+напрямую',
    r'^Отреагируй\s+живо',
    r'^Ответь\s+живо',
    r'^Обратись?\s+к\s+автору',
    r'^Обратись?\s+по\s+имени',
    r'^Ответь\s+участнику',
    r'^Поделись\s+своим\s+мнением',
    r'^Задай\s+вопрос',
    r'^Прокомментируй\s+это\s+сообщение',
    # Маркеры секций персоны
    r'^🔴',
    r'^Ты\s+Люба\b',
    r'^Люба\s+—',
]
_LEAK_RE = re.compile('|'.join(_LEAK_LINE_PATTERNS), re.IGNORECASE)

# Inline role-метки в начале строк
_ROLE_RE = re.compile(r'^(Собеседник|User|Люба|Lyuba|Assistant|Model)\s*[:：]\s*')

# Leading instruction phrases (если модель вынесла инструкцию в начало ответа)
_LEADING_PHRASES = (
    "Вступи в беседу —", "Вступай в беседу —",
    "Отреагируй живо —", "Тебе пишут напрямую.",
    "В группе поделились событием/новостью.",
)


def clean_response(text: str) -> str:
    """Очистить ответ AI от утечек промта и форматирования."""
    if not text:
        return ""

    # <think> теги
    text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think[^>]*>', '', text, flags=re.IGNORECASE)

    # Stray ChatML токены (локальная модель может их протекать)
    for tok in ("<|im_end|>", "<|im_start|>"):
        text = text.replace(tok, "")

    # Префиксы имён в начале ответа
    for prefix in ("Люба:", "Lyuba:", "ЛЮБА:", "Assistant:", "Ответ:",
                   "Ассистент:", "Собеседник:", "User:", "Model:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Обёртывающие кавычки (прямые + ёлочки)
    if len(text) > 2 and text[0] == text[-1] and text[0] in ('"', '"', '"', "'", "«", "»"):
        text = text[1:-1]

    # Markdown
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # HTML теги
    text = re.sub(r'<[^>]+>', '', text)

    # ── Построчная защита от утечек ──
    lines = text.split("\n")
    kept = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped and _LEAK_RE.match(stripped):
            # Дропаем эту строку И следующие за ней буллеты (части того же блока)
            i += 1
            while i < len(lines):
                cont = lines[i].strip()
                if not cont:
                    break
                if cont.startswith(("- ", "* ", "• ", "— ", "• ", "+ ")):
                    i += 1
                    continue
                break
            continue
        # Убираем inline role-метки в начале строк
        line = _ROLE_RE.sub('', lines[i])
        kept.append(line)
        i += 1
    text = "\n".join(kept)

    # Leading instruction phrases
    for ph in _LEADING_PHRASES:
        if text.startswith(ph):
            text = text[len(ph):].lstrip(" —.—")

    # Финальная очистка whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def contains_non_cyrillic_script(text: str) -> bool:
    """Детект CJK/Arabic/Devanagari галлюцинаций (Qwen иногда выдаёт китайский).

    Возвращает True если >5% букв — CJK/арабский/деванагари/хангыль.
    Используется для отбраковки ответов HF Qwen2.5.
    """
    if not text:
        return False
    non_cyrillic = 0
    total_letters = 0
    for ch in text:
        if ch.isalpha():
            total_letters += 1
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or  # CJK
                0x3040 <= cp <= 0x30FF or  # Hiragana/Katakana
                0x0600 <= cp <= 0x06FF or  # Arabic
                0xAC00 <= cp <= 0xD7AF):   # Hangul
                non_cyrillic += 1
    if total_letters == 0:
        return False
    return (non_cyrillic / total_letters) > 0.05
