"""
Пути к директориям и файлам проекта.
"""

import os

# Базовая директория проекта
BASE_DIR = r'C:\Users\Alexandra\Desktop\ner'

PATHS = {
    # Данные
    'raw_data': os.path.join(BASE_DIR, 'data', 'raw_data', 'train.jsonl'),
    'raw_data': os.path.join(BASE_DIR, 'data', 'raw_data', 'dev.jsonl'),
    'raw_data': os.path.join(BASE_DIR, 'data', 'raw_data', 'test.jsonl'),

    'raw_data': os.path.join(BASE_DIR, 'data', 'raw_data', 'ent_types.jsonl'),
    'raw_data': os.path.join(BASE_DIR, 'data', 'raw_data', 'rel_types.jsonl')
}
    