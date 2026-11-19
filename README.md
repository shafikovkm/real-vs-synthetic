# Real vs Synthetic — данные и протокол

Комплект участника 2. Запускайте команды из корня этой папки. Исходный файл и сплиты подготовлены из [официального Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/main). Исходник и манифест уже зафиксированы в `data/`; команды ниже нужны для воспроизведения в новой директории.

```bash
python3 split.py --input data/databricks-dolly-15k.jsonl --output-dir /tmp/dolly-reproduced
python3 -m pip install -r requirements.txt
python3 scripts/eda.py --input data/databricks-dolly-15k.jsonl --output-dir figures
```

Результаты: `data/{train_pool,val,test,train_real}.jsonl`, `data/split_manifest.json`, `figures/{categories,lengths}.pdf`, `figures/eda_summary.json`. Файлы под CC BY-SA 3.0; атрибуция и сведения о модификации — в DATA_CARD.md. Размер исходника превышает сумму сплитов: лишние записи не включаются в эксперимент. Не создавайте замороженные сплиты повторно после начала работы команды.

Локальная отладка на небольшом наборе (выходные файлы не использовать в эксперименте):

```bash
python3 split.py --input data/databricks-dolly-15k.jsonl --output-dir /tmp/dolly-smoke --limit 50
python3 scripts/eda.py --input data/databricks-dolly-15k.jsonl --output-dir /tmp/dolly-eda --limit 50
```

Когда появятся финальные `seeds.jsonl` и `data/train_synth.jsonl`, проверьте их независимо (seed-файл должен содержать полные оригинальные записи, включая ID):

```bash
python3 scripts/check_protocol.py --train-pool data/train_pool.jsonl --val data/val.jsonl --test data/test.jsonl --seeds seeds.jsonl --synth data/train_synth.jsonl --output results/leakage_audit.json
```

Проверяется совпадение всех полей seed с train_pool и максимум ROUGE-L F1 по инструкциям synthetic против val и test, без стемминга, `rouge-score==0.1.2`. Оценка >= 0.6 означает утечку; успешный результат требует ровно 2000 примеров и полный проход. Аудит необходимо согласовать с автором фильтра относительно токенизации, метрики и полей. Сокращённый прогон с `--limit 50` предназначен только для отладки и никогда не считается допуском. Отчёт и SHA-256 всех входов сохраняются в JSON. При коде возврата 1 допуск не выдан.

Откройте `EDA.ipynb` через `jupyter notebook EDA.ipynb`; ноутбук запускает тот же проверяемый CLI. Секция для общего отчёта находится в `report/sections/02_data_protocol.tex`; она намеренно не содержит неподтверждённых результатов.
