# HAR Stage 06.1 — безопасная загрузка сохранённых checkpoints

## Почему 75/75 результатов PASS, а 0/3 checkpoints?

Исходные checkpoints Stage 5G.3 содержат не только веса `state_dict`, но и массивы NumPy `mean` и `std`, сохранённые через `torch.save`. PyTorch >=2.6 по умолчанию ограничивает `torch.load` с `weights_only=True` и блокирует `numpy._core.multiarray._reconstruct`.

Эта версия аудитора использует `weights_only=True` плюс **ограниченный временный список** исключительно из стандартных конструкторов NumPy, встречающихся в таких файлах. Она НЕ использует `weights_only=False` и не принимает неожиданные глобальные классы. Проверяется совпадение model/dataset/fold, shape `(3,)`, finite `mean`/`std`, строгая совместимость `state_dict` и размерность выхода модели.

## Установка поверх Stage 06

Распакуйте содержимое ZIP в существующую папку `cross-dataset-har-benchmark`, сохранив вложенные каталоги `tools/` и `tests/`. Обновляется только `tools/audit_original_supervised.py`, добавляются `tools/safe_numpy_checkpoint.py` и `tests/test_safe_numpy_checkpoint.py`.

Перед заменой для подстраховки сделайте резервную копию:

```bash
cp tools/audit_original_supervised.py tools/audit_original_supervised.py.bak
```

Проверка на искусственно созданных legacy-checkpoints:

```bash
PY=/path/to/python
"$PY" tests/test_safe_numpy_checkpoint.py
```

Повторный аудит **без обучения**:

```bash
PROJECT=/path/to/HAR_Q1_2026
"$PY" tools/audit_original_supervised.py \
  --project-root "$PROJECT" \
  --check-checkpoints sample \
  --output /path/to/stage06_1_supervised_audit.json
```

Ожидаемый успешный результат: `Metrics against Table 2: 75/75`, `Predictions recalculated: 75/75`, `Checkpoint compatibility: 3/3`, `AUDIT: PASS`.

Если повторно возникнет ошибка, пришлите точное сообщение: **не переходите к** `weights_only=False` и не добавляйте произвольные функции в `safe_globals`.

Не загружайте checkpoints, полученные из непроверенных источников.
