# CrossHAR — повторение извлечения признаков из исходных окон

**Результат автора от 2026-11-08: `NUMERICAL_MATCH_ALL_25_FOLDS`.** Все 25 folds, 96 776 test windows, 836 144 640 значений `Z_test` совпали точно; максимальная абсолютная ошибка 0.0. Исходный отчёт не включён, его SHA256 указан в `evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json`.

Для воспроизведения требуется **авторское доверенное окружение** с оригинальным каталогом `external_models/CrossHAR`, исходными `X.npy`, `normalization.json`, `representations.npz` и `encoder_best.pt` для каждого fold. Мы не распространяем эти артефакты в публичном архиве.

- Исторический код: Stage 5L.5A/B, IPython session 196, lines 19–22.
- Encoder: `MaskedModel4Pretrain`, 3 входных канала, hidden=72, FF=144, 1 layer, 4 heads, seq_len=120, emb_norm=True.
- Извлечение: batch 512, `output_embed=True`, `shuffle=False`, `eval()`, `inference_mode()`, float32 train-only normalization, без BF16 autocast.
- Сравнение `[N_test,120,72]` с оригинальным `Z_test`: допуски `atol=1e-5`, `rtol=1e-4`, но наблюдалось **точное совпадение** для всех 25 folds.
- Проверка weights проводится через `torch.load(..., weights_only=True)` только для доверенных собственных checkpoints.

## Команды на сервере автора

```bash
PY=/path/to/author/venv/bin/python
PROJECT=/path/to/original/HAR_Q1_2026
RELEASE=/path/to/unzipped/cross-dataset-har-benchmark
REPORTS=/path/to/local/reports

# Важный исторический import: original CrossHAR/models.py imports `config` by name.
export PYTHONPATH="$PROJECT/external_models/CrossHAR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import config, models; print("CrossHAR imports: PASS")'
"$PY" "$RELEASE/tests/test_crosshar_encoder_replay_synthetic.py"
CUDA_VISIBLE_DEVICES=1 "$PY" "$RELEASE/tools/replay_crosshar_encoder.py" \
  --project-root "$PROJECT" \
  --preflight "$REPORTS/HAR_STAGE13K_CROSSHAR_PREFLIGHT_ALL25.json" \
  --output "$REPORTS/HAR_CROSSHAR_ENCODER_ALL25.json"
```

Если исходный пакет CrossHAR отсутствует, добавление чужого кода в публичный репозиторий без проверки лицензии недопустимо. Исходные веса и файлы большого размера в репозиторий не включены.

**Ограничение:** Сопоставлены `X.npy` → уже обученный encoder → сохранённые `Z_test`. Ни повторное SSL-обучение encoder, ни повторное обучение головы, ни независимая установка на чистую машину не выполнялись.
