# Финансовый аудит · Genesis FinTech Hackaton

**Участники проекта** — Мансур, Самир, Дамир, Багнур.

Среда для обучения с подкреплением на синтетической бухгалтерии. Агент читает проводки, счета, платежи и валютные курсы, затем отправляет найденные расхождения со ссылками на доказательства.

## Запуск

Нужен Python 3.11. Для запуска достаточно стандартной библиотеки. Команды выполняются из корня репозитория.

```powershell
python -m fin_audit_rl.cli --check
python -m fin_audit_rl.cli --demo
```

`--check` запускает тесты и четыре базовые политики на 20 одинаковых эпизодах. Результаты записываются в `artifacts/baseline_results.json`. `--demo` показывает сверку счёта с двумя связанными ошибками.

Для проверки в Linux-контейнере заранее загрузите базовый образ.

```powershell
docker pull python:3.11-slim
docker build --network none -t fin-audit-rl .
docker run --rm --network none --cpus 2 --memory 4g fin-audit-rl python -m fin_audit_rl.cli --check
```

## Чекер жюри

Рабочая директория контейнера — `/app`. Чекер импортирует [judge_adapter.py](judge_adapter.py). Команда ниже проверяет импорт в собранном образе.

```powershell
docker run --rm --network none fin-audit-rl python -c "import judge_adapter"
```

- `generate_episode(seed, config, output_dir)` записывает комплект файлов одного эпизода. Публичные файлы — `manifest.json`, `policy.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl` и `fx.csv`. Файлы `truth.json` и `provenance.json` доступны жюри.
- `make_env()` создаёт среду. `reset(seed, config)` принимает `mode` (`closed` или `train`), `episode_dir` и `team_config`. При наличии каталога среда читает записи в файловом порядке и сохраняет их ID. Без каталога она генерирует эпизод по `seed` и `team_config`.
- `score_episode(episode_dir, findings, inspection_steps)` возвращает `tp`, `fp`, `fn`, `precision`, `recall`, `f1` и `return`. При невалидной отправке метрики равны `null`, а `return` равен `-1`.

После `reset` наблюдение содержит `task`, `manifest`, `policy` и `remaining_steps`. Метод `step` принимает `{"name": "…", "args": {...}}` и возвращает `(observation, reward, terminated, truncated, info)`. В наблюдении есть `result`, `error` и `remaining_steps`. Ошибка содержит `code` и `message`.

| Действие | Аргументы | Результат |
| --- | --- | --- |
| `list_records` | `table`, `cursor`, `limit` | До 50 записей в `records`. Начните с `cursor=null` и передавайте выданный строковый `next_cursor`, пока он не станет `null`. |
| `get_record` | `table`, `id` | Запись. |
| `get_document` | `doc_id` | Счёт или платёж. |
| `calculator` | `expression` | Десятичная строка. |
| `submit` | `findings` | `{"accepted": true}` при корректной структуре. |

Таблицы называются `ledger`, `invoices`, `payments` и `fx`. Объект отправки имеет вид `{"episode_id": "…", "findings": [...]}`. Если `episode_id` опущен, берётся ID текущего эпизода. Каждый вывод содержит `type`, `affected_ids`, целый `claimed_amount_minor`, `currency`, `evidence_ids` и `explanation`. Лимиты — 30 выводов, 10 доказательств на вывод и 1000 символов в пояснении.

| `type` | Расхождение |
| --- | --- |
| `duplicate_invoice` | Один счёт поставщика проведён дважды. |
| `payment_allocation` | Платёж распределён частично или пропущен. |
| `fx_date_rate` | Проводка использует неверный валютный курс. |
| `period_cutoff` | Счёт проведён в другом отчётном периоде. |

На эпизод отводится 100 действий и 60 секунд с начала `reset`. Отправка входит в лимит действий. В режиме `closed` агент получает нулевую награду на каждом шаге. Истинную награду читает процесс жюри. Агент работает отдельно и обменивается с оценщиком JSON-сообщениями. Для локальной проверки изоляции можно запустить `python -m fin_audit_rl.docker_smoke` при работающем Docker и загруженном `python:3.11-slim`.

## Данные для сдачи

```powershell
python -m fin_audit_rl.cli --out artifacts
```

| Путь | Содержимое |
| --- | --- |
| `artifacts/open/{train,dev}/` | По 10 публичных эпизодов в каждом наборе. В `episodes.json` сохранены версии, seed и конфигурации. |
| `artifacts/evaluator/{train,dev}/` | Эталоны и история изменений для локальной оценки. |
| `artifacts/trajectories.jsonl` | 20 траекторий на наборе `train`, сгруппированных по `trajectory_id`. |
| `artifacts/baseline_results.json` | Метрики, награда, шаги и время 80 запусков базовых политик. |

Генератор принимает `transactions` (20–400), `base_currency` (USD, EUR, GBP, KZT, JPY), `timezone` и набор ошибок через `issue_types`, `issue_count` (0–12) или `interaction_scenario` (`partial_fx`, `cutoff_fx`). Правила учёта, формат истории изменений и результаты сверки описаны в [DESIGN.md](DESIGN.md).
