"""Минимальный агент с построчным JSON-протоколом для проверки изоляции в Docker."""
import json
import sys

reset = json.loads(sys.stdin.readline())
episode_id = reset["observation"]["manifest"]["episode_id"]
print(json.dumps({"action": {
    "action": "submit",
    "episode_id": episode_id,
    "findings": [],
}}, separators=(",", ":")), flush=True)

# Перед выходом читаем сообщение о завершении. В режиме eval
# диагностические метрики остаются на стороне оценщика.
sys.stdin.readline()
