"""Сквозная проверка изоляции агента в Docker на локальном компьютере."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .env import AuditEnv
from .runner import run_isolated_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="показать справку и завершить работу")
    parser.add_argument("--image", default="python:3.11-slim", help="имя локального образа агента")
    args = parser.parse_args()
    if not shutil.which("docker"):
        raise SystemExit("Команда Docker не найдена. Установите и запустите Docker Desktop, откройте PowerShell заново и повторите проверку.")
    info = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    if info.returncode:
        raise SystemExit("Служба Docker недоступна:\n" + info.stderr.strip())
    image = subprocess.run(["docker", "image", "inspect", args.image],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if image.returncode:
        raise SystemExit(f"Образ {args.image!r} отсутствует локально. Сначала скачайте его командой docker pull {args.image}")

    solution = Path(__file__).resolve().parent.parent / "examples" / "docker_agent"
    result = run_isolated_episode(
        AuditEnv(mode="eval"),
        ["python", "-u", "agent.py"],
        seed=2468,
        config={"transactions": 20, "issue_count": 0},
        backend="docker",
        image=args.image,
        solution_dir=solution,
    )
    metrics = result["evaluator_metrics"]
    print(f"Статус запуска — {result['status']}, F1 — {metrics['f1'] if metrics else 'нет оценки'}, "
          f"награда — {result['reward']}, число шагов — {len(result['trajectory'])}")
    if result["status"] != "completed" or metrics is None or metrics["f1"] != 1.0:
        raise SystemExit("Проверка изоляции агента в Docker завершилась ошибкой")
    print("Проверка изоляции в Docker пройдена. Использован чистый эпизод, оценка доступна оценщику.")


if __name__ == "__main__":
    main()
