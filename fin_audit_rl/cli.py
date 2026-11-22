"""Генерация открытых наборов данных, запись обучающих траекторий и проверка среды."""
from __future__ import annotations

import argparse
import json
import re
import time
import unittest
from pathlib import Path

from .env import AuditEnv
from .episode_io import write_episode, write_json, write_jsonl
from .generator import GENERATOR_VERSION, ISSUE_TYPES, generate_episode
from .policies import mark_everything, oracle, random_explorer, simple_reconciler

POLICIES = {"random": random_explorer, "reconcile": simple_reconciler,
            "mark-all": mark_everything, "oracle": oracle}
_CURRENCIES = ("USD", "EUR", "GBP", "KZT", "JPY")
_ZONES = ("America/New_York", "Europe/Paris", "Asia/Almaty", "Asia/Tokyo", "UTC")
_EPISODE_ID = re.compile(r"ep-[0-9a-f]{16}(?:[0-9a-f]{8})?")
_MARKER = ".generated-by-fin-audit-rl"


def open_cases(split):
    """Воспроизводимый открытый набор со всеми типами ошибок и обоими сценариями взаимодействия."""
    if split not in ("train", "dev"):
        raise ValueError("неизвестное имя открытого набора")
    dev = split == "dev"
    compositions = [[], [], *([kind] for kind in ISSUE_TYPES), "partial_fx", "cutoff_fx",
                    list(ISSUE_TYPES[:2] if dev else ISSUE_TYPES[2:]),
                    list(ISSUE_TYPES) + ([ISSUE_TYPES[1]] if dev else [])]
    for i, composition in enumerate(compositions):
        config = {"transactions": 65 + (i * (13 if dev else 7)) % 29,
                  "base_currency": _CURRENCIES[(i + 2 * dev) % len(_CURRENCIES)],
                  "timezone": _ZONES[(i * (3 if dev else 2) + dev) % len(_ZONES)]}
        if isinstance(composition, str):
            config["interaction_scenario"] = composition
        else:
            config["issue_types"] = composition
        yield (2000 if dev else 1000) + i, config


def benchmark_cases():
    """Набор для локальной диагностики. Закрытые случаи задаются организаторами."""
    for i in range(20):
        config = {"transactions": (65, 90, 120, 200, 400)[i % 5],
                  "base_currency": _CURRENCIES[(i * 3 + 1) % 5],
                  "timezone": _ZONES[(i * 2 + 2) % 5]}
        if i < 5:
            config["issue_count"] = 0
        elif i < 10:
            config["interaction_scenario"] = "partial_fx" if i % 2 else "cutoff_fx"
        else:
            config["issue_types"] = list(ISSUE_TYPES[i % 4:] + ISSUE_TYPES[:i % 4])[:1 + i % 4]
            if i == 19:
                config["issue_types"] *= 3
        yield 3000 + i, config


def _remove_stale(out, split, expected):
    public = out / "open" / split
    for folder in public.iterdir():
        if (folder.name in expected or not _EPISODE_ID.fullmatch(folder.name)
                or not (folder / "manifest.json").is_file()):
            continue
        private = out / "evaluator" / split / folder.name
        for target, files in ((folder, ("manifest.json", "policy.json", "ledger.csv",
                                        "fx.csv", "invoices.jsonl", "payments.jsonl")),
                              (private, ("truth.json", "provenance.json"))):
            if not target.is_dir() or out.resolve() not in target.resolve().parents:
                continue
            for name in files:
                (target / name).unlink(missing_ok=True)
            if not any(target.iterdir()):
                target.rmdir()


def build_open(out):
    marker = out / _MARKER
    for split in ("train", "dev"):
        index = []
        for seed, config in open_cases(split):
            ep = generate_episode(seed, config)
            episode_id = ep["manifest"]["episode_id"]
            index.append({"episode_id": episode_id, "seed": seed, "config": config})
            folder = out / "open" / split / episode_id
            private = out / "evaluator" / split / episode_id
            write_episode(ep, folder, private)
        if marker.is_file():
            _remove_stale(out, split, {row["episode_id"] for row in index})
        write_json(out / "open" / split / "episodes.json",
                   {"generator_version": GENERATOR_VERSION, "episodes": index})
    marker.write_text("Создано модулем fin_audit_rl.cli. Файлы можно сгенерировать повторно.\n", encoding="utf-8")


def record_training(out):
    env = AuditEnv(mode="train")
    with (out / "trajectories.jsonl").open("w", encoding="utf-8") as file:
        for seed, config in open_cases("train"):
            for name in ("reconcile", "random"):
                observation, info = env.reset(seed, config)
                trajectory_id = f"{info['episode_id']}:{name}"
                initial = {"episode_id": info["episode_id"], "step": 0, "action": None,
                           "observation": observation, "reward": 0.0, "terminated": False,
                           "truncated": False}
                POLICIES[name](env, seed)
                write_jsonl(file, ({**row, "trajectory_id": trajectory_id, "policy": name}
                                   for row in (initial, *env.trajectory)))


def run_bench(out):
    out.mkdir(parents=True, exist_ok=True)
    env = AuditEnv(mode="eval")
    results = []
    for seed, config in benchmark_cases():
        for policy_name, policy_fn in POLICIES.items():
            _, info = env.reset(seed, config)
            started = time.perf_counter()
            policy_fn(env, seed)
            elapsed = time.perf_counter() - started
            results.append({"policy": policy_name, "seed": seed, "config": config, **info,
                            "generator_version": GENERATOR_VERSION, "metrics": env.evaluator_result(),
                            "reward": env.evaluator_reward, "steps": env.steps,
                            "elapsed_seconds": elapsed, "terminated": env.terminated,
                            "truncated": env.truncated})
    write_json(out / "baseline_results.json", results)
    return results


def demo():
    env = AuditEnv(mode="train")
    observation, _ = env.reset(2468, {"transactions": 65, "interaction_scenario": "cutoff_fx"})
    simple_reconciler(env)
    first, last = env.trajectory[0], env.trajectory[-1]
    print(json.dumps({"reset": observation, "first_action": first["action"],
                      "sample_record": first["observation"]["records"][0],
                      "submission": last["action"], "metrics": env.evaluator_result(),
                      "return": env.evaluator_reward, "steps": env.steps}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="показать справку и завершить работу")
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="каталог результатов, по умолчанию artifacts")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="запустить тесты проекта и локальное сравнение базовых политик")
    modes.add_argument("--demo", action="store_true", help="показать аудит с выявлением двух взаимодействующих ошибок")
    args = parser.parse_args()
    if args.demo:
        demo()
        return
    if args.check:
        test_dir = Path(__file__).resolve().parent / "tests"
        if not test_dir.is_dir():
            parser.error("для --check нужен исходный каталог проекта с включёнными тестами")
        suite = unittest.defaultTestLoader.discover(str(test_dir))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    else:
        build_open(args.out)
        record_training(args.out)
    run_bench(args.out)


if __name__ == "__main__":
    main()
