"""Точки входа чекера Genesis FinTech для процесса жюри."""

import logging
from pathlib import Path

from fin_audit_rl.env import AuditEnv, score_episode_findings
from fin_audit_rl.episode_io import read_episode, write_episode
from fin_audit_rl.generator import generate_episode as _generate_episode
from fin_audit_rl.matcher import terminal_reward


_LOGGER = logging.getLogger(__name__)
_METRIC_NAMES = ("tp", "fp", "fn", "precision", "recall", "f1")
_ERROR_CODES = {
    "неизвестное действие": "unknown_action",
    "недопустимое действие": "invalid_arguments",
    "запись не найдена": "not_found",
    "неверная структура отправленных выводов": "invalid_submission",
    "эпизод уже завершён": "episode_finished",
}


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Записать публичные и закрытые файлы в каталог жюри."""
    directory = Path(output_dir)
    write_episode(_generate_episode(seed, config), directory, directory)


def _findings_list(submission, episode_id):
    if not isinstance(submission, dict) or submission.get("episode_id", episode_id) != episode_id:
        return None
    return submission.get("findings")


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Вычислить метрики и награду в закрытом процессе оценщика."""
    episode = read_episode(Path(episode_dir))
    try:
        metrics = score_episode_findings(episode, _findings_list(findings, episode["manifest"]["episode_id"]))
    except ValueError:
        _LOGGER.warning("Неверная структура отправленных выводов")
        return dict.fromkeys(_METRIC_NAMES) | {"return": -1}
    return {**metrics, "return": terminal_reward(metrics, inspection_steps)}


def _cursor_offset(cursor):
    if cursor is None:
        return None
    if isinstance(cursor, str) and cursor.isascii() and cursor.isdecimal() and len(cursor) <= 10:
        return int(cursor)
    return -1


class _JudgeEnv:
    def __init__(self):
        self._env = AuditEnv(mode="eval", auto_reset=False, calculator_as_text=True)

    def reset(self, seed: int, config: dict):
        config = {} if config is None else config
        mode = config.get("mode", "closed")
        if mode not in ("closed", "train"):
            raise ValueError("mode должен принимать значение closed или train")
        self._env.mode = "eval" if mode == "closed" else "train"
        observation, _ = self._env.reset(seed, config.get("team_config"), episode_dir=config.get("episode_dir"))
        observation["remaining_steps"] = observation.pop("steps_remaining")
        return observation, {}

    def step(self, action):
        name = action.get("name") if isinstance(action, dict) else None
        args = action.get("args") if isinstance(action, dict) else None
        internal = dict(args) if isinstance(args, dict) else {}
        internal["action"] = name
        if name == "list_records":
            internal["cursor"] = _cursor_offset(internal.get("cursor"))
        elif name == "submit":
            episode_id = self._env.episode["manifest"]["episode_id"]
            internal = {"action": name, "episode_id": episode_id,
                        "findings": _findings_list(internal.get("findings"), episode_id)}

        payload, reward, terminated, truncated, _ = self._env.step(internal)
        error = payload.get("error")
        result = None
        if "records" in payload:
            cursor = payload["next_cursor"]
            result = {"records": payload["records"], "next_cursor": None if cursor is None else str(cursor)}
        elif "accepted" in payload:
            result = {"accepted": True}
        elif "record" in payload or "result" in payload:
            result = payload.get("record", payload.get("result"))

        code = _ERROR_CODES.get(error)
        if code == "invalid_submission":
            _LOGGER.warning("Неверная структура отправленных выводов")
        observation = {"result": result, "remaining_steps": payload["steps_remaining"],
                       "error": {"code": code, "message": error} if error else None}
        return observation, reward, terminated, truncated, {}


def make_env():
    """Создать среду с каноническим протоколом reset и step."""
    return _JudgeEnv()
