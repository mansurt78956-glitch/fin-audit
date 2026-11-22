"""Среда с JSON-совместимым интерфейсом в стиле Gymnasium и инструментами аудита."""
from __future__ import annotations

import ast
import copy
import math
import operator
import re
import time
from decimal import Context, Decimal, DecimalException, ROUND_HALF_EVEN, localcontext
from pathlib import Path

from .episode_io import read_episode
from .generator import generate_episode
from .matcher import score_submission, terminal_reward

TABLES = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "rate_id"}
_DECIMAL = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
_ARITHMETIC = re.compile(r"[0-9.+*/%() \t\r\n-]+")
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def _calculator(expr, *, as_text=False):
    if not isinstance(expr, str) or len(expr) > 200:
        raise ValueError("калькулятор принимает короткое арифметическое выражение")
    expr = expr.strip()
    if not _ARITHMETIC.fullmatch(expr):
        raise ValueError("выражение не поддерживается калькулятором")
    tree = ast.parse(expr, mode="eval")

    def visit(node):
        if isinstance(node, ast.Constant):
            token = ast.get_source_segment(expr, node)
            if token and _DECIMAL.fullmatch(token):
                return Decimal(token)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                # Деление Decimal отбрасывает дробную часть. Сохраняем поведение операций // и %.
                quotient, remainder = divmod(left, right)
                if remainder and (left < 0) != (right < 0):
                    quotient -= 1
                    remainder += right
                return quotient if isinstance(node.op, ast.FloorDiv) else remainder
            return _OPERATORS[type(node.op)](left, right)
        raise ValueError("выражение не поддерживается калькулятором")

    with localcontext(Context(prec=80, rounding=ROUND_HALF_EVEN)):
        result = visit(tree.body)
    if not result.is_finite() or result.copy_abs() > 10**18:
        raise ValueError("результат калькулятора выходит за допустимый диапазон")
    if as_text:
        return format(result, "f")
    return int(result) if result == result.to_integral_value() else float(result)


def _row_id(table, row):
    if table == "fx" and "rate_id" not in row:
        return f"FX-{row['date']}-{row['source_currency']}-{row['target_currency']}"
    return row[TABLES[table]]


def _record_indexes(tables):
    records = {table: {_row_id(table, row): row for row in rows} for table, rows in tables.items()}
    documents = records["invoices"] | records["payments"]
    for table, key in (("invoices", "invoice_id"), ("payments", "payment_id")):
        records[table].update({row[key]: row for row in tables[table]})
    return records, documents


def score_episode_findings(episode, findings):
    evidence_ids = set()
    if findings:
        indexes, _ = _record_indexes(episode["tables"])
        evidence_ids = {record_id for index in indexes.values() for record_id in index}
    tolerance = episode["policy"].get("amount_tolerance_minor", 0)
    return score_submission(findings, episode["truth"], tolerance,
                            available_evidence_ids=evidence_ids)


class AuditEnv:
    def __init__(self, mode="train", max_steps=100, max_seconds=60, *,
                 auto_reset=True, calculator_as_text=False):
        if mode not in ("train", "eval"):
            raise ValueError("mode должен принимать значение train или eval")
        if type(max_steps) is not int:
            raise ValueError("max_steps должен быть целым числом")
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds < 0:
            raise ValueError("max_seconds должен быть конечным неотрицательным числом")
        self.mode, self.max_steps, self.max_seconds = mode, min(100, max(1, max_steps)), max_seconds
        self.calculator_as_text = calculator_as_text
        if auto_reset:
            self.reset(0)

    def reset(self, seed=0, config=None, *, episode_dir=None):
        self.started = time.monotonic()
        self.episode = (generate_episode(seed, config) if episode_dir is None
                        else read_episode(Path(episode_dir)))
        self._indexed_episode = None
        self.steps = 0
        self.done = self.terminated = self.truncated = False
        self.trajectory = []
        self._evaluator_result = self.evaluator_reward = None
        obs = {"task": "Изучите записи одной вымышленной компании и сообщите о финансовых расхождениях с подтверждающими доказательствами.",
               "manifest": self.episode["manifest"], "policy": self.episode["policy"],
               "steps_remaining": self.max_steps}
        self.initial_observation = copy.deepcopy(obs)
        return copy.deepcopy(obs), {"episode_id": obs["manifest"]["episode_id"]}

    def _get_record(self, table, ident):
        if self._indexed_episode is not self.episode:
            self._records, self._documents = _record_indexes(self.episode["tables"])
            self._indexed_episode = self.episode
        if table is None:
            return self._documents.get(ident)
        return self._records.get(table, {}).get(ident)

    def _inspect(self, name, action):
        if name == "list_records":
            table = action.get("table")
            offset = action.get("cursor")
            offset = 0 if offset is None else offset
            limit = action.get("limit", 50)
            if (not isinstance(table, str) or table not in TABLES or type(limit) is not int
                    or not 1 <= limit <= 50 or type(offset) is not int or offset < 0):
                raise ValueError("недопустимая таблица, курсор или размер страницы")
            rows = self.episode["tables"][table]
            page = rows[offset:offset + limit]
            next_offset = offset + len(page)
            return {"records": page, "next_cursor": next_offset if next_offset < len(rows) else None}
        if name in ("get_record", "get_document"):
            table = action.get("table") if name == "get_record" else None
            ident = action.get("id", action.get("doc_id")) if name == "get_record" else action.get("doc_id", action.get("id"))
            if name == "get_record" and (not isinstance(table, str) or table not in TABLES):
                raise ValueError("недопустимая таблица")
            item = self._get_record(table, ident) if isinstance(ident, str) else None
            return {"record": item} if item is not None else {"error": "запись не найдена"}
        if name == "calculator":
            return {"result": _calculator(action.get("expression"), as_text=self.calculator_as_text)}
        return {"error": "неизвестное действие"}

    def _transition(self, action, payload):
        obs = {**payload, "steps_remaining": max(0, self.max_steps - self.steps)}
        reward, info = 0.0, {}
        if self.done and self.mode == "train":
            reward = self.evaluator_reward
            info = {"metrics": copy.deepcopy(self._evaluator_result)}
            if "evaluator_error" in self._evaluator_result:
                info["evaluator_error"] = self._evaluator_result["evaluator_error"]
        elif self.done:
            info = {"evaluator_result_withheld": True}
        self.trajectory.append({"episode_id": self.episode["manifest"]["episode_id"],
                                "step": self.steps, "action": copy.deepcopy(action),
                                "observation": copy.deepcopy(obs), "reward": reward,
                                "terminated": self.terminated, "truncated": self.truncated})
        return copy.deepcopy(obs), reward, self.terminated, self.truncated, info

    def _finish(self, findings, inspection_steps, *, truncated=False, error=None):
        self.done, self.terminated, self.truncated = True, not truncated, truncated
        self._evaluator_result = score_episode_findings(self.episode, findings)
        if error:
            self._evaluator_result.update(f1=0.0, evaluator_error=error)
        self.evaluator_reward = terminal_reward(self._evaluator_result, inspection_steps)

    def abort(self, error=None):
        """Завершить эпизод при истечении времени или нарушении протокола агента."""
        if self.done:
            return self._finished()
        inspection_steps = self.steps
        self.steps += 1
        self._finish([], inspection_steps, truncated=True, error=error)
        return self._transition(None, {"final_answer": []})

    def _finished(self):
        return ({"error": "эпизод уже завершён", "steps_remaining": max(0, self.max_steps - self.steps)},
                0.0, self.terminated, self.truncated, {})

    def step(self, action):
        if self.done:
            return self._finished()
        if time.monotonic() - self.started >= self.max_seconds:
            return self.abort()
        if not isinstance(action, dict):
            action = {}
        self.steps += 1
        name = action.get("action")
        if name == "submit":
            payload = {"accepted": True}
            try:
                if action.get("episode_id") != self.episode["manifest"]["episode_id"]:
                    raise ValueError("episode_id не соответствует текущему эпизоду")
                self._finish(action.get("findings"), self.steps - 1)
            except ValueError:
                self._finish([], self.steps - 1, error="неверная структура отправленных выводов")
                payload = {"error": "неверная структура отправленных выводов"}
            if time.monotonic() - self.started >= self.max_seconds:
                self._finish([], self.steps - 1, truncated=True)
                payload = {"final_answer": []}
            return self._transition(action, payload)
        if self.steps == self.max_steps:
            self._finish([], self.steps, truncated=True)
            return self._transition(action, {"final_answer": []})
        try:
            payload = self._inspect(name, action)
        except (ValueError, TypeError, SyntaxError, OverflowError, DecimalException):
            payload = {"error": "недопустимое действие"}
        if time.monotonic() - self.started >= self.max_seconds:
            self._finish([], self.steps, truncated=True)
            payload = {"final_answer": []}
        return self._transition(action, payload)

    def evaluator_result(self):
        """Доступ к результату на стороне оценщика."""
        return copy.deepcopy(self._evaluator_result)
