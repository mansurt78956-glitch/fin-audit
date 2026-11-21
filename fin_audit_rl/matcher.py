"""Открытый детерминированный код взаимно-однозначного сопоставления выводов и расчёта награды."""
from __future__ import annotations

def _canonical_ids(value) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return ()
    return tuple(sorted(set(value)))


def _valid_finding(f: dict) -> bool:
    return (isinstance(f, dict) and f.get("type") and isinstance(f.get("type"), str)
            and type(f.get("claimed_amount_minor")) is int
            and isinstance(f.get("currency"), str)
            and isinstance(f.get("affected_ids"), list)
            and all(isinstance(x, str) for x in f["affected_ids"])
            and isinstance(f.get("evidence_ids"), list)
            and all(isinstance(x, str) for x in f["evidence_ids"])
            and len(f["evidence_ids"]) <= 10
            and isinstance(f.get("explanation"), str)
            and len(f["explanation"]) <= 1000)


def score_submission(findings: list, truth: list, amount_tolerance_minor: int = 0) -> dict:
    """Максимальное взаимно-однозначное сопоставление с детерминированным разрешением неоднозначности."""
    if not isinstance(findings, list) or len(findings) > 30 or not isinstance(truth, list):
        raise ValueError("findings и truth должны быть списками, допускается не более 30 выводов")
    if type(amount_tolerance_minor) is not int or amount_tolerance_minor < 0:
        raise ValueError("допуск должен быть целым неотрицательным числом")
    if not all(_valid_finding(f) for f in findings):
        raise ValueError("неверная структура вывода")
    # Повторный вывод получает одно место в сопоставлении при любом наборе доказательств.
    claim_edges = {}
    for f in findings:
        affected = _canonical_ids(f["affected_ids"])
        ev = frozenset(f["evidence_ids"])
        claim = (f["type"], affected, f["currency"], f["claimed_amount_minor"])
        edges = claim_edges.setdefault(claim, set())
        for ti, t in enumerate(truth):
            if f["type"] != t["type"] or affected != _canonical_ids(t["affected_ids"]):
                continue
            if f["currency"] != t["currency"] or abs(f["claimed_amount_minor"] - t["expected_amount_minor"]) > amount_tolerance_minor:
                continue
            if any(set(group).issubset(ev) for group in t["acceptable_evidence_sets"]):
                edges.add((abs(f["claimed_amount_minor"] - t["expected_amount_minor"]), ti))
    adjacency = [sorted(edges) for edges in claim_edges.values()]
    # Увеличивающие пути обеспечивают максимальное число совпадений. Кандидаты
    # упорядочены по ошибке суммы, затем по стабильному индексу эталона.
    matched_truth = {}
    def augment(fi, seen):
        for _, ti in adjacency[fi]:
            if ti in seen:
                continue
            seen.add(ti)
            if ti not in matched_truth or augment(matched_truth[ti], seen):
                matched_truth[ti] = fi
                return True
        return False
    tp = sum(1 for fi in range(len(adjacency)) if augment(fi, set()))
    fp, fn = len(findings) - tp, len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 1.0 if not findings and not truth else (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def terminal_reward(metrics: dict, inspection_steps: int) -> float:
    return max(-1.0, min(1.0, 2.0 * metrics["f1"] - 1.0 - 0.002 * inspection_steps))
