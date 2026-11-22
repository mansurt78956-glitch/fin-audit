"""Открытый детерминированный код взаимно-однозначного сопоставления выводов и расчёта награды."""
from __future__ import annotations

def _canonical_ids(value) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return ()
    return tuple(sorted(set(value)))


def _valid_finding(f: dict) -> bool:
    return (isinstance(f, dict) and isinstance(f.get("type"), str)
            and type(f.get("claimed_amount_minor")) is int
            and isinstance(f.get("currency"), str)
            and isinstance(f.get("affected_ids"), list)
            and all(isinstance(x, str) for x in f["affected_ids"])
            and isinstance(f.get("evidence_ids"), list)
            and all(isinstance(x, str) for x in f["evidence_ids"])
            and len(f["evidence_ids"]) <= 10
            and isinstance(f.get("explanation"), str)
            and len(f["explanation"]) <= 1000)


def _maximum_size(candidates, start, unavailable):
    by_finding = {}

    def augment(issue, seen):
        for finding in candidates[issue]:
            if finding in seen:
                continue
            seen.add(finding)
            if finding not in by_finding or augment(by_finding[finding], seen):
                by_finding[finding] = issue
                return True
        return False

    for issue in range(start, len(candidates)):
        augment(issue, unavailable.copy())
    return len(by_finding)


def _canonical_matching(candidates):
    """Выбрать первые лексикографически пары, сохраняющие максимальную мощность."""
    used, matches = set(), []
    target = _maximum_size(candidates, 0, used)
    for issue, neighbors in enumerate(candidates):
        if len(matches) == target:
            break
        for finding in neighbors:
            if finding in used:
                continue
            blocked = used | {finding}
            if len(matches) + 1 + _maximum_size(candidates, issue + 1, blocked) == target:
                matches.append((issue, finding))
                used = blocked
                break
    return matches


def score_submission(findings: list, truth: list, amount_tolerance_minor: int = 0, *,
                     available_evidence_ids: set[str] | None = None) -> dict:
    """Максимальное взаимно-однозначное сопоставление с детерминированным разрешением неоднозначности."""
    if not isinstance(findings, list) or len(findings) > 30 or not isinstance(truth, list):
        raise ValueError("findings и truth должны быть списками, допускается не более 30 выводов")
    if type(amount_tolerance_minor) is not int or amount_tolerance_minor < 0:
        raise ValueError("допуск должен быть целым неотрицательным числом")
    if not all(_valid_finding(f) for f in findings):
        raise ValueError("неверная структура вывода")
    candidates = [[] for _ in truth]
    ordered_truth = sorted(truth, key=lambda issue: issue.get("issue_id", ""))
    affected_ids = [_canonical_ids(issue["affected_ids"]) for issue in ordered_truth]
    for index, finding in enumerate(findings):
        affected = _canonical_ids(finding["affected_ids"])
        evidence = set(finding["evidence_ids"])
        if available_evidence_ids is not None and not evidence <= available_evidence_ids:
            continue
        for issue_index, issue in enumerate(ordered_truth):
            if finding["type"] != issue["type"] or affected != affected_ids[issue_index]:
                continue
            if finding["currency"] != issue["currency"]:
                continue
            if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > amount_tolerance_minor:
                continue
            if any(evidence.issuperset(group) for group in issue["acceptable_evidence_sets"]):
                candidates[issue_index].append(index)
    tp = len(_canonical_matching(candidates))
    fp, fn = len(findings) - tp, len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 1.0 if not findings and not truth else (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def terminal_reward(metrics: dict, inspection_steps: int) -> float:
    return max(-1.0, min(1.0, 2.0 * metrics["f1"] - 1.0 - 0.002 * inspection_steps))
