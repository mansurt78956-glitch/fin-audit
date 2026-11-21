"""Базовые политики для автономного запуска. Эталонная политика получает данные оценщика."""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext


def _submit(env, findings):
    episode_id = env.initial_observation["manifest"]["episode_id"]
    env.step({"action": "submit", "episode_id": episode_id, "findings": findings})
    return env.trajectory


def random_explorer(env, seed=0):
    rng = random.Random(seed)
    tables = ["ledger", "invoices", "payments", "fx"]
    rng.shuffle(tables)
    for table in tables:
        _, _, terminated, truncated, _ = env.step(
            {"action": "list_records", "table": table, "cursor": None, "limit": rng.randint(5, 20)})
        if terminated or truncated:
            return env.trajectory
    return _submit(env, [])


def _records(env, table):
    cursor = None
    while True:
        obs, _, terminated, truncated, _ = env.step(
            {"action": "list_records", "table": table, "cursor": cursor, "limit": 50})
        if terminated or truncated:
            return
        yield from obs["records"]
        cursor = obs["next_cursor"]
        if cursor is None:
            return


def simple_reconciler(env, seed=0):
    """Сверить открытые документы, проводки, распределения платежей и валютные курсы."""
    manifest = env.initial_observation["manifest"]
    base, exponents = manifest["base_currency"], manifest["currency_exponents"]
    start, end = (datetime.fromisoformat(manifest[key]) for key in ("period_start", "period_end"))
    registers = {}
    for table in ("ledger", "invoices", "payments", "fx"):
        registers[table] = list(_records(env, table))
        if env.done:
            return env.trajectory
    postings, settlements, groups = defaultdict(list), defaultdict(list), defaultdict(list)
    invoices = {doc["invoice_id"]: doc for doc in registers["invoices"]}
    rates = {(row["source_currency"], row["date"]): row for row in registers["fx"]}
    for row in registers["ledger"]:
        if row["payment_id"] is None:
            postings[row["invoice_id"]].append(row)
        else:
            settlements[row["payment_id"]].append(row)

    findings = []
    money_context = Context(prec=28, rounding=ROUND_HALF_EVEN)

    def report(kind, affected, amount, evidence, explanation):
        findings.append({"type": kind, "affected_ids": sorted(affected),
                         "claimed_amount_minor": amount, "currency": base,
                         "evidence_ids": sorted(evidence), "explanation": explanation})

    def invoice_details(doc):
        amount = doc["total_minor"]
        currency = doc["currency"]
        issued = datetime.fromisoformat(doc["issued_at"])
        quote = []
        if currency != base:
            rate = rates[currency, issued.date().isoformat()]
            with localcontext(money_context):
                value = Decimal(amount) * Decimal(rate["rate"]) * Decimal(10) ** (exponents[base] - exponents[currency])
                amount = int(value.to_integral_value(rounding=ROUND_HALF_UP))
            quote = [rate["rate_id"]]
        return issued, amount, quote

    for invoice_id, doc in invoices.items():
        groups[doc["entity_id"], doc["supplier_invoice_ref"]].append(doc)
        rows = postings[invoice_id]
        expense = next(row for row in rows if row["account"] == "expense:operations")
        issued, amount, quote = invoice_details(doc)
        posted = datetime.fromisoformat(expense["timestamp"])
        row_ids = [row["row_id"] for row in rows]
        affected = [invoice_id, *row_ids]
        evidence = [doc["doc_id"], *row_ids, *quote]
        if quote and expense["debit_minor"] != amount:
            report("fx_date_rate", affected, abs(expense["debit_minor"] - amount), evidence,
                   "Сумма проводки в базовой валюте отличается от суммы счёта, пересчитанной по курсу на местную дату его выставления.")
        if (start <= issued <= end) != (start <= posted <= end):
            report("period_cutoff", affected, amount, evidence,
                   "Дата выставления счёта и дата проводки относятся к разным отчётным периодам.")

    for group in groups.values():
        if len(group) < 2:
            continue
        first, *duplicates = group
        for duplicate in duplicates:
            ids = [first["invoice_id"], duplicate["invoice_id"]]
            row_ids = [row["row_id"] for invoice_id in ids for row in postings[invoice_id]]
            _, amount, quote = invoice_details(first)
            report("duplicate_invoice", ids + row_ids, amount,
                   [first["doc_id"], duplicate["doc_id"], *row_ids, *quote],
                   "Обе пары проводок ссылаются на один номер счёта поставщика.")

    for payment in registers["payments"]:
        if payment["direction"] != "payment":
            continue
        unallocated = payment["amount_minor"] - sum(ref["amount_minor"] for ref in payment["invoice_refs"])
        if not unallocated:
            continue
        payment_id = payment["payment_id"]
        rows = settlements[payment_id]
        cash = next(row for row in rows if row["account"] == "asset:cash")
        invoice_id = cash["invoice_id"]
        report("payment_allocation", [invoice_id, payment_id, cash["row_id"]], unallocated,
               [payment["doc_id"], invoices[invoice_id]["doc_id"], *(row["row_id"] for row in rows)],
               "Проводки по денежным средствам и кредиторской задолженности подтверждают погашение на сумму больше распределённой части платежа.")
    return _submit(env, findings[:30])


def mark_everything(env, seed=0):
    rows = env.step({"action": "list_records", "table": "ledger", "limit": 30})[0].get("records", [])
    return _submit(env, [{"type": "duplicate_invoice", "affected_ids": [row["row_id"]],
                          "claimed_amount_minor": row["debit_minor"] or row["credit_minor"],
                          "currency": row["currency"], "evidence_ids": [row["row_id"]],
                          "explanation": "Строка помечена базовой политикой массового выявления ошибок."} for row in rows])


def oracle(env, seed=0):
    """Эталонная политика для использования оценщиком."""
    return _submit(env, [{"type": issue["type"], "affected_ids": issue["affected_ids"],
                          "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
                          "evidence_ids": issue["acceptable_evidence_sets"][0],
                          "explanation": "Вывод подтверждается перечисленными исходными записями."}
                         for issue in env.episode["truth"]])
