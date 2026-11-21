"""Детерминированные синтетические книги с последующим внесением и регистрацией ошибок учёта."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from decimal import Context, Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = "1.2"
GENERATOR_VERSION = "2.2"
ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_date_rate", "period_cutoff")
_NAMES = ("Полярная звезда", "Можжевельник", "Кобальт", "Янтарь", "Пионер", "Секвойя", "Мозаика")
_EXPONENT = {"USD": 2, "EUR": 2, "GBP": 2, "KZT": 2, "JPY": 0}
_USD_QUOTES = {"USD": "1", "EUR": "1.08", "GBP": "1.27", "KZT": "0.002", "JPY": "0.0067"}
_JANUARY_OFFSETS = {"UTC": 0, "America/New_York": -5, "America/Chicago": -6,
                    "America/Los_Angeles": -8, "Europe/London": 0, "Europe/Paris": 1,
                    "Asia/Almaty": 5, "Asia/Qyzylorda": 5, "Asia/Tokyo": 9}


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def _config(config, rng):
    config = dict(config or {})
    requested = config.get("issue_types")
    interaction = config.get("interaction_scenario")
    if interaction is not None:
        interactions = {"partial_fx": ["payment_allocation", "fx_date_rate"],
                        "cutoff_fx": ["period_cutoff", "fx_date_rate"]}
        if not isinstance(interaction, str) or interaction not in interactions:
            raise ValueError("неизвестный сценарий взаимодействия ошибок")
        chosen = interactions[interaction]
    elif requested is not None:
        if not isinstance(requested, (list, tuple)):
            raise ValueError("issue_types должен быть списком или кортежем")
        chosen = list(requested)
    else:
        count = config.get("issue_count", rng.randrange(5))
        if type(count) is not int or not 0 <= count <= 12:
            raise ValueError("issue_count должен быть целым числом от 0 до 12")
        chosen = rng.choices(ISSUE_TYPES, k=count)
    if len(chosen) > 12 or any(kind not in ISSUE_TYPES for kind in chosen):
        raise ValueError("укажите не более 12 поддерживаемых типов ошибок")
    return config, chosen, interaction


def _snapshot_hash(tables):
    canonical = {table: sorted(json.dumps(row, sort_keys=True) for row in rows)
                 for table, rows in tables.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def generate_episode(seed: int, config: dict | None = None) -> dict:
    """Построить корректную историю и сохранить обратимые изменения для оценщика."""
    seed = int(seed)
    rng = random.Random(seed)
    config, chosen, interaction = _config(config, random.Random(f"issue-schedule:{seed}"))
    count = max(20, min(int(config.get("transactions", 64)), 400))
    base = config.get("base_currency", "USD")
    if base not in _EXPONENT:
        raise ValueError("базовая валюта не поддерживается")
    zone = config.get("timezone", "America/New_York")
    year = rng.choice((2024, 2025, 2026))
    almaty_before_change = zone == "Asia/Almaty" and year == 2024
    if zone in _JANUARY_OFFSETS:
        offset = 6 if almaty_before_change else _JANUARY_OFFSETS[zone]
        tz = timezone(timedelta(hours=offset), name=zone)
    else:
        try:
            tz = ZoneInfo(zone)
        except (ZoneInfoNotFoundError, TypeError, ValueError):
            raise ValueError("часовой пояс не поддерживается") from None
    start = datetime(year, 1, 1, tzinfo=tz)
    end = datetime(year, 2, 1, tzinfo=tz) - timedelta(seconds=1)
    company = rng.choice(_NAMES) + " " + rng.choice(("Системы", "Торговля", "Лаборатории", "Сервис"))
    tables = {k: [] for k in ("ledger", "invoices", "payments", "fx")}
    ids = set()

    def new_id(prefix):
        while True:
            value = f"{prefix}-{rng.getrandbits(64):016x}"
            if value not in ids:
                ids.add(value)
                return value

    foreign = [cur for cur in _EXPONENT if cur != base]
    rates = {}
    money_context = Context(prec=28, rounding=ROUND_HALF_EVEN)
    with localcontext(money_context):
        references = {cur: Decimal(_USD_QUOTES[cur]) / Decimal(_USD_QUOTES[base]) for cur in foreign}
        for offset in range(-12, 31):
            day = (start + timedelta(days=offset)).date().isoformat()
            for cur, reference in references.items():
                jitter = Decimal(rng.randrange(960, 1041)).scaleb(-3)
                rate = (reference * jitter).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
                row = {"rate_id": new_id("FX"), "date": day, "source_currency": cur,
                       "target_currency": base, "rate": str(rate)}
                rates[day, cur] = row
                tables["fx"].append(row)

    def value_in_base(amount, cur, day):
        if cur == base:
            return amount, None
        rate_row = rates[day, cur]
        with localcontext(money_context):
            value = Decimal(amount) * Decimal(rate_row["rate"])
            value = value.scaleb(_EXPONENT[base] - _EXPONENT[cur])
            return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)), rate_row

    def add_pair(debit_account, credit_account, amount, stamp, entity, invoice, payment=None):
        common = {"timestamp": _iso(stamp), "currency": base, "entity_id": entity,
                  "invoice_id": invoice, "payment_id": payment}
        debit = {"row_id": new_id("LR"), **common, "account": debit_account,
                 "debit_minor": amount, "credit_minor": 0}
        credit = {"row_id": new_id("LR"), **common, "account": credit_account,
                  "debit_minor": 0, "credit_minor": amount}
        tables["ledger"].extend((debit, credit))
        return debit, credit

    foreign_indices = set(rng.sample(range(count), max(12, count // 3)))
    shared_boundary = rng.choice(sorted(foreign_indices))
    boundary_indices = {shared_boundary, *rng.sample(
        [i for i in range(count) if i != shared_boundary], max(12, count // 4) - 1)}
    paid_indices = set(rng.sample(range(count), count * 3 // 4))
    suppliers = [new_id("ENT") for _ in range(max(4, count // 5))]
    histories = []
    for index in range(count):
        cur = rng.choice(foreign) if index in foreign_indices else base
        boundary = index in boundary_indices
        offsets = (-4, -3, -2, -1, 0, 1, 2, 3) if boundary else range(4, 31)
        issued = start + timedelta(days=rng.choice(offsets), hours=rng.randrange(24),
                                    minutes=rng.choice((0, 15, 30, 45)))
        amount = rng.choice((25_000, 75_000)) if rng.random() < 0.2 else rng.randrange(5_000, 500_000)
        invoice_id, entity = new_id("INV"), rng.choice(suppliers)
        parts = [amount]
        if rng.random() < 0.4:
            parts = [amount // 3, amount - amount // 3]
        due = issued + timedelta(days=30)
        # По базе IANA 2024a Алматы перешёл на UTC+05 с 1 марта 2024 года.
        if almaty_before_change and due.month == 3:
            due = due.replace(tzinfo=timezone(timedelta(hours=5), name=zone))
        invoice = {"doc_id": new_id("DOC"), "invoice_id": invoice_id, "entity_id": entity,
                   "supplier_invoice_ref": new_id("REF"),
                   "issued_at": _iso(issued), "due_at": _iso(due),
                   "currency": cur, "total_minor": amount,
                   "line_items": [{"description": rng.choice(("хостинг", "консалтинг", "оборудование", "перевозка", "материалы")),
                                   "amount_minor": part} for part in parts]}
        tables["invoices"].append(invoice)
        posted_amount, rate_row = value_in_base(amount, cur, issued.date().isoformat())
        posting = add_pair("expense:operations", "liability:accounts_payable", posted_amount,
                           issued, entity, invoice_id)
        history = {"invoice": invoice, "issued": issued, "posting": posting, "base_amount": posted_amount,
                   "rate": rate_row, "boundary": boundary, "payment": None, "settlement": None}
        histories.append(history)
        if index not in paid_indices:
            continue

        # Договор фиксирует сумму погашения в базовой валюте по курсу на дату счёта.
        installments = [posted_amount]
        settlement_kind = rng.random()
        if posted_amount > 1 and settlement_kind < 0.18:
            installments = [posted_amount // 2, posted_amount - posted_amount // 2]
        elif posted_amount > 1 and settlement_kind < 0.28:
            installments = [posted_amount // 2]
        for part_index, part in enumerate(installments):
            stamp = issued + timedelta(days=7 + part_index * 3)
            payment_id = new_id("PAY")
            payment = {"doc_id": new_id("DOC"), "payment_id": payment_id, "timestamp": _iso(stamp),
                       "currency": base, "amount_minor": part, "direction": "payment",
                       "invoice_refs": [{"invoice_id": invoice_id, "amount_minor": part}]}
            tables["payments"].append(payment)
            settlement = add_pair("liability:accounts_payable", "asset:cash", part, stamp,
                                  entity, invoice_id, payment_id)
            if part_index == 0:
                history["payment"], history["settlement"] = payment, settlement
        if installments and sum(installments) == posted_amount and rng.random() < 0.08:
            stamp = issued + timedelta(days=15)
            refund_id, refund = new_id("PAY"), max(1, posted_amount // 20)
            tables["payments"].append({"doc_id": new_id("DOC"), "payment_id": refund_id,
                "timestamp": _iso(stamp), "currency": base, "amount_minor": refund,
                "direction": "refund", "invoice_refs": [{"invoice_id": invoice_id, "amount_minor": refund}]})
            add_pair("asset:cash", "expense:operations", refund, stamp, entity, invoice_id, refund_id)

    provenance = [{"stage": "base_book", "book_sha256": _snapshot_hash(tables),
                   "record_counts": {k: len(v) for k, v in tables.items()}}]
    eligible = {
        "duplicate_invoice": lambda h: True,
        "payment_allocation": lambda h: h["payment"] is not None,
        "fx_date_rate": lambda h: h["rate"] is not None,
        "period_cutoff": lambda h: h["boundary"],
    }
    if interaction:
        target = rng.choice([h for h in histories if all(eligible[k](h) for k in chosen)])
        targets = [target] * len(chosen)
    else:
        unused = list(histories)
        targets = []
        for kind in chosen:
            # Для каждого типа ошибки исходно доступно не менее 12 подходящих историй.
            target = rng.choice([h for h in unused if eligible[kind](h)])
            targets.append(target)
            unused.remove(target)

    truth = []
    for kind, history in zip(chosen, targets):
        invoice, issued = history["invoice"], history["issued"]
        posting, base_amount, rate_row = history["posting"], history["base_amount"], history["rate"]
        invoice_id = invoice["invoice_id"]
        issue_id = new_id("ISS")
        affected = [invoice_id, *(row["row_id"] for row in posting)]
        evidence = [invoice["doc_id"], *(row["row_id"] for row in posting)]
        expected = base_amount
        changes = []
        if kind == "duplicate_invoice":
            duplicate = {**copy.deepcopy(invoice), "doc_id": new_id("DOC"), "invoice_id": new_id("INV")}
            tables["invoices"].append(duplicate)
            duplicate_rows = add_pair("expense:operations", "liability:accounts_payable", base_amount,
                                      issued, invoice["entity_id"], duplicate["invoice_id"])
            affected.extend([duplicate["invoice_id"], *(row["row_id"] for row in duplicate_rows)])
            evidence.extend([duplicate["doc_id"], *(row["row_id"] for row in duplicate_rows)])
            changes = [{"table": "invoices", "before": None, "after": duplicate}]
            changes.extend({"table": "ledger", "before": None, "after": row} for row in duplicate_rows)
        elif kind == "payment_allocation":
            payment = history["payment"]
            ap, cash = history["settlement"]
            before = copy.deepcopy(payment)
            allocated = rng.choice((0, payment["amount_minor"] // 2))
            payment["invoice_refs"] = [{"invoice_id": invoice_id, "amount_minor": allocated}] if allocated else []
            expected = payment["amount_minor"] - allocated
            affected = [invoice_id, payment["payment_id"], cash["row_id"]]
            evidence = [invoice["doc_id"], payment["doc_id"], ap["row_id"], cash["row_id"]]
            changes = [{"table": "payments", "before": before, "after": payment}]
        elif kind == "fx_date_rate":
            # Ошибка использует курс соседней даты или неверно введённый курс.
            wrong_amount, _ = value_in_base(invoice["total_minor"], invoice["currency"],
                                           (issued - timedelta(days=1)).date().isoformat())
            if wrong_amount == base_amount or rng.random() < 0.35:
                wrong_amount = base_amount + max(1, base_amount // 8)
            expected = abs(wrong_amount - base_amount)
            for row, field in zip(posting, ("debit_minor", "credit_minor")):
                before = row.copy()
                row[field] = wrong_amount
                changes.append({"table": "ledger", "before": before, "after": row})
        else:
            wrong_date = start - timedelta(days=1) if issued >= start else start
            for row in posting:
                before = row.copy()
                row["timestamp"] = _iso(wrong_date.replace(hour=issued.hour, minute=issued.minute))
                changes.append({"table": "ledger", "before": before, "after": row})
        if rate_row is not None and kind != "payment_allocation":
            evidence.append(rate_row["rate_id"])
        truth.append({"issue_id": issue_id, "type": kind, "affected_ids": sorted(affected),
                      "expected_amount_minor": expected, "currency": base,
                      "acceptable_evidence_sets": [evidence]})
        provenance.append({"stage": "mutation", "issue_id": issue_id, "type": kind,
                           "changes": copy.deepcopy(changes)})

    for rows in tables.values():
        rng.shuffle(rows)
    episode_key = json.dumps([GENERATOR_VERSION, seed, count, base, zone, chosen, interaction]).encode()
    episode_salt = rng.getrandbits(128).to_bytes(16, "big")
    episode_id = f"ep-{hashlib.blake2b(episode_key, key=episode_salt, digest_size=12).hexdigest()}"
    manifest = {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
                "episode_id": episode_id, "company": company, "base_currency": base,
        "period_start": _iso(start), "period_end": _iso(end), "timezone": zone,
        "record_counts": {k: len(v) for k, v in tables.items()},
        "document_count": len(tables["invoices"]) + len(tables["payments"]),
        "ledger_row_count": len(tables["ledger"]), "currency_exponents": dict(_EXPONENT),
                "rounding": "ROUND_HALF_UP", "csv_null": "null", "transaction_count": count}
    policy = {"fx_date": "invoice_issue_date_local", "fx_fallback": "nearest_on_or_before",
              "fx_quote": "target_major_per_source_major", "rounding": "ROUND_HALF_UP",
              "settlement_currency": "базовая валюта по договорному курсу на дату счёта",
              "period_bounds": "inclusive", "register_scope": "январь и операции соседних периодов для сверки",
              "cutoff_amount": "корректная сумма в базовой валюте по курсу на дату счёта",
              "fx_discrepancy": "модуль разности проводки и корректной суммы по курсу на дату счёта",
              "duplicate_key": ["entity_id", "supplier_invoice_ref"],
              "amount_tolerance_minor": 0,
              "reporting": {
                  "duplicate_invoice": {
                      "affected_ids": "ID обоих счетов и строки обеих пар проводок по расходам и кредиторской задолженности",
                      "evidence_ids": "ID документов обоих счетов, строки обеих пар проводок и ID курса для счёта в иностранной валюте",
                      "amount": "сумма одного счёта в базовой валюте"},
                  "payment_allocation": {
                      "affected_ids": "ID счёта, ID платежа и ID строки проводки по денежным средствам",
                      "evidence_ids": "ID документов счёта и платежа, ID строк погашения по денежным средствам и кредиторской задолженности",
                      "amount": "сумма платежа за вычетом суммы распределений"},
                  "fx_date_rate": {
                      "affected_ids": "ID счёта и ID строк проводок по расходам и кредиторской задолженности",
                      "evidence_ids": "ID документа счёта, ID строк проводок по расходам и кредиторской задолженности и ID курса на дату счёта",
                      "amount": "модуль разности проведённой и корректной сумм в базовой валюте"},
                  "period_cutoff": {
                      "affected_ids": "ID счёта и ID строк проводок по расходам и кредиторской задолженности",
                      "evidence_ids": "ID документа счёта, ID обеих строк проводки и ID курса для счёта в иностранной валюте",
                      "amount": "корректная сумма счёта в базовой валюте"},
              }}
    return {"manifest": manifest, "policy": policy, "tables": tables, "truth": truth,
            "provenance": provenance}
