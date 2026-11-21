import copy
import csv
import hashlib
import json
import tempfile
import unittest
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Context, Decimal, Inexact, ROUND_HALF_UP, ROUND_UP, localcontext
from pathlib import Path

from fin_audit_rl.cli import benchmark_cases, build_open, open_cases, record_training, run_bench
from fin_audit_rl.env import AuditEnv
from fin_audit_rl.generator import ISSUE_TYPES, generate_episode
from fin_audit_rl.policies import simple_reconciler


_KEY = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "rate_id"}


def expected_base(invoice, manifest, quote=None):
    base, source = manifest["base_currency"], invoice["currency"]
    amount = invoice["total_minor"]
    if base == source:
        return amount
    assert quote["date"] == invoice["issued_at"][:10]
    assert (quote["source_currency"], quote["target_currency"]) == (source, base)
    precision = manifest["currency_exponents"]
    normalized = Decimal(amount) / 10 ** precision[source]
    converted = normalized * Decimal(quote["rate"]) * 10 ** precision[base]
    return int(converted.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


class DatasetTests(unittest.TestCase):
    def test_russian_explanations_survive_generation_and_demo_output(self):
        """Русские задания, документы и обоснования сохраняются в JSON и выводе демо."""
        env = AuditEnv()
        observation, _ = env.reset(2468, {"transactions": 65, "interaction_scenario": "cutoff_fx"})
        self.assertIn("Изучите", observation["task"])
        for rules in observation["policy"]["reporting"].values():
            for description in rules.values():
                self.assertTrue(any("а" <= char.lower() <= "я" for char in description), description)
        for invoice in env.episode["tables"]["invoices"]:
            self.assertTrue(all(any("а" <= char <= "я" for char in item["description"])
                                for item in invoice["line_items"]))
        simple_reconciler(env)
        findings = env.trajectory[-1]["action"]["findings"]
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(any("а" <= char <= "я" for char in f["explanation"]) for f in findings))
        output = subprocess.check_output([sys.executable, "-X", "utf8", "-B",
                                          "-m", "fin_audit_rl.cli", "--demo"]).decode("utf-8")
        self.assertIn("Изучите", output)
        self.assertNotIn("\\u0418", output)
        self.assertEqual(json.loads(output)["metrics"]["f1"], 1.0)

    def test_generation_and_reconciliation_are_independent_of_decimal_context(self):
        """Генерация и сверка сохраняют результат при внешних настройках Decimal."""
        config = {"transactions": 30, "base_currency": "KZT",
                  "issue_types": ["fx_date_rate", "period_cutoff"]}
        expected = generate_episode(7, config)
        with localcontext(Context(prec=6, rounding=ROUND_UP, traps=[Inexact])):
            env = AuditEnv()
            env.reset(7, config)
            self.assertEqual(env.episode, expected)
            simple_reconciler(env)
        self.assertEqual(env.evaluator_result()["f1"], 1.0)

    def test_almaty_uses_historical_offsets_even_without_system_timezone_data(self):
        """Исторические смещения Алматы учитываются без системной базы часовых поясов."""
        config = {"transactions": 400, "timezone": "Asia/Almaty", "issue_count": 0}
        before = generate_episode(1001, config)
        after = generate_episode(1000, config)
        self.assertEqual(before["manifest"]["period_start"], "2024-01-01T00:00:00+06:00")
        self.assertTrue(after["manifest"]["period_start"].endswith("+05:00"))
        march_dues = [doc["due_at"] for doc in before["tables"]["invoices"] if doc["due_at"][:7] == "2024-03"]
        self.assertTrue(march_dues)
        self.assertTrue(all(due.endswith("+05:00") for due in march_dues))

    def test_issue_composition_reuses_the_same_underlying_company_history(self):
        """Разные наборы ошибок используют одну исходную историю компании."""
        configs = ({"issue_count": 0}, {"issue_count": 12}, {"issue_types": ["period_cutoff"]},
                   {"interaction_scenario": "cutoff_fx"}, {"interaction_scenario": "partial_fx"})
        episodes = [generate_episode(99, config) for config in configs]
        self.assertEqual(len({ep["manifest"]["episode_id"] for ep in episodes}), len(configs))
        self.assertEqual(len({ep["provenance"][0]["book_sha256"] for ep in episodes}), 1)
        self.assertEqual(episodes[-1], generate_episode(99, {**configs[-1], "issue_count": 2}))

    def test_benchmark_uses_same_cases_for_all_policies(self):
        """Все базовые политики проверяются на одинаковых случаях."""
        with tempfile.TemporaryDirectory() as temp:
            results = run_bench(Path(temp))
        cases = defaultdict(list)
        for run in results:
            cases[run["seed"]].append(run)
            self.assertFalse(run["truncated"])
            self.assertTrue(run["terminated"])
            self.assertLessEqual(run["steps"], 100)
            if run["policy"] == "mark-all":
                self.assertEqual(run["reward"], -1)
                self.assertEqual(run["metrics"]["fp"], 30)
            if run["policy"] in ("reconcile", "oracle"):
                self.assertEqual(run["metrics"]["f1"], 1)
        self.assertEqual(len(cases), 20)
        for runs in cases.values():
            self.assertEqual(len({r["episode_id"] for r in runs}), 1)
            self.assertEqual(len(runs), 4)

    def test_exported_splits_and_train_trajectories(self):
        """Экспорт содержит обучающий и проверочный наборы и связанные траектории."""
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            build_open(out)
            record_training(out)
            splits = {}
            seeds = {}
            for split in ("train", "dev"):
                directory = out / "open" / split
                episodes = json.loads((directory / "episodes.json").read_text())["episodes"]
                splits[split] = {ep["episode_id"] for ep in episodes}
                seeds[split] = {ep["seed"] for ep in episodes}
                self.assertEqual(len(splits[split]), 10)
                self.assertEqual({p.name for p in directory.iterdir() if p.is_dir()}, splits[split])
                self.assertEqual({ep["seed"] for ep in episodes}, {seed for seed, _ in open_cases(split)})
                for folder in (directory / episode_id for episode_id in splits[split]):
                    manifest = json.loads((folder / "manifest.json").read_text())
                    self.assertGreaterEqual(manifest["ledger_row_count"], 200)
                    self.assertLessEqual(manifest["document_count"], 200)
                    for table in _KEY:
                        file = folder / f"{table}.{('csv' if table in ('ledger', 'fx') else 'jsonl')}"
                        with file.open(encoding="utf-8", newline="") as stream:
                            records = (list(csv.DictReader(stream)) if table in ("ledger", "fx")
                                       else [json.loads(line) for line in stream])
                        self.assertEqual(len(records), manifest["record_counts"][table])
                        self.assertEqual(len({r[_KEY[table]] for r in records}), len(records))
                        if table == "ledger":
                            self.assertIn(manifest["csv_null"], {r["payment_id"] for r in records})
                            self.assertNotIn("", {r["payment_id"] for r in records})
            self.assertTrue(splits["train"].isdisjoint(splits["dev"]))
            self.assertTrue(seeds["train"].isdisjoint(seeds["dev"]))
            self.assertFalse(list((out / "open").rglob("truth.json")))
            self.assertFalse(list((out / "open").rglob("provenance.json")))
            runs = defaultdict(list)
            for line in (out / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertIn(row["episode_id"], splits["train"])
                self.assertIn(row["policy"], ("random", "reconcile"))
                self.assertTrue({"truth", "provenance", "seed", "metrics"}.isdisjoint(row["observation"]))
                runs[row["trajectory_id"]].append(row)
            self.assertEqual(len(runs), 20)
            for rows in runs.values():
                self.assertEqual([r["step"] for r in rows], list(range(len(rows))))
                self.assertIn("manifest", rows[0]["observation"])
                self.assertTrue(all(r["reward"] == 0 for r in rows[:-1]))
                self.assertTrue(rows[-1]["terminated"])
                self.assertFalse(rows[-1]["truncated"])

    def test_mutations_are_reversible_and_public_evidence_proves_each_issue(self):
        """Изменения обратимы, каждая ошибка подтверждается открытыми доказательствами."""
        cases = [*benchmark_cases(), *open_cases("train"), *open_cases("dev")]
        for seed, config in cases:
            with self.subTest(seed=seed, config=config):
                ep = generate_episode(seed, config)
                manifest = ep["manifest"]
                tables = {table: {row[_KEY[table]]: row for row in rows}
                          for table, rows in ep["tables"].items()}
                issues = {issue["issue_id"]: issue for issue in ep["truth"]}
                expected_count = (2 if "interaction_scenario" in config else
                                  config.get("issue_count", len(config.get("issue_types", []))))
                self.assertEqual(len(issues), expected_count)
                self.assertEqual(len({i for table in tables.values() for i in table}),
                                 sum(len(table) for table in tables.values()))
                record_index = {record_id: (table, row) for table, rows in tables.items()
                                for record_id, row in rows.items()}
                for issue in ep["truth"]:
                    self.assertEqual(issue["currency"], manifest["base_currency"])
                    for group in issue["acceptable_evidence_sets"]:
                        self.assertLessEqual(len(group), 10)
                        evidence = defaultdict(list)
                        for item in group:
                            table, row = record_index[item]
                            evidence[table].append(row)
                        self.assert_evidence(issue, evidence, manifest)
                restored = copy.deepcopy(tables)
                for mutation in reversed(ep["provenance"][1:]):
                    self.assertEqual(mutation["type"], issues[mutation["issue_id"]]["type"])
                    for change in mutation["changes"]:
                        table = change["table"]
                        current = change["after"]
                        key = current[_KEY[table]]
                        self.assertEqual(restored[table].pop(key), current)
                        if change["before"] is not None:
                            restored[table][key] = change["before"]
                canonical = {table: sorted(json.dumps(row, sort_keys=True) for row in rows.values())
                             for table, rows in restored.items()}
                base = ep["provenance"][0]
                self.assertEqual(hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
                                 base["book_sha256"])
                self.assertEqual({table: len(rows) for table, rows in restored.items()}, base["record_counts"])
                self.assert_clean_book(restored, manifest)

    def assert_clean_book(self, books, manifest):
        invoice_register = {doc["invoice_id"]: doc for doc in books["invoices"].values()}
        payment_register = {doc["payment_id"]: doc for doc in books["payments"].values()}
        quotes = {(r["source_currency"], r["date"]): r for r in books["fx"].values()}
        dates = defaultdict(list)
        ledgers = defaultdict(list)
        for row in books["ledger"].values():
            self.assertEqual(row["currency"], manifest["base_currency"])
            self.assertIn(row["invoice_id"], invoice_register)
            self.assertEqual(row["entity_id"], invoice_register[row["invoice_id"]]["entity_id"])
            if row["payment_id"] is not None:
                self.assertIn(row["payment_id"], payment_register)
            group = (row["invoice_id"], row["payment_id"])
            ledgers[group].append(row)
            dates[group].append(row["timestamp"])
        paid_total = Counter()
        references = set()
        for inv, doc in invoice_register.items():
            source_ref = (doc["entity_id"], doc["supplier_invoice_ref"])
            self.assertNotIn(source_ref, references)
            references.add(source_ref)
            self.assertEqual(sum(item["amount_minor"] for item in doc["line_items"]), doc["total_minor"])
            group = (inv, None)
            self.assertEqual(dates[group], [doc["issued_at"]] * 2)
            amount = expected_base(doc, manifest, quotes.get((doc["currency"], doc["issued_at"][:10])))
            self.assert_pair(ledgers[group], "expense:operations", "liability:accounts_payable", amount)
            paid_total[inv] = amount
        for payment_id, pay in payment_register.items():
            self.assertEqual(len(pay["invoice_refs"]), 1)
            ref = pay["invoice_refs"][0]
            amount, inv = pay["amount_minor"], ref["invoice_id"]
            self.assertEqual(ref["amount_minor"], amount)
            group = (inv, payment_id)
            self.assertEqual(dates[group], [pay["timestamp"]] * 2)
            self.assertGreater(datetime.fromisoformat(pay["timestamp"]),
                               datetime.fromisoformat(invoice_register[inv]["issued_at"]))
            if pay["direction"] == "refund":
                self.assert_pair(ledgers[group], "asset:cash", "expense:operations", amount)
            else:
                self.assert_pair(ledgers[group], "liability:accounts_payable", "asset:cash", amount)
                paid_total[inv] -= amount
        self.assertTrue(all(amount >= 0 for amount in paid_total.values()))

    def assert_pair(self, rows, debit, credit, amount):
        self.assertEqual(len(rows), 2)
        self.assertCountEqual([(r["account"], r["debit_minor"], r["credit_minor"]) for r in rows],
                              [(debit, amount, 0), (credit, 0, amount)])

    def assert_evidence(self, issue, evidence, manifest):
        kind = issue["type"]
        self.assertIn(kind, ISSUE_TYPES)
        invoice, *other = evidence["invoices"]
        rows = evidence["ledger"]
        ids = [r["row_id"] for r in rows]
        inv = invoice["invoice_id"]
        quote = next(iter(evidence["fx"]), None)
        if kind == "payment_allocation":
            payment, = evidence["payments"]
            self.assertEqual(payment["currency"], manifest["base_currency"])
            self.assertEqual({r["payment_id"] for r in rows}, {payment["payment_id"]})
            self.assertEqual({r["invoice_id"] for r in rows}, {inv})
            self.assert_pair(rows, "liability:accounts_payable", "asset:cash", payment["amount_minor"])
            self.assertTrue(all(ref["invoice_id"] == inv for ref in payment["invoice_refs"]))
            amount = payment["amount_minor"] - sum(ref["amount_minor"] for ref in payment["invoice_refs"])
            cash = next(r for r in rows if r["account"] == "asset:cash")
            affected = [inv, payment["payment_id"], cash["row_id"]]
        else:
            amount = expected_base(invoice, manifest, quote)
            affected = [inv, *ids]
            if kind == "duplicate_invoice":
                duplicate, = other
                dup_id = duplicate["invoice_id"]
                self.assertNotEqual(inv, dup_id)
                for key in invoice.keys() - {"doc_id", "invoice_id"}:
                    self.assertEqual(invoice[key], duplicate[key])
                self.assertEqual({r["invoice_id"] for r in rows}, {inv, dup_id})
                for invoice_id in (inv, dup_id):
                    self.assert_pair([r for r in rows if r["invoice_id"] == invoice_id],
                                     "expense:operations", "liability:accounts_payable", amount)
                affected.append(dup_id)
            else:
                self.assertEqual({r["invoice_id"] for r in rows}, {inv})
                posting = next(r for r in rows if r["account"] == "expense:operations")
                self.assert_pair(rows, "expense:operations", "liability:accounts_payable", posting["debit_minor"])
                if kind == "fx_date_rate":
                    self.assertIsNotNone(quote)
                    amount = abs(posting["debit_minor"] - amount)
                else:
                    start, end, issued, posted = map(datetime.fromisoformat, (
                        manifest["period_start"], manifest["period_end"], invoice["issued_at"], posting["timestamp"]))
                    self.assertNotEqual(start <= issued <= end, start <= posted <= end)
        self.assertGreater(amount, 0)
        self.assertEqual(issue["expected_amount_minor"], amount)
        self.assertCountEqual(affected, issue["affected_ids"])
