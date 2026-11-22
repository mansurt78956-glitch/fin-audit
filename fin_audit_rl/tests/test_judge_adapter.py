import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import judge_adapter
from fin_audit_rl.episode_io import read_episode, write_episode
from fin_audit_rl.matcher import _canonical_matching


def _action(name, **args):
    return {"name": name, "args": args}


def _finding(issue):
    return {"type": issue["type"], "affected_ids": issue["affected_ids"],
            "currency": issue["currency"], "claimed_amount_minor": issue["expected_amount_minor"],
            "evidence_ids": issue["acceptable_evidence_sets"][0], "explanation": "Сверены документы"}


class JudgeAdapterTests(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.directory = Path(temp) / "original"
        judge_adapter.generate_episode(42, {"transactions": 40, "issue_types": ["payment_allocation"]},
                                       str(self.directory))
        self.config = {"mode": "closed", "episode_dir": str(self.directory), "team_config": {"issue_count": 0}}
        self.episode = read_episode(self.directory)
        self.episode["provenance"] = json.loads((self.directory / "provenance.json").read_text(encoding="utf-8"))
        self.env = judge_adapter.make_env()

    def test_generated_files_roundtrip_through_judge_protocol(self):
        with patch("fin_audit_rl.env.generate_episode", side_effect=AssertionError("Фикстура должна читаться с диска")):
            initial, info = self.env.reset(999, self.config)
        self.assertEqual(initial["manifest"], self.episode["manifest"])
        self.assertEqual(initial["remaining_steps"], 100)
        self.assertEqual(info, {})
        self.assertEqual(set(initial), {"task", "manifest", "policy", "remaining_steps"})

        cursor, records = None, []
        while True:
            result, reward, term, trunc, info = self.env.step(_action("list_records", table="ledger", cursor=cursor, limit=50))
            json.dumps(result, allow_nan=False)
            self.assertEqual((reward, term, trunc, info), (0, False, False, {}))
            self.assertIsNone(result["error"])
            records.extend(result["result"]["records"])
            cursor = result["result"]["next_cursor"]
            if cursor is None:
                break
            self.assertIsInstance(cursor, str)
        self.assertEqual(records, self.episode["tables"]["ledger"])
        row = records[0]
        result = self.env.step(_action("get_record", table="ledger", id=row["row_id"]))[0]
        self.assertEqual(result["result"], row)
        document = self.episode["tables"]["invoices"][0]
        result = self.env.step(_action("get_document", doc_id=document["doc_id"]))[0]
        self.assertEqual(result["result"], document)
        result = self.env.step(_action("calculator", expression="0.1+0.2"))[0]
        self.assertEqual(result["result"], "0.3")
        error = self.env.step(_action("get_document", doc_id="missing"))[0]
        self.assertEqual(error["error"]["code"], "not_found")
        self.assertEqual(error["remaining_steps"], result["remaining_steps"] - 1)

    def test_renamed_fixture_scores_and_keeps_closed_feedback_private(self):
        transformed = copy.deepcopy(self.episode)
        transformed["manifest"]["episode_id"] = "judge-transformed"
        ledger = transformed["tables"]["ledger"]
        ledger.reverse()
        ids = {row["row_id"]: f"judge-row-{index}" for index, row in enumerate(ledger)}
        for row in ledger:
            row["row_id"] = ids[row["row_id"]]
        issue = transformed["truth"][0]
        issue["affected_ids"] = [ids.get(value, value) for value in issue["affected_ids"]]
        issue["acceptable_evidence_sets"] = [[ids.get(value, value) for value in group]
                                             for group in issue["acceptable_evidence_sets"]]
        directory = self.directory.with_name("transformed")
        write_episode(transformed, directory, directory)

        outcomes = []
        for episode, path in ((self.episode, self.directory), (transformed, directory)):
            submission = {"episode_id": episode["manifest"]["episode_id"], "findings": [_finding(episode["truth"][0])]}
            config = {**self.config, "episode_dir": str(path)}
            for mode, expected_reward in (("closed", 0), ("train", 0.998)):
                initial, _ = self.env.reset(0, {**config, "mode": mode})
                self.assertEqual(initial["manifest"]["episode_id"], submission["episode_id"])
                result = self.env.step(_action("list_records", table="ledger", cursor=None, limit=1))[0]
                self.assertEqual(result["result"]["records"], episode["tables"]["ledger"][:1])
                result, reward, term, trunc, info = self.env.step(_action("submit", findings=submission))
                self.assertEqual((result["result"], reward, term, trunc, info),
                                 ({"accepted": True}, expected_reward, True, False, {}))
                self.assertEqual(self.env.step(_action("submit", findings={}))[0]["error"]["code"], "episode_finished")
            outcomes.append(judge_adapter.score_episode(str(path), submission, 1))
        self.assertEqual(outcomes, [{"tp": 1, "fp": 0, "fn": 0, "precision": 1, "recall": 1,
                                     "f1": 1, "return": 0.998}] * 2)

    def test_matching_and_invalid_submission(self):
        issue = self.episode["truth"][0]
        finding = _finding(issue)
        directory = str(self.directory)
        unknown = {**finding, "evidence_ids": [*finding["evidence_ids"], "missing"]}
        result = judge_adapter.score_episode(directory, {"findings": [unknown]}, 10)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

        a = {**issue, "issue_id": "A"}
        extra_row = next(row["row_id"] for row in self.episode["tables"]["ledger"]
                         if row["row_id"] not in finding["evidence_ids"])
        b = {**issue, "issue_id": "B", "acceptable_evidence_sets": [[extra_row]]}
        self.episode["truth"] = [b, a]
        write_episode(self.episode, self.directory, self.directory)
        flexible = {**finding, "evidence_ids": [*finding["evidence_ids"], extra_row]}
        result = judge_adapter.score_episode(directory, {"findings": [flexible, _finding(a)]}, 10)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        result = judge_adapter.score_episode(directory, {"findings": [flexible, flexible]}, 10)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(_canonical_matching([[0, 1], [0, 1]]), [(0, 0), (1, 1)])

        invalid = {"findings": [finding] * 31}
        with self.assertLogs("judge_adapter", level="WARNING"):
            result = judge_adapter.score_episode(directory, invalid, 10)
        self.assertEqual(result, {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                                  "f1": None, "return": -1})
        self.env.reset(0, self.config)
        with self.assertLogs("judge_adapter", level="WARNING"):
            obs, reward, term, trunc, info = self.env.step(_action("submit", findings=invalid))
        self.assertEqual((obs["error"]["code"], reward, term, trunc, info),
                         ("invalid_submission", 0, True, False, {}))

    def test_last_action_and_timeout_cover_the_judge_reset(self):
        self.env.reset(0, self.config)
        for _ in range(99):
            self.env.step(_action("unknown"))
        obs, _, term, trunc, _ = self.env.step(_action("submit", findings={"findings": []}))
        self.assertEqual((obs["remaining_steps"], term, trunc), (0, True, False))
        self.env.reset(0, self.config)
        for _ in range(100):
            obs, _, term, trunc, _ = self.env.step(_action("unknown"))
        self.assertEqual((obs["remaining_steps"], term, trunc), (0, False, True))

        with patch("fin_audit_rl.env.time.monotonic", side_effect=[0, 61]):
            self.env.reset(0, self.config)
            obs, reward, term, trunc, info = self.env.step(_action("submit", findings={"findings": []}))
        self.assertEqual((obs["result"], reward, term, trunc, info), (None, 0, False, True, {}))
