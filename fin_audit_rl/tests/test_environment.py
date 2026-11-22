import unittest
import copy
import random
import hashlib
import sys
import tempfile
from decimal import Context, Inexact, localcontext
from pathlib import Path
from unittest.mock import patch

from fin_audit_rl.env import AuditEnv
from fin_audit_rl.generator import generate_episode, ISSUE_TYPES
from fin_audit_rl.matcher import score_submission, terminal_reward
from fin_audit_rl.policies import oracle, simple_reconciler
from fin_audit_rl.runner import run_isolated_episode, _docker_argv


class EnvironmentTests(unittest.TestCase):
    def test_all_four_issue_families_are_oracle_resolvable(self):
        """Все четыре типа ошибок распознаются эталонной политикой."""
        for kind in ISSUE_TYPES:
            env = AuditEnv()
            obs, _ = env.reset(10, {"transactions": 30, "issue_types": [kind]})
            self.assertNotIn("truth", obs)
            oracle(env)
            self.assertEqual(env.evaluator_result()["f1"], 1.0, kind)

    def test_two_interactions_have_two_linked_truths(self):
        """Оба сценария взаимодействия создают две связанные ошибки."""
        for scenario in ("partial_fx", "cutoff_fx"):
            ep = generate_episode(77, {"transactions": 40, "interaction_scenario": scenario})
            self.assertEqual(len(ep["truth"]), 2)
            self.assertTrue(set(ep["truth"][0]["affected_ids"]) &
                            set(ep["truth"][1]["affected_ids"]))

    def test_clean_silence_and_false_accusation(self):
        """Пустой ответ на чистых книгах оценивается выше ложного обвинения."""
        env = AuditEnv()
        env.reset(21, {"transactions": 30, "issue_count": 0})
        env.step({"action": "submit", "episode_id": env.episode["manifest"]["episode_id"], "findings": []})
        silence_reward = env.trajectory[-1]["reward"]
        self.assertEqual(silence_reward, 1.0)
        env.reset(21, {"transactions": 30, "issue_count": 0})
        false = {"type": "duplicate_invoice", "affected_ids": ["bogus"], "claimed_amount_minor": 1,
                 "currency": "USD", "evidence_ids": ["bogus"], "explanation": "без подтверждения"}
        env.step({"action": "submit", "episode_id": env.episode["manifest"]["episode_id"], "findings": [false]})
        self.assertLess(env.trajectory[-1]["reward"], silence_reward)

    def test_oracle_beats_silence_on_error_book(self):
        """Подтверждённые выводы оцениваются выше пустого ответа на ошибочных книгах."""
        env = AuditEnv()
        env.reset(8, {"transactions": 30, "issue_types": ["duplicate_invoice"]})
        env.step({"action": "submit", "episode_id": env.episode["manifest"]["episode_id"], "findings": []})
        silence = env.trajectory[-1]["reward"]
        env.reset(8, {"transactions": 30, "issue_types": ["duplicate_invoice"]})
        oracle(env)
        self.assertGreater(env.trajectory[-1]["reward"], silence)

    def test_simple_reconciler_finds_a_duplicate_invoice(self):
        """Программная сверка обнаруживает дублированный счёт."""
        env = AuditEnv()
        env.reset(8, {"transactions": 30, "issue_types": ["duplicate_invoice"]})
        simple_reconciler(env)
        self.assertEqual(env.evaluator_result()["tp"], 1)

    def test_agent_communicates_only_over_json_and_eval_reward_is_withheld(self):
        """Агент обменивается JSON-сообщениями, награда закрытой оценки сохраняется у оценщика."""
        worker = (
            "import json,sys\n"
            "reset=json.loads(sys.stdin.readline()); obs=reset['observation']\n"
            "assert 'truth' not in json.dumps(obs)\n"
            "print(json.dumps({'action':{'action':'list_records','table':'ledger','cursor':None,'limit':1}}),flush=True)\n"
            "reply=json.loads(sys.stdin.readline()); assert 'reward' not in reply\n"
            "print(json.dumps({'action':{'action':'submit','episode_id':obs['manifest']['episode_id'],'findings':[]}}),flush=True)\n"
        )
        env = AuditEnv(mode="eval")
        with self.assertRaises(ValueError):
            run_isolated_episode(env, [sys.executable, "-I", "-c", worker],
                backend="local", cwd=tempfile.gettempdir())
        result = run_isolated_episode(env, [sys.executable, "-I", "-c", worker],
            seed=44, config={"transactions": 20, "issue_count": 0},
            backend="local", cwd=tempfile.gettempdir(), allow_insecure_local=True)
        self.assertEqual(result["status"], "completed", result["trajectory"])
        self.assertEqual(result["evaluator_metrics"]["f1"], 1.0)
        self.assertNotIn("metrics", result["trajectory"][0]["observation"])

    def test_training_reward_is_delivered_only_through_terminal_protocol(self):
        """Обучающая награда передаётся агенту в итоговом сообщении."""
        worker = (
            "import json,sys\n"
            "reset=json.loads(sys.stdin.readline()); obs=reset['observation']\n"
            "print(json.dumps({'action':{'action':'submit','episode_id':obs['manifest']['episode_id'],'findings':[]}}),flush=True)\n"
            "terminal=json.loads(sys.stdin.readline()); assert terminal['terminated'] and terminal['reward']==1.0\n"
        )
        result = run_isolated_episode(AuditEnv(mode="train"),
            [sys.executable, "-I", "-c", worker], seed=45,
            config={"transactions": 20, "issue_count": 0},
            backend="local", cwd=tempfile.gettempdir())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["reward"], 1.0)

    def test_actual_terminal_protocol_fields_are_mode_specific(self):
        """Поля фактического итогового сообщения соответствуют режиму работы."""
        for mode in ("train", "eval"):
            with self.subTest(mode=mode), patch("fin_audit_rl.runner._AgentProcess") as process:
                env = AuditEnv(mode=mode)
                config = {"issue_count": 0}
                _, info = env.reset(45, config)
                process.return_value.receive.side_effect = [
                    {"action": "list_records", "table": "ledger", "limit": 1},
                    {"action": "submit", **info, "findings": []},
                ]
                result = run_isolated_episode(env, [sys.executable], seed=45, config=config,
                                              backend="local", allow_insecure_local=True)
                reset, inspection, terminal = [call.args[0] for call in process.return_value.send.call_args_list]
                self.assertEqual(set(reset), {"event", "observation"})
                fields = {"event", "observation", "terminated", "truncated"}
                if mode == "train":
                    fields.add("reward")
                    self.assertEqual(inspection["reward"], 0)
                    self.assertEqual(terminal["reward"], 0.998)
                self.assertEqual(set(inspection), fields)
                self.assertEqual(set(terminal), fields)
                self.assertEqual(terminal["observation"], {"accepted": True, "steps_remaining": 98})
                self.assertTrue(terminal["terminated"])
                self.assertEqual(result["reward"], 0.998)

    def test_docker_smoke_agent_protocol_runs_locally_for_debugging(self):
        """Пример контейнерного агента поддерживает локальную отладку протокола."""
        agent_dir = Path(__file__).resolve().parents[2] / "examples" / "docker_agent"
        result = run_isolated_episode(AuditEnv(mode="eval"),
            [sys.executable, "-I", "agent.py"], seed=2468,
            config={"transactions": 20, "issue_count": 0}, backend="local",
            cwd=agent_dir, allow_insecure_local=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evaluator_metrics"]["f1"], 1.0)

    def test_docker_backend_is_networkless_and_mounts_only_solution_readonly(self):
        """Параметры Docker отключают сеть и монтируют каталог агента для чтения."""
        with tempfile.TemporaryDirectory() as solution:
            args = _docker_argv(["python", "agent.py"], "audit-agent:offline", solution)
        self.assertIn("--network", args)
        self.assertIn("none", args)
        self.assertIn("--pull=never", args)
        self.assertIn("--read-only", args)
        self.assertIn("--cap-drop=ALL", args)
        self.assertIn("--mount", args)
        mount = args[args.index("--mount") + 1]
        self.assertIn("target=/solution,readonly", mount)
        self.assertNotIn("truth.json", mount)

    def test_duplicate_prediction_does_not_add_credit(self):
        """Повторный вывод учитывается как ложное срабатывание."""
        ep = generate_episode(8, {"transactions": 30, "issue_types": ["duplicate_invoice"]})
        t = ep["truth"][0]
        f = {"type": t["type"], "affected_ids": t["affected_ids"],
             "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
             "evidence_ids": t["acceptable_evidence_sets"][0], "explanation": ""}
        once = score_submission([f], ep["truth"])
        twice = score_submission([f, f], ep["truth"])
        self.assertEqual(once["tp"], twice["tp"])
        self.assertEqual(twice["fp"], 1)
        self.assertLess(twice["f1"], once["f1"])

    def test_evidence_currency_amount_and_ids_are_required(self):
        """Совпадение требует доказательств, валюты, суммы и идентификаторов."""
        ep = generate_episode(8, {"transactions": 30, "issue_types": ["duplicate_invoice"]})
        t = ep["truth"][0]
        good = {"type": t["type"], "affected_ids": t["affected_ids"],
                "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
                "evidence_ids": t["acceptable_evidence_sets"][0], "explanation": ""}
        for changes in ({"evidence_ids": ["guessed"]}, {"currency": "EUR"},
                        {"claimed_amount_minor": good["claimed_amount_minor"] + 1},
                        {"affected_ids": ["guessed"]}):
            f = {**good, **changes}
            self.assertEqual(score_submission([f], ep["truth"])["tp"], 0)

    def test_eval_withholds_terminal_diagnostics(self):
        """Закрытый режим сохраняет итоговую диагностику на стороне оценщика."""
        env = AuditEnv(mode="eval")
        env.reset(8, {"transactions": 30, "issue_types": ["duplicate_invoice"]})
        issue = env.episode["truth"][0]
        finding = {"type": issue["type"], "affected_ids": issue["affected_ids"],
                   "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
                   "evidence_ids": issue["acceptable_evidence_sets"][0], "explanation": ""}
        _, reward, terminated, truncated, info = env.step({"action": "submit", "episode_id": env.episode["manifest"]["episode_id"], "findings": [finding]})
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0)
        self.assertEqual(env.evaluator_reward, 1)
        self.assertEqual(env.evaluator_result()["f1"], 1)
        self.assertNotIn("metrics", info)
        self.assertIsNone(env.trajectory[-1]["observation"].get("truth"))

    def test_pagination_calculator_and_limits(self):
        """Страницы, калькулятор и ограничения запросов работают по контракту."""
        env = AuditEnv()
        obs, _ = env.reset(4, {"transactions": 30})
        self.assertEqual(obs["steps_remaining"], 100)
        page, *_ = env.step({"action": "list_records", "table": "ledger", "limit": 50})
        self.assertEqual(len(page["records"]), 50)
        self.assertIsNotNone(page["next_cursor"])
        result, *_ = env.step({"action": "calculator", "expression": "(2 + 3) * 4"})
        self.assertEqual(result["result"], 20)
        bad, *_ = env.step({"action": "calculator", "expression": "__import__('os').system('id')"})
        self.assertIn("error", bad)
        oversized, *_ = env.step({"action": "list_records", "table": "ledger", "limit": 51})
        self.assertIn("error", oversized)

    def test_bad_submission_and_timeout(self):
        """Неверная отправка и исчерпание шагов корректно завершают эпизод."""
        env = AuditEnv(max_steps=4)
        env.reset(5, {"transactions": 20, "issue_count": 0})
        _, reward, term, trunc, _ = env.step({"action": "submit", "episode_id": env.episode["manifest"]["episode_id"], "findings": [{"not": "valid"}]})
        self.assertTrue(term)
        self.assertEqual(reward, -1)
        self.assertFalse(trunc)
        env.reset(5, {"transactions": 20, "issue_count": 0})
        for _ in range(4):
            _, _, term, trunc, _ = env.step({"action": "unknown"})
        self.assertFalse(term)
        self.assertTrue(trunc)

    def test_reward_cost_is_small_and_clipped(self):
        """Стоимость действий мала, итоговая награда ограничена диапазоном."""
        perfect = {"f1": 1.0}
        self.assertAlmostEqual(terminal_reward(perfect, 0), 1.0)
        self.assertAlmostEqual(terminal_reward(perfect, 10), 0.98)
        self.assertEqual(terminal_reward({"f1": 0}, 500), -1.0)

    def test_matcher_uses_maximum_cardinality_for_ambiguous_edges(self):
        """Неоднозначные выводы сопоставляются с максимальным числом совпадений."""
        truth = [
            {"type": "fx_date_rate", "affected_ids": ["x"], "expected_amount_minor": 10,
             "currency": "USD", "acceptable_evidence_sets": [["proof"]]},
            {"type": "fx_date_rate", "affected_ids": ["x"], "expected_amount_minor": 12,
             "currency": "USD", "acceptable_evidence_sets": [["proof"]]},
        ]
        findings = [
            {"type": "fx_date_rate", "affected_ids": ["x"], "claimed_amount_minor": 10,
             "currency": "USD", "evidence_ids": ["proof"], "explanation": ""},
            {"type": "fx_date_rate", "affected_ids": ["x"], "claimed_amount_minor": 8,
             "currency": "USD", "evidence_ids": ["proof"], "explanation": ""},
        ]
        metrics = score_submission(findings, truth, amount_tolerance_minor=2)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 0, 0))

    def test_unavailable_evidence_prevents_matching_even_with_complete_proof(self):
        """Несуществующее доказательство исключает совпадение даже при полном наборе."""
        truth_core = {"type": "fx_date_rate", "affected_ids": ["row", "invoice"],
                      "expected_amount_minor": 10, "currency": "USD"}
        truth = [{**truth_core, "acceptable_evidence_sets": [[proof]]} for proof in ("one", "two")]
        finding = {"type": "fx_date_rate", "affected_ids": ["invoice", "row"],
                   "claimed_amount_minor": 10, "currency": "USD",
                   "evidence_ids": ["one", "two"], "explanation": ""}
        unsupported = {**finding, "evidence_ids": ["padding"]}
        padded = {**finding, "affected_ids": ["row", "invoice"],
                  "evidence_ids": ["two", "padding", "one"], "explanation": "другое обоснование"}
        metrics = score_submission([unsupported, finding, padded], truth,
                                   available_evidence_ids={"one", "two"})
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 2, 1))

    def test_generated_books_remain_balanced_by_currency(self):
        """Сгенерированные книги сбалансированы по валютам."""
        for seed in range(20):
            ep = generate_episode(seed, {"transactions": 30, "issue_count": seed % 5})
            totals = {}
            for row in ep["tables"]["ledger"]:
                pair = totals.setdefault(row["currency"], [0, 0])
                pair[0] += row["debit_minor"]
                pair[1] += row["credit_minor"]
            self.assertTrue(all(debit == credit for debit, credit in totals.values()), seed)

    def test_seed_and_actions_reproduce_observations_and_reward(self):
        """Начальное значение генератора и действия воспроизводят наблюдения и награду."""
        traces = []
        for _ in range(2):
            env = AuditEnv()
            env.reset(999, {"transactions": 30, "issue_types": ["period_cutoff"]})
            env.step({"action": "list_records", "table": "ledger", "cursor": None, "limit": 7})
            oracle(env)
            traces.append(env.trajectory)
        self.assertEqual(traces[0], traces[1])

    def test_twenty_paired_order_and_id_permutations_preserve_oracle_score(self):
        """Двадцать повторных и преобразованных случаев сохраняют результаты."""
        for seed in range(20):
            config = ({"transactions": 30, "issue_count": 0} if seed < 5 else
                      {"transactions": 30, "interaction_scenario": ("partial_fx" if seed % 2 else "cutoff_fx")}
                      if seed < 10 else
                      {"transactions": 30, "issue_types": [ISSUE_TYPES[seed % 4]]})
            original = generate_episode(500 + seed, config)
            regenerated = generate_episode(500 + seed, config)
            self.assertEqual(regenerated, original, seed)
            transformed = copy.deepcopy(original)
            id_values = set()
            def collect(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if (k.endswith("_id") or k == "supplier_invoice_ref") and isinstance(v, str):
                            id_values.add(v)
                        collect(v)
                elif isinstance(value, list):
                    for v in value: collect(v)
            collect(transformed["manifest"])
            collect(transformed["tables"])
            collect(transformed["truth"])
            renames = {old: "R-" + hashlib.sha256(f"{seed}:{old}".encode()).hexdigest()[:12]
                       for old in id_values}
            def rewrite(value):
                if isinstance(value, dict):
                    return {k: rewrite(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [rewrite(v) for v in value]
                return renames.get(value, value) if isinstance(value, str) else value
            transformed["tables"] = rewrite(transformed["tables"])
            transformed["truth"] = rewrite(transformed["truth"])
            transformed["manifest"] = rewrite(transformed["manifest"])
            for rows in transformed["tables"].values():
                random.Random(seed).shuffle(rows)
            scores = []
            traces = []
            for episode in (original, regenerated, transformed):
                env = AuditEnv()
                outcomes, episode_traces = [], []
                for policy in (oracle, simple_reconciler):
                    with patch("fin_audit_rl.env.generate_episode", return_value=episode):
                        env.reset(900 + seed, {"transactions": 30})
                    policy(env, seed)
                    self.assertEqual(env.evaluator_result()["f1"], 1.0, seed)
                    outcomes.append((env.evaluator_result()["f1"], env.evaluator_reward))
                    episode_traces.append(env.trajectory)
                scores.append(outcomes)
                traces.append(episode_traces)
            self.assertEqual(traces[0], traces[1], seed)
            self.assertEqual(scores[0], scores[2], seed)

    def test_binary_and_nonfinite_calculator_inputs_are_rejected(self):
        """Калькулятор проверяет допустимость выражений, диапазон и точность."""
        env = AuditEnv()
        for expression in ("True + 1", "0xff + 1", "1e999 - 1e999", "1/0", "1+2 # комментарий",
                           "1000000000000000000.0000000001"):
            obs, reward, *_ = env.step({"action": "calculator", "expression": expression})
            self.assertIn("error", obs, expression)
            self.assertEqual(reward, 0)
        for expression, expected in (("(0.1 + 0.2) * 100", 30), ("-5 // 2", -3), ("-5 % 2", 1)):
            self.assertEqual(env.step({"action": "calculator", "expression": expression})[0]["result"], expected)
        with localcontext(Context(prec=2, traps=[Inexact])):
            self.assertAlmostEqual(env.step({"action": "calculator", "expression": "1/3"})[0]["result"], 1/3)

    def test_document_lookup_and_observation_mutation(self):
        """Документы доступны по идентификаторам, изменение ответа сохраняет исходные данные."""
        env = AuditEnv()
        obs, _ = env.reset(3)
        original_id = obs["manifest"]["episode_id"]
        obs["manifest"]["episode_id"] = "altered"
        self.assertEqual(env.initial_observation["manifest"]["episode_id"], original_id)
        for table in ("invoices", "payments", "fx"):
            page, *_ = env.step({"action": "list_records", "table": table, "limit": 1})
            expected = copy.deepcopy(page["records"][0])
            page["records"][0].clear()
            request = ({"action": "get_record", "table": "fx", "id": expected["rate_id"]} if table == "fx"
                       else {"action": "get_document", "doc_id": expected["doc_id"]})
            self.assertEqual(env.step(request)[0]["record"], expected)
            self.assertEqual(env.trajectory[-2]["observation"]["records"], [expected])
            if table != "fx":
                explicit_id = expected["invoice_id" if table == "invoices" else "payment_id"]
                self.assertEqual(env.step({"action": "get_record", "table": table,
                                           "id": explicit_id})[0]["record"], expected)

    def test_last_step_submit_and_late_submit(self):
        """Отправка на последнем шаге и после истечения времени обрабатывается корректно."""
        env = AuditEnv(max_steps=2)
        obs, _ = env.reset(10, {"issue_count": 0})
        submit = {"action": "submit", "episode_id": obs["manifest"]["episode_id"], "findings": []}
        env.step({"action": "list_records", "table": "ledger", "limit": True})
        self.assertEqual(env.step(submit)[1:4], (0.998, True, False))
        self.assertEqual(env.step(submit)[1:4], (0, True, False))
        self.assertEqual(len(env.trajectory), 2)
        env.max_seconds = 0
        env.reset(10, {"issue_count": 1})
        obs, reward, terminated, truncated, _ = env.step(submit)
        self.assertEqual(obs["final_answer"], [])
        self.assertEqual((reward, terminated, truncated), (-1.0, False, True))
        self.assertEqual(env.step(submit)[1:4], (0, False, True))

    def test_closed_invalid_submission_has_private_penalty(self):
        """Штраф за неверную отправку в закрытом режиме доступен оценщику."""
        env = AuditEnv(mode="eval")
        obs, _ = env.reset(23, {"issue_count": 0})
        action = {"action": "submit", "episode_id": obs["manifest"]["episode_id"],
                  "findings": [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": True,
                                "currency": "USD", "evidence_ids": ["b"], "explanation": ""}]}
        _, reward, terminated, truncated, info = env.step(action)
        self.assertEqual((reward, terminated, truncated), (0.0, True, False))
        self.assertEqual(info, {"evaluator_result_withheld": True})
        self.assertEqual(env.evaluator_reward, -1.0)
        self.assertEqual(env.evaluator_result()["evaluator_error"], "неверная структура отправленных выводов")

    def test_runner_timeout_and_invalid_protocol_truncate_consistently(self):
        """Ограничение времени и ошибка протокола корректно прерывают запуск."""
        for worker, max_seconds, status in (("import time; time.sleep(3)", 0.2, "agent_timeout"),
                                            ("print('invalid')", 2, "agent_protocol_error")):
            env = AuditEnv(mode="eval", max_seconds=max_seconds)
            result = run_isolated_episode(env, [sys.executable, "-I", "-c", worker], backend="local",
                config={"issue_count": 1}, allow_insecure_local=True)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["reward"], -1.0)
            self.assertEqual(result["trajectory"][-1]["reward"], 0.0)
            self.assertEqual(result["trajectory"][-1]["observation"]["final_answer"], [])
            self.assertTrue(env.done and env.truncated)


if __name__ == "__main__":
    unittest.main()
