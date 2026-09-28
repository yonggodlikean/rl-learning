"""Tests for the RL checkpoint bank + server (stdlib unittest only).

Run:  python3 -m unittest -v test_server.py
"""

import http.client
import json
import os
import runpy
import threading
import unittest

import bank
import server

HERE = os.path.dirname(os.path.abspath(__file__))


class BankTests(unittest.TestCase):
    def test_totals(self):
        self.assertEqual(bank.TOTAL_SCORE, 100)
        self.assertEqual(
            bank.SCORING["objective"] + bank.SCORING["open_self_review"]
            + bank.SCORING["code_browser_tests"],
            100,
        )
        self.assertEqual(len(bank.QUESTIONS), 12)

    def test_question_mix(self):
        counts = {}
        for q in bank.QUESTIONS:
            counts[q["grading_mode"]] = counts.get(q["grading_mode"], 0) + 1
        self.assertEqual(counts["auto"], 8)
        self.assertEqual(counts["self"], 2)
        self.assertEqual(counts["browser"], 2)

    def test_public_has_no_answer_leakage(self):
        payload = bank.public_payload()
        blob = json.dumps(payload, ensure_ascii=False)
        for secret in ("_answer", "_expected", "_solution", "model_answer", "rubric"):
            self.assertNotIn('"%s":' % secret, blob, "secret key %r leaked into public payload" % secret)
        for q in payload["questions"]:
            self.assertNotIn("_answer", q)
            self.assertNotIn("_expected", q)
            self.assertNotIn("_solution", q)
            self.assertNotIn("model_answer", q)

    def test_code_questions_expose_public_tests(self):
        for q in bank.QUESTIONS:
            if q["type"] != "code":
                continue
            pub = bank.public_question(q)
            self.assertIn("starter_code", pub)
            self.assertIn("entry_point", pub)
            self.assertTrue(pub["tests"])
            for t in pub["tests"]:
                self.assertIn("code", t)
                self.assertIn("description", t)

    def test_industry_case_data_matches_return_recursion(self):
        case = bank.public_payload()["industry_case"]
        self.assertEqual(case["commit"], "a2ad9f6")
        self.assertEqual(len(case["stages"]), 7)
        self.assertEqual(len(case["samples"]), 4)
        for stage in case["stages"]:
            self.assertNotIn("verl/verl/workers/reward_manager/naive.py", stage["path"])
        results = {}
        for sample in case["samples"]:
            self.assertEqual(sample["data_source"], "openai/gsm8k")
            self.assertEqual("".join(sample["positions"][:sample["valid_length"]]),
                             sample["response"])
            length = sample["valid_length"]
            rewards = [0.0] * len(sample["positions"])
            rewards[length - 1] = sample["score"]
            running = 0.0
            for reward in reversed(rewards[:length]):
                running = reward + 0.9 * running
            results[sample["id"]] = running
        self.assertAlmostEqual(results["A"], 0.729)
        self.assertAlmostEqual(results["B"], 0.9)
        self.assertEqual(results["C"], 0.0)
        self.assertEqual(results["D"], 0.0)

    def test_industry_case_matches_verl_source_when_present(self):
        """The published project need not vendor an unrelated 3rd-party tree."""
        case = bank.public_payload()["industry_case"]
        source_root = os.path.dirname(HERE)
        scorer_path = os.path.join(source_root, "verl/verl/utils/reward_score/gsm8k.py")
        if not os.path.isfile(scorer_path):
            self.skipTest("optional external verl checkout is not present")
        scorer = runpy.run_path(scorer_path)["compute_score"]
        for stage in case["stages"]:
            self.assertTrue(os.path.isfile(os.path.join(source_root, stage["path"])))
        for sample in case["samples"]:
            self.assertEqual(
                scorer(sample["response"], sample["ground_truth"]),
                sample["score"],
                sample["id"],
            )

    def test_choice_grading_forms(self):
        self.assertTrue(bank.grade("q03", 0)["correct"])
        self.assertTrue(bank.grade("q03", "A")["correct"])
        self.assertFalse(bank.grade("q03", "B")["correct"])
        self.assertEqual(bank.grade("q03", "B")["score"], 0)
        self.assertEqual(bank.grade("q03", "A")["score"], 5)
        self.assertEqual(bank.grade("q03", "A")["max_score"], 5)

    def test_numeric_tolerance_and_forms(self):
        self.assertTrue(bank.grade("q05", 3.25)["correct"])
        self.assertTrue(bank.grade("q05", 3.2500000001)["correct"])
        self.assertTrue(bank.grade("q05", "3.25")["correct"])
        self.assertFalse(bank.grade("q05", 4.0)["correct"])
        self.assertTrue(bank.grade("q08", [2.5, 3.75])["correct"])
        self.assertFalse(bank.grade("q08", [2.5, 3.7])["correct"])
        self.assertTrue(bank.grade("q07", 3.7)["correct"])
        self.assertTrue(bank.grade("q06", 9.2)["correct"])

    def test_numeric_answer_count_mismatch(self):
        with self.assertRaises(bank.InvalidAnswer):
            bank.grade("q08", [2.5])

    def test_non_finite_rejected(self):
        with self.assertRaises(bank.InvalidAnswer):
            bank.grade("q05", float("nan"))
        with self.assertRaises(bank.InvalidAnswer):
            bank.grade("q05", float("inf"))

    def test_grade_rejects_open_and_code(self):
        with self.assertRaises(bank.NotAutoGradable):
            bank.grade("q09", "some prose")
        with self.assertRaises(bank.NotAutoGradable):
            bank.grade("q11", "def compute_return(): ...")

    def test_unknown_question(self):
        with self.assertRaises(bank.UnknownQuestion):
            bank.get_question("nope")
        with self.assertRaises(bank.UnknownQuestion):
            bank.grade("nope", 0)

    def test_reveal_open(self):
        out = bank.reveal("q09")
        self.assertTrue(out["self_assessment"])
        self.assertTrue(out["rubric"])
        self.assertEqual(sum(r["points"] for r in out["rubric"]), out["max_score"])
        self.assertTrue(out["model_answer"])

    def test_reveal_code_includes_solution(self):
        out = bank.reveal("q12")
        self.assertFalse(out["self_assessment"])
        self.assertIn("reference_solution", out)
        self.assertEqual(out["entry_point"], "bellman_backup")

    def test_reveal_rejects_objective(self):
        with self.assertRaises(bank.NotRevealable):
            bank.reveal("q01")

    def test_restart_rubric_points_match_max(self):
        for q in bank.QUESTIONS:
            if q.get("rubric"):
                self.assertEqual(
                    sum(r["points"] for r in q["rubric"]),
                    q["max_score"],
                    "rubric points != max_score for %s" % q["id"],
                )


class StaticPathTests(unittest.TestCase):
    def test_public_files_allowed(self):
        for name in ("index.html", "app.js", "worker.js", "styles.css"):
            self.assertIsNotNone(server.safe_static_path(HERE, "/" + name))
        self.assertIsNotNone(server.safe_static_path(HERE, "/"))

    def test_vendor_pyodide_allowed(self):
        self.assertIsNotNone(
            server.safe_static_path(HERE, "/vendor/pyodide/pyodide.js")
        )
        self.assertIsNotNone(
            server.safe_static_path(HERE, "/vendor/katex/katex.min.css")
        )

    def test_traversal_and_cross_root_rejected(self):
        for bad in (
            "/../EasyRL_v1.0.6.pdf",
            "/../book.pdf",
            "/..%2fbook.pdf",
            "/vendor/../bank.py",
            "/bank.py",
            "/test_server.py",
            "/../",
            "/etc/passwd",
            "/vendor/pyodide/../../bank.py",
            "/vendor/katex/../../bank.py",
        ):
            self.assertIsNone(server.safe_static_path(HERE, bad), bad)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.build_server("127.0.0.1", 0, HERE)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def _conn(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)

    def _get(self, path):
        conn = self._conn()
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def _post(self, path, obj):
        conn = self._conn()
        body = json.dumps(obj).encode("utf-8")
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, json.loads(data.decode("utf-8"))

    def test_health(self):
        status, data = self._get("/api/health")
        self.assertEqual(status, 200)
        obj = json.loads(data.decode("utf-8"))
        self.assertEqual(obj["status"], "ok")
        self.assertEqual(obj["total_score"], 100)

    def test_questions_no_leakage_over_http(self):
        status, data = self._get("/api/questions")
        self.assertEqual(status, 200)
        blob = data.decode("utf-8")
        for secret in ("_answer", "_expected", "_solution", "model_answer"):
            self.assertNotIn('"%s":' % secret, blob)
        obj = json.loads(blob)
        self.assertEqual(obj["total_score"], 100)
        self.assertEqual(len(obj["questions"]), 12)

    def test_grade_ok(self):
        status, obj = self._post("/api/grade", {"id": "q02", "answer": 2})
        self.assertEqual(status, 200)
        self.assertTrue(obj["correct"])
        self.assertEqual(obj["max_score"], 5)

    def test_grade_wrong_number(self):
        status, obj = self._post("/api/grade", {"id": "q06", "answer": 5.0})
        self.assertEqual(status, 200)
        self.assertFalse(obj["correct"])
        self.assertEqual(obj["score"], 0)

    def test_grade_open_returns_422(self):
        status, obj = self._post("/api/grade", {"id": "q09", "answer": "hi"})
        self.assertEqual(status, 422)
        self.assertIn("error", obj)

    def test_grade_unknown_returns_404(self):
        status, obj = self._post("/api/grade", {"id": "zzz", "answer": 0})
        self.assertEqual(status, 404)

    def test_grade_missing_fields_returns_400(self):
        status, obj = self._post("/api/grade", {"id": "q01"})
        self.assertEqual(status, 400)

    def test_reveal_open_ok(self):
        status, obj = self._post("/api/reveal", {"id": "q10"})
        self.assertEqual(status, 200)
        self.assertTrue(obj["self_assessment"])
        self.assertTrue(obj["rubric"])

    def test_reveal_objective_returns_422(self):
        status, obj = self._post("/api/reveal", {"id": "q04"})
        self.assertEqual(status, 422)

    def test_unknown_api_404(self):
        status, _ = self._get("/api/bogus")
        self.assertEqual(status, 404)

    def test_static_traversal_over_http(self):
        conn = self._conn()
        conn.request("GET", "/../EasyRL_v1.0.6.pdf")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 404)

    def test_vendor_asset_served(self):
        status, data = self._get("/vendor/pyodide/pyodide.js")
        self.assertEqual(status, 200)
        self.assertGreater(len(data), 0)

    def test_frontend_entry_is_served(self):
        status, data = self._get("/index.html")
        self.assertEqual(status, 200)
        self.assertIn(b"/app.js", data)


class SyntaxSanityTests(unittest.TestCase):
    """The two coding exercises must be importable/executable pure Python."""

    def test_compute_return_solution(self):
        ns = {}
        exec(bank.get_question("q11")["_solution"], ns)
        f = ns["compute_return"]
        self.assertEqual(f([], 0.9), 0.0)
        self.assertAlmostEqual(f([1, 2, 3, 4], 0.5), 3.25)

    def test_bellman_backup_solution(self):
        ns = {}
        exec(bank.get_question("q12")["_solution"], ns)
        f = ns["bellman_backup"]
        out = f([1.0, 2.0], [[0.5, 0.5], [0.25, 0.75]], [2.0, 4.0], 0.5)
        self.assertAlmostEqual(out[0], 2.5)
        self.assertAlmostEqual(out[1], 3.75)

    def test_public_test_snippets_execute(self):
        for qid in ("q11", "q12"):
            q = bank.get_question(qid)
            ns = {}
            exec(q["_solution"], ns)
            for t in q["tests"]:
                exec(t["code"], ns)  # must not raise for the reference solution


if __name__ == "__main__":
    unittest.main()
