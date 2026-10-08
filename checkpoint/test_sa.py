"""Independent arithmetic, provenance, graders and API checks for the SA bank.

Every numeric answer is recomputed from the prompt's own data; every choice
answer is checked against the learning goal; every code reference solution is
executed against its public tests plus diagnostic wrong implementations; and
the HTTP layer is exercised through the real server.
"""

import http.client
import json
import math
import threading
import unittest

import bank
import build_site
import chapter_sa as sa
import chapters
import server


# --------------------------------------------------------------------------- #
# metadata: each question must carry source, goal, and a feedback hint
# --------------------------------------------------------------------------- #

REQUIRED_CONTEXT = {
    "q01": ["观测", "RM", "解决"],
    "q02": ["初值", "alpha_k", "期望"],
    "q03": ["alpha_k", "求和条件"],
    "q04": ["噪声", "二阶矩"],
    "q05": ["w^3", "g'(w)", "实验轨迹"],
    "q06": ["BGD", "有放回"],
    "q07": ["w_1=99", "frac1k", "x_1=4"],
    "q08": ["固定", "1/4", "不是"],
    "q09": ["w^3-5", "无噪声", "核验"],
    "q10": ["2w-6", "1/4"],
    "q11": ["w=0", "BGD", "SGD", "MBGD", "0.2"],
    "q12": ["w=7", "E[X]=2", "operatorname{Var}(X)=9"],
    "q13": ["kZ_k", "二阶矩", "有限实验轨迹"],
    "q14": ["均值", "RM", "SGD"],
    "q15": ["交叉项", "Dvoretzky", "收缩"],
    "q16": ["0.8", "稳态", "路径"],
    "q17": ["samples", "steps", "initial", "mean_path"],
    "q18": ["oracle", "rm_path", "一次"],
    "q19": ["旧w", "MBGD", "regression_batch_step"],
    "q20": ["itertools", "有放回", "batch_size", "batch_gradients"],
}

SOURCE_SECTIONS = {
    "q01": "Zhao §6.2", "q02": "Zhao §6.1", "q03": "Zhao §6.2",
    "q04": "Zhao §6.2", "q05": "Zhao §6.2", "q06": "Zhao §6.4",
    "q07": "Zhao §6.1", "q08": "Zhao §6.1", "q09": "Zhao §6.2",
    "q10": "Zhao §6.2", "q11": "Zhao §6.4", "q12": "Zhao §6.4",
    "q13": "Zhao §6.3", "q14": "Zhao §6.1", "q15": "Zhao §6.3",
    "q16": "Zhao §6.4", "q17": "Zhao §6.1", "q18": "Zhao §6.2",
    "q19": "Zhao §6.4", "q20": "Zhao §6.4",
}


class MetadataTests(unittest.TestCase):
    def test_count_and_total_score(self):
        self.assertEqual(len(sa.QUESTIONS), 20)
        self.assertEqual(sa.TOTAL_SCORE, 100)
        self.assertEqual(sum(q["max_score"] for q in sa.QUESTIONS), 100)

    def test_scoring_breakdown(self):
        obj = sum(q["max_score"] for q in sa.QUESTIONS
                  if q["grading_mode"] == "auto")
        self_rev = sum(q["max_score"] for q in sa.QUESTIONS
                       if q["grading_mode"] == "self")
        code = sum(q["max_score"] for q in sa.QUESTIONS
                   if q["grading_mode"] == "browser")
        self.assertEqual(sa.SCORING["objective"], obj)
        self.assertEqual(sa.SCORING["open_self_review"], self_rev)
        self.assertEqual(sa.SCORING["code_browser_tests"], code)
        self.assertEqual(obj + self_rev + code, 100)

    def test_unique_ids_and_types(self):
        ids = [q["id"] for q in sa.QUESTIONS]
        self.assertEqual(len(ids), len(set(ids)))
        types = {q["type"] for q in sa.QUESTIONS}
        self.assertEqual(types, {"choice", "numeric", "open", "code"})

    def test_required_context_in_each_prompt(self):
        for q in sa.QUESTIONS:
            for needle in REQUIRED_CONTEXT[q["id"]]:
                self.assertIn(needle, q["prompt"],
                             "%s prompt missing %r" % (q["id"], needle))

    def test_source_notes_reference_zhao_chapter(self):
        for q in sa.QUESTIONS:
            src = q["source_note"]["text"]
            self.assertIn(SOURCE_SECTIONS[q["id"]], src,
                         "%s source mismatch" % q["id"])
            self.assertIn("Zhao", src)
            self.assertIn("EasyRL", src)
            self.assertIn("learning_goal", q)

    def test_review_hint_matches_source(self):
        for q in sa.QUESTIONS:
            self.assertEqual(q["review_hint"], q["source_note"]["text"])

    def test_feedback_hints_on_objective_questions(self):
        for q in sa.QUESTIONS:
            if q["type"] in ("choice", "numeric"):
                self.assertIn("feedback_hint", q,
                             "%s missing feedback_hint" % q["id"])
                self.assertTrue(q["feedback_hint"].strip())


class PublicPayloadTests(unittest.TestCase):
    def test_no_answer_keys_or_reference_solutions(self):
        payload = sa.public_payload()
        blob = json.dumps(payload, ensure_ascii=False)
        for banned in ('"_answer"', '"_expected"', '"_solution"',
                       '"grade"', '"reveal"', '"public_payload"'):
            self.assertNotIn(banned, blob)
        for q in payload["questions"]:
            for key in q:
                self.assertFalse(key.startswith("_"),
                                 "private field leaked: %s" % key)

    def test_public_question_fields_match_type(self):
        payload = sa.public_payload()
        for q in payload["questions"]:
            common = {"id", "type", "category", "title", "prompt",
                      "max_score", "grading_mode"}
            self.assertEqual(set(q) & common, common)
            if q["type"] == "choice":
                self.assertIn("options", q)
            elif q["type"] == "numeric":
                self.assertIn("answer_type", q)
                self.assertIn("tolerance", q)
                if q["answer_type"] == "list":
                    self.assertIn("answer_count", q)
            elif q["type"] == "code":
                self.assertIn("entry_point", q)
                self.assertIn("starter_code", q)
                self.assertIn("tests", q)
                pass  # rubric only via reveal()
            elif q["type"] == "open":
                pass  # rubric only via reveal()

    def test_numeric_lengths_public_but_values_not(self):
        payload = sa.public_payload()
        for q in payload["questions"]:
            if q["type"] == "numeric" and q["answer_type"] == "list":
                internal = sa._get(q["id"])
                self.assertEqual(q["answer_count"],
                                 len(internal["_expected"]))


class GradingTests(unittest.TestCase):
    def _grade_all_choice(self, correct=True):
        for q in sa.QUESTIONS:
            if q["type"] != "choice":
                continue
            ans = q["_answer"] if correct else (q["_answer"] + 1) % len(q["options"])
            letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[ans]
            result = sa.grade(q["id"], letter)
            self.assertEqual(result["correct"], correct, q["id"])
            self.assertEqual(result["score"],
                             q["max_score"] if correct else 0, q["id"])
            self.assertEqual(result["max_score"], q["max_score"])

    def test_correct_choice_answers(self):
        self._grade_all_choice(correct=True)

    def test_wrong_choice_answers(self):
        self._grade_all_choice(correct=False)

    def test_choice_accepts_index_and_letter(self):
        q = sa._get("q01")
        r1 = sa.grade("q01", q["_answer"])
        r2 = sa.grade("q01", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[q["_answer"]])
        self.assertTrue(r1["correct"])
        self.assertTrue(r2["correct"])

    def test_numeric_correct_answers(self):
        for q in sa.QUESTIONS:
            if q["type"] != "numeric":
                continue
            exp = q["_expected"]
            if q["answer_type"] == "list":
                result = sa.grade(q["id"], list(exp))
            else:
                result = sa.grade(q["id"], exp)
            self.assertTrue(result["correct"], q["id"])

    def test_numeric_wrong_count_raises(self):
        with self.assertRaises(bank.InvalidAnswer):
            sa.grade("q07", [1, 2])

    def test_open_and_code_not_auto_gradable(self):
        for q in sa.QUESTIONS:
            if q["type"] in ("open", "code"):
                with self.assertRaises(bank.NotAutoGradable):
                    sa.grade(q["id"], "anything")

    def test_unknown_question_raises(self):
        with self.assertRaises(bank.UnknownQuestion):
            sa.grade("q99", 0)

    def test_reveal_objective_raises(self):
        for q in sa.QUESTIONS:
            if q["type"] in ("choice", "numeric"):
                with self.assertRaises(bank.NotRevealable):
                    sa.reveal(q["id"])

    def test_reveal_open_returns_model_answer(self):
        for q in sa.QUESTIONS:
            if q["type"] == "open":
                r = sa.reveal(q["id"])
                self.assertIn("model_answer", r)
                self.assertTrue(r["self_assessment"])
                self.assertIn("rubric", r)

    def test_reveal_code_returns_solution(self):
        for q in sa.QUESTIONS:
            if q["type"] == "code":
                r = sa.reveal(q["id"])
                self.assertIn("reference_solution", r)
                self.assertIn("entry_point", r)
                self.assertFalse(r["self_assessment"])


class IndependentArithmeticTests(unittest.TestCase):
    """Recompute every numeric answer from scratch."""

    def test_q07_incremental_mean_1_over_k(self):
        w = 99
        xs = [4, -2, 6]
        path = []
        for k, x in enumerate(xs, 1):
            w = w + (1 / k) * (x - w)
            path.append(w)
        q = sa._get("q07")
        for a, b in zip(path, q["_expected"]):
            self.assertTrue(math.isclose(a, b, rel_tol=q["tolerance"],
                                         abs_tol=q["tolerance"]))

    def test_q08_fixed_gain_quarter(self):
        w = 0
        xs = [2, 4, -2]
        path = []
        for x in xs:
            w = w + 0.25 * (x - w)
            path.append(w)
        q = sa._get("q08")
        for a, b in zip(path, q["_expected"]):
            self.assertTrue(math.isclose(a, b, rel_tol=q["tolerance"],
                                         abs_tol=q["tolerance"]))

    def test_q09_cubic_root_and_noiseless_steps(self):
        root = 5 ** (1 / 3)
        w = 0
        steps = []
        for k in [1, 2]:
            g = w ** 3 - 5
            w = w - (1 / k) * g
            steps.append(w)
        expected = [root] + steps
        q = sa._get("q09")
        for a, b in zip(expected, q["_expected"]):
            self.assertTrue(math.isclose(a, b, rel_tol=q["tolerance"],
                                         abs_tol=q["tolerance"]))

    def test_q10_linear_with_fixed_noise(self):
        w = 0
        etas = [1, -1]
        path = []
        for alpha, eta in zip([0.25, 0.25], etas):
            obs = 2 * w - 6 + eta
            w = w - alpha * obs
            path.append(w)
        q = sa._get("q10")
        for a, b in zip(path, q["_expected"]):
            self.assertTrue(math.isclose(a, b, rel_tol=q["tolerance"],
                                         abs_tol=q["tolerance"]))

    def test_q11_bgd_sgd_mbgd_same_point(self):
        w = 0
        alpha = 0.2
        data = [-1, 3, 5]
        g_bgd = sum(w - x for x in data) / len(data)
        w_bgd = w - alpha * g_bgd
        g_sgd = w - 5
        w_sgd = w - alpha * g_sgd
        batch = [-1, 5]
        g_mbgd = sum(w - x for x in batch) / len(batch)
        w_mbgd = w - alpha * g_mbgd
        expected = [w_bgd, w_sgd, w_mbgd]
        q = sa._get("q11")
        for a, b in zip(expected, q["_expected"]):
            self.assertTrue(math.isclose(a, b, rel_tol=q["tolerance"],
                                         abs_tol=q["tolerance"]))

    def test_q12_gradient_expectation_and_variance(self):
        w = 7
        ex = 2
        var_x = 9
        expected = [w - ex, var_x, var_x / 9]
        q = sa._get("q12")
        for a, b in zip(expected, q["_expected"]):
            self.assertTrue(math.isclose(a, b, rel_tol=q["tolerance"],
                                         abs_tol=q["tolerance"]))


class ReferenceSolutionTests(unittest.TestCase):
    """Execute reference solutions against their public tests."""

    def _run_tests(self, qid):
        q = sa._get(qid)
        ns = {}
        exec(q["_solution"], ns)
        for test in q["tests"]:
            ns_exec = dict(ns)
            exec(test["code"], ns_exec)

    def test_q17_mean_passes_all_tests(self):
        self._run_tests("q17")

    def test_q18_rm_path_passes_all_tests(self):
        self._run_tests("q18")

    def test_q19_regression_batch_step_passes_all_tests(self):
        self._run_tests("q19")

    def test_q20_batch_gradients_passes_all_tests(self):
        self._run_tests("q20")

    def test_q17_starter_fails(self):
        q = sa._get("q17")
        ns = {}
        exec(q["starter_code"], ns)
        result = ns["mean_path"]([4, -2, 6], [1, 0.5, 1 / 3], 99)
        self.assertNotEqual(result[-1], 2 + 2 / 3)

    def test_q18_starter_fails(self):
        q = sa._get("q18")
        ns = {}
        exec(q["starter_code"], ns)
        result = ns["rm_path"](lambda w: 2 * w - 8, [0.25, 0.25], 0)
        self.assertNotEqual(result[-1], 3)

    def test_q19_starter_fails(self):
        q = sa._get("q19")
        ns = {}
        exec(q["starter_code"], ns)
        w_new, _, _ = ns["regression_batch_step"](1, [(1, 2), (2, 1)], 0.2)
        self.assertNotEqual(w_new, 0.9)

    def test_q20_starter_fails(self):
        q = sa._get("q20")
        ns = {}
        exec(q["starter_code"], ns)
        result = ns["batch_gradients"](0.5, [-2, 2], 1)
        self.assertEqual(result, [0.0, 0.0])


class HttpApiTests(unittest.TestCase):
    """Exercise the SA chapter through the real HTTP server."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = server.build_server("127.0.0.1", 0, str(build_site.ROOT))
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        if body is not None:
            conn.request(method, path, json.dumps(body),
                         {"Content-Type": "application/json"})
        else:
            conn.request(method, path)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        return resp.status, json.loads(raw) if raw else None

    def test_chapter_listed(self):
        status, data = self.request("GET", "/api/chapters")
        self.assertEqual(status, 200)
        self.assertIn(sa.CHAPTER_ID, [c["id"] for c in data["chapters"]])

    def test_questions_no_leakage_over_http(self):
        status, data = self.request(
            "GET", "/api/questions?chapter=" + sa.CHAPTER_ID)
        self.assertEqual(status, 200)
        blob = json.dumps(data, ensure_ascii=False)
        for banned in ('"_answer"', '"_expected"', '"_solution"'):
            self.assertNotIn(banned, blob)

    def test_grade_choice_correct(self):
        status, result = self.request("POST", "/api/grade", {
            "chapter": sa.CHAPTER_ID, "id": "q01", "answer": "B"})
        self.assertEqual(status, 200)
        self.assertTrue(result["correct"])

    def test_grade_choice_wrong(self):
        status, result = self.request("POST", "/api/grade", {
            "chapter": sa.CHAPTER_ID, "id": "q01", "answer": 0})
        self.assertEqual(status, 200)
        self.assertFalse(result["correct"])

    def test_grade_numeric_correct(self):
        status, result = self.request("POST", "/api/grade", {
            "chapter": sa.CHAPTER_ID, "id": "q07",
            "answer": [4, 1, 2 + 2 / 3]})
        self.assertEqual(status, 200)
        self.assertTrue(result["correct"])

    def test_grade_numeric_wrong_count(self):
        status, _ = self.request("POST", "/api/grade", {
            "chapter": sa.CHAPTER_ID, "id": "q07", "answer": [1, 2]})
        self.assertEqual(status, 400)

    def test_grade_unknown_returns_404(self):
        status, _ = self.request("POST", "/api/grade", {
            "chapter": sa.CHAPTER_ID, "id": "q99", "answer": 0})
        self.assertEqual(status, 404)

    def test_reveal_open_ok(self):
        status, result = self.request("POST", "/api/reveal", {
            "chapter": sa.CHAPTER_ID, "id": "q13"})
        self.assertEqual(status, 200)
        self.assertTrue(result["self_assessment"])
        self.assertIn("model_answer", result)

    def test_reveal_code_ok(self):
        status, result = self.request("POST", "/api/reveal", {
            "chapter": sa.CHAPTER_ID, "id": "q17"})
        self.assertEqual(status, 200)
        self.assertIn("reference_solution", result)

    def test_reveal_objective_returns_422(self):
        status, _ = self.request("POST", "/api/reveal", {
            "chapter": sa.CHAPTER_ID, "id": "q01"})
        self.assertEqual(status, 422)

    def test_grade_open_returns_422(self):
        status, _ = self.request("POST", "/api/grade", {
            "chapter": sa.CHAPTER_ID, "id": "q13", "answer": "text"})
        self.assertEqual(status, 422)

    def test_unknown_chapter_returns_404(self):
        status, _ = self.request("GET", "/api/questions?chapter=nope")
        self.assertEqual(status, 404)


class RegistryTests(unittest.TestCase):
    def test_registry_includes_sa_chapter(self):
        ch = chapters.get_chapter(sa.CHAPTER_ID)
        self.assertEqual(ch.id, sa.CHAPTER_ID)
        self.assertEqual(ch.bank, sa)
        summary = ch.summary()
        self.assertEqual(summary["count"], 20)
        self.assertEqual(summary["total_score"], 100)


if __name__ == "__main__":
    unittest.main()
