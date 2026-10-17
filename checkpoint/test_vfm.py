"""Independent verification, provenance, arithmetic, graders and code tests for revised VFM chapter."""

import math
import unittest
import numpy as np

import bank
import chapter_vfm as vfm


class VfmChapterTests(unittest.TestCase):
    def test_total_count_and_structure(self):
        questions = vfm.list_questions()
        self.assertEqual(len(questions), 40)
        self.assertEqual(vfm.TOTAL_SCORE, 190)
        self.assertEqual(vfm.SCORING["objective"], 130)
        self.assertEqual(vfm.SCORING["open_self_review"], 24)
        self.assertEqual(vfm.SCORING["code_browser_tests"], 36)

    def test_metadata_fields(self):
        for q in vfm.list_questions():
            self.assertIn("id", q)
            self.assertIn("type", q)
            self.assertIn("learning_goal", q)
            self.assertIn("source_note", q)
            self.assertIn("review_hint", q)

    def test_objective_grading(self):
        # 20 Choice questions
        choices = [
            ("q01", "B", 4),
            ("q02", "A", 4),
            ("q03", "C", 4),
            ("q05", "A", 4),
            ("q06", "B", 4),
            ("q07", "A", 4),
            ("q09", "B", 4),
            ("q10", "B", 4),
            ("q12", "B", 4),
            ("q13", "B", 4),
            ("q17", "B", 4),
            ("q18", "C", 4),
            ("q19", "B", 4),
            ("q20", "A", 4),
            ("q21", "A", 4),
            ("q30", "A", 4),
            ("q31", "B", 4),
            ("q32", "A", 4),
            ("q33", "A", 4),
            ("q34", "A", 4),
        ]
        for qid, ans, pts in choices:
            res = vfm.grade(qid, ans)
            self.assertTrue(res["correct"], f"Failed choice {qid}: {res}")
            self.assertEqual(res["score"], pts)

        # 10 Numeric questions
        numerics = [
            ("q04", 2.0, 5),
            ("q08", [1.2, 2.0], 5),
            ("q14", [0.72, 1.22], 5),
            ("q15", [2.5, 0.8, 1.3], 5),
            ("q22", 10.0, 5),
            ("q23", [4.6, 1.96], 5),
            ("q24", [1.0, 8.92, 7.92], 5),
            ("q25", 0.729, 5),
            ("q35", [5.32, 4.42], 5),
            ("q36", [14.0, 16.0], 5),
        ]
        for qid, ans, pts in numerics:
            res = vfm.grade(qid, ans)
            self.assertTrue(res["correct"], f"Failed numeric {qid}: {res}")
            self.assertEqual(res["score"], pts)

    def test_code_solutions_run_public_tests(self):
        code_count = 0
        for q in vfm.list_questions():
            if q["type"] == "code":
                code_count += 1
                env = {}
                exec(q["_solution"], env)
                func_or_class = env[q["entry_point"]]
                self.assertIsNotNone(func_or_class)
                test_list = q.get("tests", q.get("public_tests", []))
                self.assertGreater(len(test_list), 0, f"No tests for {q['id']}")
                for test in test_list:
                    test_env = dict(env)
                    try:
                        exec(test["code"], test_env)
                    except Exception as exc:
                        self.fail(f"Test '{test['name']}' failed for {q['id']}: {exc}")
        self.assertEqual(code_count, 6)

    def test_open_review_questions(self):
        open_count = 0
        for q in vfm.list_questions():
            if q["type"] == "open":
                open_count += 1
                rev = vfm.reveal(q["id"])
                self.assertTrue(rev["self_assessment"])
                self.assertIn("rubric", rev)
                self.assertIn("model_answer", rev)
                self.assertGreater(len(rev["rubric"]), 0)
        self.assertEqual(open_count, 4)

    def test_public_payload_hides_private_keys(self):
        payload = vfm.public_payload()
        self.assertEqual(payload["count"], 40)
        self.assertEqual(payload["total_score"], 190)
        for q in payload["questions"]:
            self.assertNotIn("_answer", q)
            self.assertNotIn("_expected", q)
            self.assertNotIn("_solution", q)


if __name__ == "__main__":
    unittest.main()
