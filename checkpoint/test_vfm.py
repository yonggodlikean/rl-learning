"""Independent verification, provenance, arithmetic, graders and code tests for revised VFM chapter."""

import math
import unittest
import numpy as np

import bank
import chapter_vfm as vfm


class VfmChapterTests(unittest.TestCase):
    def test_total_count_and_structure(self):
        questions = vfm.list_questions()
        self.assertEqual(len(questions), 21)
        self.assertEqual(vfm.TOTAL_SCORE, 108)
        self.assertEqual(vfm.SCORING["objective"], 54)
        self.assertEqual(vfm.SCORING["open_self_review"], 24)
        self.assertEqual(vfm.SCORING["code_browser_tests"], 30)

    def test_metadata_fields(self):
        for q in vfm.list_questions():
            self.assertIn("id", q)
            self.assertIn("type", q)
            self.assertIn("learning_goal", q)
            self.assertIn("source_note", q)
            self.assertIn("review_hint", q)

    def test_objective_grading(self):
        # Choice question q01 - B
        res_q01 = vfm.grade("q01", "B")
        self.assertTrue(res_q01["correct"])
        self.assertEqual(res_q01["score"], 4)

        # Choice question q02 - B
        res_q02 = vfm.grade("q02", "B")
        self.assertTrue(res_q02["correct"])

        # Choice question q03 - B
        res_q03 = vfm.grade("q03", "B")
        self.assertTrue(res_q03["correct"])

        # Numeric q07 (scalar: 10.0)
        res_q07 = vfm.grade("q07", 10.0)
        self.assertTrue(res_q07["correct"])
        self.assertEqual(res_q07["score"], 5)

        # Numeric q08 (list: 4.6, 1.96)
        res_q08 = vfm.grade("q08", [4.6, 1.96])
        self.assertTrue(res_q08["correct"])

        # Numeric q09 (list: 1.0, 8.92, 7.92)
        res_q09 = vfm.grade("q09", [1.0, 8.92, 7.92])
        self.assertTrue(res_q09["correct"])

        # Numeric q10 (list: 5.32, 4.42)
        res_q10 = vfm.grade("q10", [5.32, 4.42])
        self.assertTrue(res_q10["correct"])

        # Numeric q11 (list: 14.0, 16.0)
        res_q11 = vfm.grade("q11", [14.0, 16.0])
        self.assertTrue(res_q11["correct"])

        # Numeric q12 (scalar: 0.729)
        res_q12 = vfm.grade("q12", 0.729)
        self.assertTrue(res_q12["correct"])

    def test_code_solutions_run_public_tests(self):
        for q in vfm.list_questions():
            if q["type"] == "code":
                env = {}
                exec(q["_solution"], env)
                func_or_class = env[q["entry_point"]]
                self.assertIsNotNone(func_or_class)
                test_list = q.get("tests", q.get("public_tests", []))
                for test in test_list:
                    test_env = dict(env)
                    try:
                        exec(test["code"], test_env)
                    except Exception as exc:
                        self.fail(f"Test '{test['name']}' failed for {q['id']}: {exc}")

    def test_public_payload_hides_private_keys(self):
        payload = vfm.public_payload()
        self.assertEqual(payload["count"], 21)
        self.assertEqual(payload["total_score"], 108)
        for q in payload["questions"]:
            self.assertNotIn("_answer", q)
            self.assertNotIn("_expected", q)
            self.assertNotIn("_solution", q)


if __name__ == "__main__":
    unittest.main()
