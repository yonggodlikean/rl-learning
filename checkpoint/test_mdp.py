"""Independent arithmetic and authoring checks for the real §2.3 bank."""

import copy
import json
import os
import unittest
from pathlib import Path

import bank
import chapter_mdp as mdp


class MdpBankTests(unittest.TestCase):
    def test_counts_and_objectives(self):
        self.assertEqual(mdp.TOTAL_SCORE, 100)
        self.assertEqual(len(mdp.QUESTIONS), 12)
        self.assertEqual(sum(mdp.SCORING.values()), 100)
        self.assertEqual([q["type"] for q in mdp.QUESTIONS].count("code"), 2)
        self.assertEqual(len({q["id"] for q in mdp.QUESTIONS}), 12)
        for q in mdp.QUESTIONS:
            self.assertIn("印刷页", q["source_note"]["text"])
            self.assertIn("PDF 页", q["source_note"]["text"])
            if "rubric" in q:
                self.assertEqual(sum(x["points"] for x in q["rubric"]), q["max_score"])

    def test_no_answer_keys_in_public_payload(self):
        public = json.dumps(mdp.public_payload(), ensure_ascii=False)
        for secret in ('"_answer"', '"_expected"', '"_solution"', '"model_answer"', '"rubric"'):
            self.assertNotIn(secret, public)
        self.assertNotEqual(bank.grade("q01", 1)["correct"], mdp.grade("q01", 1)["correct"])

    def test_grading_correct_incorrect_invalid_and_reveal(self):
        for qid, correct, wrong in [
            ("q01", 1, 2), ("q02", 2, 0), ("q03", 2, 1), ("q04", 1, 3),
            ("q05", 3, 2), ("q06", [.425, .575], [.575, .425]),
            ("q07", 2.575, 2.8), ("q08", [2.8, 3.9], [2.575, 3.9]),
        ]:
            self.assertTrue(mdp.grade(qid, correct)["correct"], qid)
            self.assertFalse(mdp.grade(qid, wrong)["correct"], qid)
        with self.assertRaises(bank.InvalidAnswer):
            mdp.grade("q08", [2.8])
        with self.assertRaises(bank.InvalidAnswer):
            mdp.grade("q07", float("nan"))
        with self.assertRaises(bank.UnknownQuestion):
            mdp.grade("not-present", 0)
        with self.assertRaises(bank.NotRevealable):
            mdp.reveal("q01")
        with self.assertRaises(bank.NotAutoGradable):
            mdp.grade("q09", "解释")
        for qid in ("q09", "q10", "q11", "q12"):
            self.assertEqual(sum(x["points"] for x in mdp.reveal(qid)["rubric"]),
                             mdp._get(qid)["max_score"])

    def test_constructed_model_independently(self):
        case = mdp.INDUSTRY_CASE
        model = case["model"]
        self.assertEqual(sum(row["exposures"] for row in case["rows"]), 40)
        for row in case["rows"]:
            s, a = row["state"], row["action"]
            self.assertEqual(sum(row["next_counts"]), row["exposures"])
            self.assertEqual([n / row["exposures"] for n in row["next_counts"]],
                             model["transitions"][s][a])
        vals = model["values"]
        q = [
            [model["rewards"][s][a] +
             model["gamma"] * sum(prob * vals[next_s]
                                  for next_s, prob in enumerate(model["transitions"][s][a]))
             for a in range(2)]
            for s in range(2)
        ]
        for row, expected in zip(q, [[2.5, 2.8], [3.9, 3.0]]):
            for got, wanted in zip(row, expected):
                self.assertAlmostEqual(got, wanted)
        evaluated = [sum(p * qa for p, qa in zip(model["policy"][s], q[s])) for s in range(2)]
        for got, wanted in zip(evaluated, [2.575, 3.9]):
            self.assertAlmostEqual(got, wanted)
        for got, wanted in zip([max(row) for row in q], [2.8, 3.9]):
            self.assertAlmostEqual(got, wanted)
        self.assertEqual([
            sum(p * model["transitions"][0][a][0]
                for a, p in enumerate(model["policy"][0])),
            sum(p * model["transitions"][0][a][1]
                for a, p in enumerate(model["policy"][0])),
        ], [.425, .575])
        for batch in case["code_path"]["batch"]:
            n = sum(batch["mask"])
            cumulative = 0
            returns = [0] * len(batch["mask"])
            for t in reversed(range(n)):
                cumulative = batch["rm_scores"][t] + .5 * cumulative
                returns[t] = cumulative
            self.assertEqual(returns, [.25, .5, 1] if batch["id"] == "A" else [.5, 1, 0])

    def test_q08_contains_every_reward_and_transition_without_later_case(self):
        prompt = mdp._get("q08")["prompt"]
        model = mdp.MODEL
        self.assertIn("状态顺序为 [候选, 留存]", prompt)
        self.assertIn("两行都使用给定的旧 V", prompt)
        for s, state in enumerate(model["states"]):
            for a, action in enumerate(model["actions"]):
                probabilities = ", ".join(f"{p:g}" for p in model["transitions"][s][a])
                row = f"{state} / {action}：R={model['rewards'][s][a]:g}，P=[{probabilities}]"
                self.assertIn(row, prompt)

    def test_reference_solutions_pass_public_tests_without_mutation(self):
        for qid in ("q11", "q12"):
            question = copy.deepcopy(mdp._get(qid))
            namespace = {"__name__": "__main__"}
            exec(question["_solution"], namespace)
            for test in question["tests"]:
                with self.subTest(qid=qid, test=test["name"]):
                    exec(test["code"], namespace)

    def test_wrong_implementations_fail_diagnostic_tests(self):
        q_eval = mdp._get("q11")
        q_control = mdp._get("q12")
        self.assertIn("policy_weights", [x["name"] for x in q_eval["tests"]])
        self.assertIn("joint_max", [x["name"] for x in q_control["tests"]])
        self.assertIn("synchronous", [x["name"] for x in q_control["tests"]])
        wrong = {"optimal_backup": lambda r, p, v, g:
                 ([max(r[s]) + g * max(sum(prob * v[t] for t, prob in enumerate(p[s][a]))
                                     for a in range(len(r[s])))
                   for s in range(len(r))], [0] * len(r))}
        with self.assertRaises(AssertionError):
            exec(next(x["code"] for x in q_control["tests"] if x["name"] == "joint_max"), wrong)

    def test_source_paths_when_checkout_present(self):
        here = Path(__file__).resolve().parent
        root = next((parent for parent in (here.parent, here.parent.parent)
                     if (parent / "verl").is_dir()), None)
        if root is None:
            self.skipTest("external verl checkout not present in public repository")
        for stage in mdp.INDUSTRY_CASE["code_path"]["stages"]:
            self.assertTrue(os.path.isfile(os.path.join(root, stage["path"])), stage["path"])
        with open(os.path.join(root, "verl/verl/trainer/ppo/core_algos.py"),
                  encoding="utf-8") as source_file:
            algos = source_file.read()
        self.assertIn("lastgaelam_ = delta + gamma * lam * lastgaelam", algos)
        self.assertIn("returns = advantages + values", algos)


if __name__ == "__main__":
    unittest.main()
