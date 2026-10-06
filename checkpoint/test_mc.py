"""Independent arithmetic, provenance, graders and API checks for the MC bank."""

import copy
from fractions import Fraction as F
import http.client
import json
import math
import random
import threading
import unittest

import bank
import build_site
import chapter_mc as mc
import chapters
import server


REQUIRED_CONTEXT = {
    "q01": ["固定策略", "完整轨迹", "不知道奖励分布与状态转移概率"],
    "q02": ["不知道环境模型", "固定策略"],
    "q03": ["(s1,a2)、(s2,a4)、(s1,a2)、(s2,a3)", "这一回合"],
    "q04": ["自然终止", "终止后的回报为0"],
    "q05": ["初始状态", "第一个动作"],
    "q06": ["0.2", "不衰减", "采样和可达性条件充分"],
    "q07": ["[0,-1,2]", "0.9", "G_3=0", "时间正序"],
    "q08": ["已有3个回报样本", "样本均值为2", "新回报6", "0.2", "不是先后执行"],
    "q09": ["r1=0", "r2=0", "r3=0", "r4=1", "0.5", "没有历史样本"],
    "q10": ["[1,4,2,0,-1]", "0.2", "全部5个动作", "包括贪心动作"],
    "q11": [
        "a1上、a2右、a3下、a4左、a5停留", "撞边界留在原地",
        "s6、s7", "禁区可进入", "其余奖励0", "到 s9 不终止", "γ=0.9",
        "s1上、s2下、s3右、s4右、s5下、s6下、s7右、s8右、s9停",
        "然后一直遵循 π0", "本题不是直接求 Q*",
    ],
    "q12": ["每步停留的奖励均为+1", "0.9", "3个奖励", "第4个奖励", "永不自然终止"],
    "q13": ["前两次奖励为−1、+1", "前四次奖励为0、0、0、+1", "γ=0.9", "立即自然终止"],
    "q14": ["均值为0、方差为1", "100", "相互独立、同分布", "只抽一次"],
    "q15": ["不允许任意指定初始", "完全不可达", "固定 ε>0"],
    "q16": ["自然终止后的 G_T=0", "[0,1]", "空序列返回 []", "不得修改输入", "ValueError"],
    "q17": ["返回 (Q, N)", "两序列长度相等", "跨回合重新判断", "不要依赖其他题", "ValueError"],
    "q18": ["全部动作", "最小下标", "epsilon=0", "epsilon=1", "不依赖其他题", "ValueError"],
}


def forward_returns(rewards, gamma):
    return [sum(gamma ** (k - t) * rewards[k] for k in range(t, len(rewards)))
            for t in range(len(rewards))]


class McBankTests(unittest.TestCase):
    def test_metadata_sources_and_rubrics(self):
        self.assertEqual(len(mc.QUESTIONS), 18)
        self.assertEqual(mc.TOTAL_SCORE, 100)
        self.assertEqual(mc.SCORING, {
            "objective": 54, "open_self_review": 18, "code_browser_tests": 28})
        self.assertEqual(len({q["id"] for q in mc.QUESTIONS}), 18)
        for question in mc.QUESTIONS:
            with self.subTest(question=question["id"]):
                self.assertTrue(question["learning_goal"])
                for source in ("Zhao", "EasyRL", "印刷页", "PDF 页"):
                    self.assertIn(source, question["source_note"]["text"])
                if "rubric" in question:
                    self.assertEqual(sum(row["points"] for row in question["rubric"]),
                                     question["max_score"])

    def test_every_question_contains_its_own_required_inputs(self):
        self.assertEqual(set(REQUIRED_CONTEXT), {q["id"] for q in mc.QUESTIONS})
        for qid, fragments in REQUIRED_CONTEXT.items():
            for fragment in fragments:
                self.assertIn(fragment, mc._get(qid)["prompt"], (qid, fragment))

    def test_public_payload_has_no_answer_keys_or_reference_solutions(self):
        text = json.dumps(mc.public_payload(), ensure_ascii=False)
        for secret in ('"_answer"', '"_expected"', '"_solution"', '"rubric"', '"model_answer"'):
            self.assertNotIn(secret, text)
        coding_text = " ".join(
            q["prompt"] + " ".join(t["code"] for t in q["tests"])
            for q in mc.QUESTIONS if q["type"] == "code")
        self.assertNotIn("[-10,-10,8,7.29,-9]", coding_text)
        self.assertNotIn("[.72,.8,2]", coding_text)
        self.assertNotIn(".3125", coding_text)

    def test_numeric_lengths_are_public_but_values_are_not(self):
        questions = mc.public_payload()["questions"]
        lengths = {q["id"]: q["answer_count"] for q in questions if q["type"] == "numeric"}
        self.assertEqual(lengths, {"q07": 3, "q08": 2, "q09": 2, "q10": 5, "q11": 6, "q12": 2})
        self.assertTrue(all("answer_count" not in q for q in bank.public_payload()["questions"]))
        for question in questions:
            self.assertNotIn("_expected", question)

    def test_numeric_answers_from_independent_arithmetic(self):
        rewards = [F(0), F(-1), F(2)]
        episode = [F(0), F(0), F(0), F(1)]
        visits = forward_returns(episode, F(1, 2))
        gamma = F(9, 10)
        expected = {
            "q07": forward_returns(rewards, gamma),
            "q08": [F(3 * 2 + 6, 4), F(2) + F(1, 5) * (6 - 2)],
            "q09": [visits[0], (visits[0] + visits[2]) / 2],
            "q10": [F(1, 25), F(21, 25), F(1, 25), F(1, 25), F(1, 25)],
            "q12": [sum(gamma ** k for k in range(3)), gamma ** 3 / (1 - gamma)],
        }
        for qid, answers in expected.items():
            self.assertEqual([float(x) for x in answers], mc._get(qid)["_expected"], qid)
            self.assertTrue(mc.grade(qid, [float(x) for x in answers])["correct"], qid)

    def test_s3_values_from_independent_rollout(self):
        gamma = .9
        policy = [0, 2, 1, 1, 2, 2, 1, 1, 4]
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

        def step(state, action):
            row, col = divmod(state - 1, 3)
            dr, dc = moves[action]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 3 and 0 <= nc < 3):
                return state, -1
            next_state = nr * 3 + nc + 1
            reward = -1 if next_state in (6, 7) else 1 if next_state == 9 else 0
            return next_state, reward

        values = []
        for first_action in range(5):
            state, total = 3, 0
            for t in range(250):
                state, reward = step(state, first_action if t == 0 else policy[state - 1])
                total += gamma ** t * reward
            values.append(total)
        for value, answer in zip(values, mc._get("q11")["_expected"][:5]):
            self.assertAlmostEqual(value, answer, places=8)
        action = max(range(5), key=lambda index: values[index]) + 1
        self.assertEqual(action, 3)
        self.assertTrue(mc.grade("q11", values + [action])["correct"])

    def test_counterfactual_and_variance_arithmetic(self):
        gamma = F(9, 10)
        self.assertEqual(-1 + gamma / (1 - gamma), F(8))
        self.assertEqual(gamma ** 3 / (1 - gamma), F(729, 100))
        self.assertEqual(-1 + gamma, F(-1, 10))
        self.assertEqual(gamma ** 3, F(729, 1000))
        self.assertIn("0.729", mc.reveal("q13")["model_answer"])
        self.assertIn("0.01", mc.reveal("q14")["model_answer"])
        self.assertIn("方差始终为1", mc.reveal("q14")["model_answer"])

    def test_correct_wrong_invalid_and_reveal(self):
        for question in mc.QUESTIONS:
            if question["type"] == "choice":
                answer = question["_answer"]
                self.assertTrue(mc.grade(question["id"], answer)["correct"])
                self.assertFalse(mc.grade(question["id"], (answer + 1) % 4)["correct"])
                self.assertTrue(mc.grade(question["id"], "ABCD"[answer])["correct"])
                with self.assertRaises(bank.InvalidAnswer):
                    mc.grade(question["id"], True)
            elif question["type"] == "numeric":
                answer = question["_expected"]
                self.assertTrue(mc.grade(question["id"], answer)["correct"])
                wrong = [value + 1 for value in answer]
                self.assertFalse(mc.grade(question["id"], wrong)["correct"])
                for invalid in ([True] * len(answer), [float("nan")] * len(answer),
                                [float("inf")] * len(answer), [10 ** 1000] * len(answer),
                                [], {}, None):
                    with self.assertRaises(bank.InvalidAnswer):
                        mc.grade(question["id"], invalid)
                with self.assertRaises(bank.NotRevealable):
                    mc.reveal(question["id"])
            else:
                with self.assertRaises(bank.NotAutoGradable):
                    mc.grade(question["id"], "student submission")
                self.assertEqual(mc.reveal(question["id"])["type"], question["type"])
        with self.assertRaises(bank.UnknownQuestion):
            mc.grade([], 0)
        with self.assertRaises(bank.UnknownQuestion):
            mc.reveal("missing")

    def test_reference_solutions_and_public_tests(self):
        for qid in ("q16", "q17", "q18"):
            question = mc._get(qid)
            for test in question["tests"]:
                with self.subTest(question=qid, test=test["name"]):
                    namespace = {"__name__": "__main__"}
                    exec(question["_solution"], namespace)
                    exec(test["code"], namespace)

    def test_reference_solutions_against_independent_forward_oracles(self):
        namespace = {}
        for qid in ("q16", "q17", "q18"):
            exec(mc._get(qid)["_solution"], namespace)
        rng = random.Random(73)
        for _ in range(40):
            gamma = rng.choice([0, .5, .9, 1])
            rewards = [rng.randint(-3, 4) for _ in range(rng.randrange(9))]
            got = namespace["discounted_returns"](rewards, gamma)
            for value, target in zip(got, forward_returns(rewards, gamma)):
                self.assertAlmostEqual(value, target)
            episodes = [
                {"pairs": [(str(rng.randrange(2)), str(rng.randrange(2)))
                           for _ in rewards], "rewards": rewards},
                {"pairs": [("0", "0")], "rewards": [2]},
            ]
            before = copy.deepcopy(episodes)
            for visit in ("first", "every"):
                samples = {}
                for episode in episodes:
                    seen = set()
                    for pair, value in zip(episode["pairs"],
                                           forward_returns(episode["rewards"], gamma)):
                        if visit == "every" or pair not in seen:
                            samples.setdefault(pair, []).append(value)
                        seen.add(pair)
                q, counts = namespace["mc_action_values"](episodes, gamma, visit)
                self.assertEqual(counts, {key: len(values) for key, values in samples.items()})
                for key, values in samples.items():
                    self.assertAlmostEqual(q[key], sum(values) / len(values))
            self.assertEqual(episodes, before)
            qs = [rng.randint(-2, 3) for _ in range(rng.randrange(1, 6))]
            epsilon = rng.choice([0, .2, 1])
            greedy = qs.index(max(qs))
            pmf = namespace["epsilon_greedy_probs"](qs, epsilon)
            expected = [epsilon / len(qs) + (1 - epsilon if i == greedy else 0)
                        for i in range(len(qs))]
            self.assertEqual(pmf, expected)
            self.assertAlmostEqual(sum(pmf), 1)

    def test_starters_and_typical_wrong_implementations_fail(self):
        for qid in ("q16", "q17", "q18"):
            question = mc._get(qid)
            namespace = {}
            exec(question["starter_code"], namespace)
            with self.assertRaises(AssertionError):
                exec(question["tests"][0]["code"], namespace)
        wrong_first = {
            "mc_action_values": lambda *args: ({("A", "x"): 2}, {("A", "x"): 1})
        }
        with self.assertRaises(AssertionError):
            exec(next(t["code"] for t in mc._get("q17")["tests"]
                      if t["name"] == "first_is_earliest"), wrong_first)
        with self.assertRaises(AssertionError):
            exec(mc._get("q18")["tests"][0]["code"],
                 {"epsilon_greedy_probs": lambda *_: [.15, .7, .15]})

    def test_registry_and_static_export_keep_legacy_default(self):
        self.assertEqual(chapters.DEFAULT_CHAPTER_ID, "easyrl-2.1-2.2.2")
        self.assertIs(chapters.resolve_bank(mc.CHAPTER_ID), mc)
        exported = next(ch for ch in build_site.export_data()["chapters"]
                        if ch["id"] == mc.CHAPTER_ID)
        self.assertEqual(exported["count"], 18)
        self.assertEqual(exported["answers"]["q11"]["expected"],
                         [-10, -10, 8, 7.29, -9, 3])
        self.assertEqual(exported["reveals"]["q17"]["reference_solution"], mc._MC_SOLUTION)


class McApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.build_server("127.0.0.1", 0, str(build_site.ROOT))
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=5)
        raw = None if body is None else json.dumps(body).encode("utf-8")
        try:
            conn.request(method, path, body=raw,
                         headers={} if raw is None else {"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def test_real_mc_routes_and_same_id_isolation(self):
        status, listing = self.request("GET", "/api/chapters")
        self.assertEqual(status, 200)
        self.assertIn(mc.CHAPTER_ID, [row["id"] for row in listing["chapters"]])
        status, public = self.request("GET", "/api/questions?chapter=monte-carlo")
        self.assertEqual(status, 200)
        self.assertEqual(public, mc.public_payload())
        status, result = self.request("POST", "/api/grade",
                                     {"chapter": mc.CHAPTER_ID, "id": "q01", "answer": 0})
        self.assertEqual(status, 200)
        self.assertTrue(result["correct"])
        self.assertFalse(bank.grade("q01", 0)["correct"])
        status, result = self.request("POST", "/api/reveal",
                                     {"chapter": mc.CHAPTER_ID, "id": "q13"})
        self.assertEqual(status, 200)
        self.assertTrue(result["self_assessment"])
        status, result = self.request("POST", "/api/grade",
                                     {"chapter": mc.CHAPTER_ID, "id": "q11", "answer": [True] * 6})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
