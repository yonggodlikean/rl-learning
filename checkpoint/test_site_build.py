"""Regression tests for multi-chapter APIs and the portable site export."""

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bank
import build_site
import chapters
import server


class ChapterApiTests(unittest.TestCase):
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
        raw = json.dumps(body).encode("utf-8") if body is not None else None
        conn.request(method, path, body=raw,
                     headers={"Content-Type": "application/json"} if raw is not None else {})
        response = conn.getresponse()
        result = response.status, json.loads(response.read().decode("utf-8"))
        conn.close()
        return result

    def test_chapters_and_explicit_default(self):
        status, listing = self.request("GET", "/api/chapters")
        self.assertEqual(status, 200)
        self.assertEqual(listing["default_chapter_id"], chapters.DEFAULT_CHAPTER_ID)
        self.assertEqual(listing["chapters"][0]["count"], 12)
        status, questions = self.request(
            "GET", "/api/questions?chapter=" + chapters.DEFAULT_CHAPTER_ID)
        self.assertEqual(status, 200)
        self.assertEqual(questions, bank.public_payload())
        status, result = self.request("POST", "/api/grade",
                                      {"chapter": chapters.DEFAULT_CHAPTER_ID,
                                       "id": "q02", "answer": 2})
        self.assertEqual(status, 200)
        self.assertTrue(result["correct"])
        status, result = self.request("POST", "/api/reveal",
                                      {"chapter": chapters.DEFAULT_CHAPTER_ID, "id": "q10"})
        self.assertEqual(status, 200)
        self.assertEqual(result, bank.reveal("q10"))

    def test_unknown_and_invalid_chapters(self):
        self.assertEqual(self.request("GET", "/api/questions?chapter=unknown")[0], 404)
        self.assertEqual(self.request("GET", "/api/questions?chapter=")[0], 400)
        self.assertEqual(self.request("GET", "/api/questions?chapter=a&chapter=b")[0], 400)
        self.assertEqual(self.request("POST", "/api/grade",
                                      {"id": "q01", "answer": 2, "chapter": "unknown"})[0], 404)
        self.assertEqual(self.request("POST", "/api/reveal",
                                      {"id": "q09", "chapter": 123})[0], 400)

    def test_registered_second_chapter_has_isolated_questions_and_answers(self):
        tiny_question = {
            "id": "q01", "type": "choice", "category": "concept",
            "title": "第二章", "prompt": "不同章节可复用 q01 吗？",
            "max_score": 1, "grading_mode": "auto",
            "options": ["可以", "不可以"], "_answer": 0,
        }
        tiny_bank = SimpleNamespace(
            REFERENCE="测试来源",
            list_questions=lambda: [tiny_question],
            public_payload=lambda: {
                "reference": "测试来源", "conventions": "",
                "industry_case": None, "count": 1, "total_score": 1,
                "scoring": {"objective": 1, "open_self_review": 0, "code_browser_tests": 0},
                "questions": [{key: value for key, value in tiny_question.items()
                               if not key.startswith("_")}],
            },
            grade=lambda question_id, answer: {
                "correct": answer == 0, "score": int(answer == 0),
                "max_score": 1, "feedback": "测试", "expected": "A",
            },
            reveal=lambda question_id: {},
        )
        with patch.dict(chapters._REGISTRY):
            chapters.register_bank("second-chapter", "测试第二章", tiny_bank)
            status, body = self.request("GET", "/api/chapters")
            self.assertEqual(status, 200)
            self.assertEqual(len(body["chapters"]), 2)
            status, body = self.request("GET", "/api/questions?chapter=second-chapter")
            self.assertEqual(status, 200)
            self.assertEqual(body["total_score"], 1)
            self.assertEqual(body["questions"][0]["title"], "第二章")
            status, body = self.request("POST", "/api/grade", {
                "chapter": "second-chapter", "id": "q01", "answer": 0,
            })
            self.assertEqual(status, 200)
            self.assertTrue(body["correct"])
            self.assertEqual(build_site.export_data()["chapters"][1]["answers"]["q01"]["answer"], 0)
        self.assertEqual(len(chapters.list_chapters()), 1)


class ExportTests(unittest.TestCase):
    def test_export_includes_only_intended_static_files_and_scoped_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp) / "checkpoint"
            data = build_site.build(site)
            chapter = data["chapters"][0]
            self.assertEqual(chapter["id"], chapters.DEFAULT_CHAPTER_ID)
            self.assertEqual(chapter["payload"]["count"], 12)
            public_json = json.dumps(chapter["payload"])
            self.assertNotIn('"_answer":', public_json)
            self.assertNotIn('"_expected":', public_json)
            self.assertNotIn('"_solution":', public_json)
            self.assertEqual(chapter["answers"]["q01"]["answer"], 2)
            self.assertEqual(chapter["answers"]["q05"]["expected"], 3.25)
            self.assertEqual(chapter["reveals"]["q11"]["reference_solution"],
                             bank.reveal("q11")["reference_solution"])

            html = (site / "index.html").read_text(encoding="utf-8")
            self.assertLess(html.index("./site-data.js"), html.index("./app.js"))
            self.assertIn('src="./vendor/katex/katex.min.js"', html)
            self.assertTrue((site / "vendor/pyodide/pyodide.asm.wasm").is_file())
            self.assertFalse((site / "bank.py").exists())
            self.assertFalse(list(site.rglob("*.pdf")))
            expected = json.dumps(data, ensure_ascii=False, allow_nan=False)
            self.assertEqual((site / "site-data.js").read_text(encoding="utf-8"),
                             "window.RL_SITE_DATA = " + expected + ";\n")

    def test_never_build_over_source(self):
        with self.assertRaises(ValueError):
            build_site.build(build_site.ROOT)


if __name__ == "__main__":
    unittest.main()
