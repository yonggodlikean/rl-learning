"""End-to-end fixture: a second chapter with reused question IDs and Python code.

The fixture is registered only during this test's static export. It is never
published as an actual learning chapter.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import build_site
import chapters

try:
    from playwright.sync_api import Error as PlaywrightError, sync_playwright
except ImportError:
    sync_playwright = None


def fixture_bank():
    choice = {
        "id": "q01", "type": "choice", "category": "concept",
        "title": "第二章的第一题", "prompt": r"选择正确的数字：\(1+0\)。",
        "max_score": 1, "grading_mode": "auto",
        "options": ["零", "一"], "_answer": 1,
        "review_hint": "回看第二章 §1：数字。",
    }
    code = {
        "id": "q02", "type": "code", "category": "coding",
        "title": "第二章代码题", "prompt": r"实现 \(f(x)=2x\) 的 double(x)。",
        "max_score": 2, "grading_mode": "browser",
        "language": "python", "entry_point": "double",
        "starter_code": "def double(x):\n    return 0",
        "tests": [
            {"name": "positive", "description": "double(3) 应等于 6。",
             "code": "assert double(3) == 6"},
        ],
        "_solution": "def double(x):\n    return 2 * x",
    }
    questions = [choice, code]
    payload = {
        "reference": "合成测试来源（非真实教材章节）",
        "conventions": "", "industry_case": None,
        "count": 2, "total_score": 3,
        "scoring": {"objective": 1, "open_self_review": 0, "code_browser_tests": 2},
        "presentation": {
            "eyebrow": "仅供自动化测试",
            "intro": "第二章应有自己独立的页首。",
            "field_notes": [
                {"label": "01 / TEST", "formula": "2x", "tex": "2x",
                 "description": "第二章便签，不得显示第一章旧公式。"},
            ],
        },
        "questions": [
            {key: value for key, value in question.items() if not key.startswith("_")}
            for question in questions
        ],
    }
    return SimpleNamespace(
        REFERENCE=payload["reference"],
        list_questions=lambda: questions,
        public_payload=lambda: payload,
        grade=lambda question_id, answer: {
            "correct": answer == 1, "score": int(answer == 1),
            "max_score": 1, "feedback": "测试", "expected": "B",
        },
        reveal=lambda question_id: {
            "id": question_id, "type": "code", "max_score": 2,
            "self_assessment": False, "entry_point": "double",
            "reference_solution": code["_solution"], "rubric": [],
        },
    )


class SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class BrowserChapterTests(TestCase):
    def test_direct_links_scoped_progress_and_code(self):
        if sync_playwright is None:
            self.skipTest("Playwright 未安装；浏览器测试需手动运行")
        with TemporaryDirectory() as temp:
            with patch.dict(chapters._REGISTRY):
                chapters.register_bank("fixture-second", "第二章 · 合成浏览器测试", fixture_bank())
                build_site.build(Path(temp) / "checkpoint")
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(SilentHandler, directory=temp))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.exercise_site(f"http://127.0.0.1:{server.server_port}/checkpoint/")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def exercise_site(self, base):
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as error:
                if "Executable doesn't exist" in str(error):
                    self.skipTest(f"Chromium 浏览器未安装：{error}")
                raise
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 850})
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("response", lambda response: errors.append(
                    f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)
                page.goto(base + "?chapter=fixture-second&question=q02", wait_until="networkidle")
                self.assertEqual(page.locator("#chapter-select").input_value(), "fixture-second")
                self.assertEqual(page.locator(".question-title").inner_text(), "第二章代码题")
                self.assertEqual(page.locator(".hero-stamp strong").inner_text(), "2题")
                self.assertIn("第二章便签", page.locator(".reference").inner_text())
                self.assertNotIn("从后往前算", page.locator(".reference").inner_text())
                self.assertEqual(page.locator(".cm-editor").count(), 1)
                self.assertGreater(page.locator(".question-prompt .katex").count(), 0)
                self.assertIn("chapter=fixture-second", page.locator(".chapter-permalink")
                              .get_attribute("href"))
                self.assertIn("question=q02", page.locator(".question-permalink")
                              .get_attribute("href"))
                page.get_by_role("button", name="运行公开测试").click()
                page.get_by_text("公开测试 0 / 1 通过").wait_for(timeout=60000)
                page.locator(".cm-content").click()
                page.keyboard.press("Meta+A")
                page.keyboard.insert_text("def double(x):\n    return x * 2")
                page.get_by_role("button", name="运行公开测试").click()
                page.get_by_text("公开测试 1 / 1 通过").wait_for(timeout=60000)
                page.get_by_role("button", name="看参考实现").click()
                self.assertIn("return 2 * x", page.locator(".reference-answer").inner_text())
                page.reload(wait_until="networkidle")
                self.assertEqual(page.locator("#progress-count").inner_text(), "1 / 2")
                page.get_by_role("button", name="01 第二章的第一题").click()
                self.assertIn("question=q01", page.url)
                page.get_by_role("radio", name="B").check()
                page.get_by_role("button", name="提交判断").click()
                page.get_by_text("判断正确").wait_for()
                self.assertEqual(page.locator("#progress-count").inner_text(), "2 / 2")
                page.get_by_role("radio", name="A").check()
                page.get_by_role("button", name="提交判断").click()
                page.get_by_text("再对照定义看一眼").wait_for()
                page.locator("#chapter-select").select_option(chapters.DEFAULT_CHAPTER_ID)
                self.assertEqual(page.locator("#progress-count").inner_text(), "0 / 12")
                self.assertEqual(page.locator(".hero-stamp strong").inner_text(), "12题")
                self.assertIn("从后往前算", page.locator(".reference").inner_text())
                page.locator("#chapter-select").select_option("fixture-second")
                self.assertEqual(page.locator("#progress-count").inner_text(), "2 / 2")
                page.get_by_role("button", name="查看学习复盘").click()
                self.assertIn("第二章 §1", page.locator(".review-list").inner_text())
                fresh = browser.new_context(viewport={"width": 390, "height": 844},
                                            is_mobile=True, has_touch=True)
                mobile = fresh.new_page()
                mobile.goto(base + "?chapter=fixture-second&question=q01",
                            wait_until="networkidle")
                self.assertEqual(mobile.locator("#chapter-select").input_value(), "fixture-second")
                self.assertEqual(mobile.locator(".question-title").inner_text(), "第二章的第一题")
                self.assertGreater(mobile.locator(".question-prompt .katex").count(), 0)
                self.assertLessEqual(mobile.evaluate(
                    "document.documentElement.scrollWidth - innerWidth"), 0)
                unknown = fresh.new_page()
                unknown.goto(base + "?chapter=missing&question=q999", wait_until="networkidle")
                self.assertEqual(unknown.locator("#chapter-select").input_value(),
                                 chapters.DEFAULT_CHAPTER_ID)
                self.assertIn("chapter=" + chapters.DEFAULT_CHAPTER_ID, unknown.url)
                self.assertIn("不属于本章", unknown.locator(".feedback").inner_text())
                self.assertFalse(errors, errors)
                fresh.close()
                context.close()
            finally:
                browser.close()
