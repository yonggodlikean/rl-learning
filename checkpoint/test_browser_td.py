"""Real TD chapter regression: numeric submission and saved progress."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase

import build_site
import chapter_td

try:
    from playwright.sync_api import Error as PlaywrightError, sync_playwright
except ImportError:
    sync_playwright = None


class SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class TDNumericBrowserTests(TestCase):
    def test_real_numeric_questions_accept_the_requested_answer_shapes(self):
        if sync_playwright is None:
            self.skipTest("Playwright is not installed")
        artifact_root = build_site.ROOT / "dist" / "td-browser-tests"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=artifact_root) as temp:
            build_site.build(Path(temp) / "checkpoint")
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(SilentHandler, directory=temp))
            thread = Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                self.exercise_site(
                    f"http://127.0.0.1:{httpd.server_port}/checkpoint/")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def exercise_site(self, base):
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as error:
                if "Executable doesn't exist" in str(error):
                    self.skipTest(f"Chromium is not installed: {error}")
                raise
            try:
                for width in (1280, 390):
                    with self.subTest(width=width):
                        context = browser.new_context(
                            viewport={"width": width, "height": 850})
                        try:
                            self.exercise_numeric_flow(context, base)
                        finally:
                            context.close()
            finally:
                browser.close()

    def exercise_numeric_flow(self, context, base):
        errors = []
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: errors.append(
            f"HTTP {response.status}: {response.url}")
            if response.status >= 400 else None)
        url = base + "?chapter=" + chapter_td.CHAPTER_ID + "&question="

        page.goto(url + "q01", wait_until="networkidle")
        page.get_by_role("radio").nth(0).check()
        page.get_by_role("button", name="提交判断").click()
        page.get_by_text("判断正确", exact=True).wait_for()
        self.assertEqual(page.locator("#progress-count").inner_text(), "1 / 23")

        page.goto(url + "q09", wait_until="networkidle")
        page.locator(".numeric-field").fill("2.55")
        page.reload(wait_until="networkidle")
        self.assertEqual(page.locator(".numeric-field").input_value(), "2.55")
        self.assertEqual(page.locator("#progress-count").inner_text(), "1 / 23")
        page.get_by_role("button", name="核对答案").click()
        page.get_by_text("计算正确", exact=True).wait_for()
        self.assertEqual(page.locator("#progress-count").inner_text(), "2 / 23")

        for raw in ("NaN", "Infinity", "2.55, 3"):
            page.locator(".numeric-field").fill(raw)
            page.get_by_role("button", name="核对答案").click()
            page.get_by_text("请输入一个有限数值。", exact=True).wait_for()
        page.locator(".numeric-field").fill("0")
        page.get_by_role("button", name="核对答案").click()
        page.get_by_text("拆开算一次", exact=True).wait_for()

        for qid, raw in (("q09", "2.55"), ("q10", "3.1"),
                         ("q11", "3.2"), ("q12", "6.85")):
            page.goto(url + qid, wait_until="networkidle")
            self.assertEqual(page.locator("#question-view .answer-label").inner_text(),
                             "输入计算结果")
            page.locator(".numeric-field").fill(raw)
            page.get_by_role("button", name="核对答案").click()
            page.get_by_text("计算正确", exact=True).wait_for()
            self.assertNotIn("请按顺序输入", page.locator("#question-view").inner_text())
            page.reload(wait_until="networkidle")
            self.assertEqual(page.locator(".numeric-field").input_value(), raw)
            self.assertEqual(page.get_by_text("计算正确", exact=True).count(), 1)

        page.goto(url + "q13", wait_until="networkidle")
        self.assertEqual(page.locator("#question-view .answer-label").inner_text(),
                         "输入两个数值")
        page.locator(".numeric-field").fill("0.8")
        page.get_by_role("button", name="核对答案").click()
        page.get_by_text("请按顺序输入2个数值。", exact=True).wait_for()
        page.locator(".numeric-field").fill("[0.8, 0.1]")
        page.get_by_role("button", name="核对答案").click()
        page.get_by_text("计算正确", exact=True).wait_for()
        page.reload(wait_until="networkidle")
        self.assertEqual(page.locator("#progress-count").inner_text(), "6 / 23")
        self.assertLessEqual(page.evaluate(
            "document.documentElement.scrollWidth - innerWidth"), 0)
        self.assertFalse(errors, errors)
