"""Real §2.3 chapter in exported static site: attempt, review, code, mobile."""

import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase

import build_site
import chapter_mdp

try:
    from playwright.sync_api import Error as PlaywrightError, sync_playwright
except ImportError:
    sync_playwright = None


class QuietStatic(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class MdpBrowserTests(TestCase):
    def test_real_chapter_flows(self):
        if sync_playwright is None:
            self.skipTest("Playwright 未安装")
        with TemporaryDirectory() as temp:
            build_site.build(Path(temp) / "checkpoint")
            server = ThreadingHTTPServer(("127.0.0.1", 0),
                                         partial(QuietStatic, directory=temp))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.exercise(f"http://127.0.0.1:{server.server_port}/checkpoint/")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def exercise(self, base):
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as error:
                if "Executable doesn't exist" in str(error):
                    self.skipTest(str(error))
                raise
            try:
                context = browser.new_context(viewport={"width": 1365, "height": 900})
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + "?chapter=easyrl-2.3&question=q09",
                          wait_until="networkidle")
                self.assertEqual(page.locator("#chapter-select").input_value(), "easyrl-2.3")
                self.assertEqual(page.locator(".hero-stamp strong").inner_text(), "12题")
                self.assertIn("印刷页", page.locator(".question-source").inner_text())
                self.assertIn("2.575", page.locator(".mdp-output-table").inner_text())
                self.assertIn("2.8", page.locator(".mdp-output-table").inner_text())
                self.assertIn("response_mask", page.locator(".mdp-training-table").inner_text())
                self.assertIn("verl", page.locator(".mdp-source").inner_text().lower())
                self.assertEqual(page.locator(".mdp-case .case-step").count(), 8)
                self.assertLessEqual(page.evaluate(
                    "document.documentElement.scrollWidth - innerWidth"), 0)
                page.get_by_role("button", name="写完了，对照评分要点").click()
                self.assertIn("先写下", page.locator(".feedback").inner_text())
                page.locator("#open-q09").fill("给定模型的规划不同于根据曝光日志估计因果效果。")
                page.get_by_role("button", name="写完了，对照评分要点").click()
                self.assertIn("反事实", page.locator(".rubric").inner_text())
                page.locator("#score-q09").fill("7")
                page.get_by_role("button", name="记录自评").click()
                self.assertIn("7 / 9", page.locator(".rubric").inner_text())

                page.goto(base + "?chapter=easyrl-2.3&question=q08", wait_until="networkidle")
                self.assertGreater(page.locator(".question-prompt .katex").count(), 0)
                page.locator("#numeric-q08").fill("2.575, 3.9")
                page.get_by_role("button", name="核对答案").click()
                page.get_by_text("拆开算一次").wait_for()
                page.locator("#numeric-q08").fill("2.8, 3.9")
                page.get_by_role("button", name="核对答案").click()
                page.get_by_text("计算正确").wait_for()
                page.goto(base + "?chapter=easyrl-2.3&question=q11", wait_until="networkidle")
                self.assertEqual(page.locator(".cm-editor").count(), 1)
                page.get_by_role("button", name="运行公开测试").click()
                page.get_by_text(re.compile(r"公开测试 [0-5] / 6 通过")).wait_for(timeout=60000)
                page.locator(".cm-content").click()
                page.keyboard.press("Meta+A")
                page.keyboard.insert_text("def policy_backup(:\n    pass")
                page.get_by_role("button", name="运行公开测试").click()
                page.locator(".syntax-diagnostic").get_by_text("第 1 行", exact=False).wait_for(timeout=60000)
                page.locator(".cm-content").click()
                page.keyboard.press("Meta+A")
                page.keyboard.insert_text("def policy_backup(rewards, transitions, policy, values, gamma):\n  return [0]")
                page.get_by_role("button", name="运行公开测试").click()
                page.get_by_text(re.compile(r"公开测试 [0-5] / 6 通过")).wait_for(timeout=60000)
                page.locator(".cm-content").click()
                page.keyboard.press("Meta+A")
                page.keyboard.insert_text(chapter_mdp.reveal("q11")["reference_solution"])
                page.get_by_role("button", name="运行公开测试").click()
                page.get_by_text("公开测试 6 / 6 通过").wait_for(timeout=60000)
                page.get_by_role("button", name="看参考实现").click()
                self.assertIn("expected_next", page.locator(".reference-answer").inner_text())
                page.reload(wait_until="networkidle")
                self.assertIn("公开测试 6 / 6 通过", page.locator(".feedback").inner_text())

                page.goto(base + "?chapter=easyrl-2.3&question=q12", wait_until="networkidle")
                page.locator(".cm-content").click()
                page.keyboard.press("Meta+A")
                page.keyboard.insert_text(chapter_mdp.reveal("q12")["reference_solution"])
                page.get_by_role("button", name="运行公开测试").click()
                page.get_by_text("公开测试 6 / 6 通过").wait_for(timeout=60000)
                page.locator("#chapter-select").select_option("easyrl-2.1-2.2.2")
                self.assertEqual(page.locator("#progress-count").inner_text(), "0 / 12")
                self.assertNotIn("mdp-planning", page.locator("#question-view").inner_text())
                page.locator("#chapter-select").select_option("easyrl-2.3")
                self.assertEqual(page.locator("#progress-count").inner_text(), "4 / 12")

                mobile = browser.new_context(viewport={"width": 390, "height": 844},
                                             is_mobile=True, has_touch=True)
                screen = mobile.new_page()
                screen.on("pageerror", lambda error: errors.append(str(error)))
                screen.goto(base + "?chapter=easyrl-2.3&question=q09", wait_until="networkidle")
                self.assertLessEqual(screen.evaluate(
                    "document.documentElement.scrollWidth - innerWidth"), 0)
                self.assertGreater(screen.locator("#chapter-select").bounding_box()["width"], 200)
                self.assertGreaterEqual(screen.locator("#chapter-select").bounding_box()["x"], 0)
                self.assertTrue(screen.locator(".nav-strip").evaluate(
                    "el => el.scrollWidth > el.clientWidth"))
                self.assertGreater(screen.locator(".mdp-input-table").evaluate(
                    "el => el.scrollWidth"), 0)
                self.assertFalse(errors, errors)
                mobile.close()
                context.close()
            finally:
                browser.close()


if __name__ == "__main__":
    import unittest
    unittest.main()
