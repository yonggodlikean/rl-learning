"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const base = process.env.MC_BASE_URL || "http://127.0.0.1:8765/";
const chapterId = "monte-carlo";
const screenshots = process.env.MC_SCREENSHOT_DIR;
const route = (id) => `${base}?chapter=${chapterId}&question=${id}`;
const required = {
  q08: ["已有3个回报样本", "新回报6", "不是先后执行"],
  q09: ["r1=0", "r2=0", "r3=0", "r4=1", "没有历史样本"],
  q11: [
    "a1上、a2右、a3下、a4左、a5停留", "撞边界留在原地",
    "禁区可进入", "到 s9 不终止", "γ=0.9",
    "s1上、s2下、s3右、s4右、s5下、s6下、s7右、s8右、s9停",
    "然后一直遵循 π0", "本题不是直接求 Q*",
  ],
  q12: ["3个奖励", "第4个奖励", "永不自然终止"],
  q13: ["γ=0.9", "立即自然终止", "前四次奖励为0、0、0、+1"],
  q17: ["返回 (Q, N)", "跨回合重新判断", "不要依赖其他题"],
};

async function api(endpoint, body) {
  const response = await fetch(new URL(endpoint, base), body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  assert.equal(response.status, 200, endpoint);
  return response.json();
}

function observe(page, errors) {
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("response", (response) => {
    if (response.status() >= 400 && !response.url().endsWith("/favicon.ico")) {
      errors.push(`HTTP ${response.status()}: ${response.url()}`);
    }
  });
}

async function open(page, id) {
  await page.goto(route(id), { waitUntil: "networkidle" });
  await page.locator(".question-title").waitFor();
  assert.equal(await page.locator("#chapter-select").inputValue(), chapterId);
  assert.equal(await page.locator(".hero-stamp strong").innerText(), "18题");
  assert.equal(await page.locator(".katex-error").count(), 0, `KaTeX: ${id}`);
}

async function capture(page, name) {
  if (!screenshots) return;
  fs.mkdirSync(screenshots, { recursive: true });
  await page.screenshot({ path: path.join(screenshots, name), fullPage: true });
}

async function typeCode(page, source) {
  await page.locator(".cm-content").click();
  await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
  await page.keyboard.insertText(source);
}

async function main() {
  const bank = await api(`/api/questions?chapter=${chapterId}`);
  assert.equal(bank.count, 18);
  assert.equal(bank.total_score, 100);
  const browser = await chromium.launch({
    headless: true, channel: process.env.MC_BROWSER_CHANNEL || "chrome",
  });
  const errors = [];
  try {
    // Every direct link starts without earlier navigation or saved answers.
    for (const question of process.env.MC_SKIP_COLD_START === "1" ? [] : bank.questions) {
      const fresh = await browser.newContext({ viewport: { width: 1365, height: 900 } });
      try {
        const page = await fresh.newPage();
        observe(page, errors);
        await open(page, question.id);
        assert.equal(await page.locator(".question-title").innerText(), question.title);
        assert.equal(await page.locator("#progress-count").innerText(), "0 / 18");
        const source = await page.locator(".question-source").innerText();
        for (const marker of ["Zhao", "EasyRL", "印刷页", "PDF 页"]) {
          assert.ok(source.includes(marker), `${question.id}: ${marker}`);
        }
        const prompt = await page.locator(".question-prompt").innerText();
        for (const marker of required[question.id] || []) {
          assert.ok(prompt.includes(marker), `${question.id}: ${marker}`);
        }
        assert.equal(await page.locator(".reference-answer").count(), 0);
        assert.equal(await page.locator(".rubric").count(), 0);
        if (question.type === "code") {
          assert.equal(await page.locator(".cm-editor").count(), 1);
        }
        if (question.type === "numeric") {
          const label = await page.locator(`label[for="numeric-${question.id}"]`).innerText();
          assert.ok(label.includes(question.answer_count === 2 ? "两个" : String(question.answer_count)));
        }
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - innerWidth);
        assert.ok(overflow <= 0, `desktop overflow: ${question.id}: ${overflow}`);
        if (question.id === "q11") await capture(page, "mc-desktop.png");
      } finally {
        await fresh.close();
      }
    }
    if (process.env.MC_SKIP_COLD_START !== "1") {
      assert.deepEqual(errors, []);
      console.log("PASS: 18 isolated direct links; complete visible inputs; no premature answer reveal.");
    }

    const context = await browser.newContext({ viewport: { width: 1365, height: 900 } });
    try {
      const page = await context.newPage();
      observe(page, errors);
      await open(page, "q01");
      assert.equal(await page.getByRole("radio").count(), 4);
      await page.getByRole("radio").nth(2).check();
      await page.getByRole("button", { name: "提交判断" }).click();
      await page.getByText("再对照定义看一眼", { exact: true }).waitFor();
      await page.getByRole("radio").nth(0).check();
      await page.getByRole("button", { name: "提交判断" }).click();
      await page.getByText("判断正确", { exact: true }).waitFor();

      await open(page, "q11");
      await page.locator("#numeric-q11").fill("0, 0");
      await page.getByRole("button", { name: "核对答案" }).click();
      assert.ok((await page.locator(".feedback").innerText()).includes("6个数值"));
      await page.locator("#numeric-q11").fill("0, 0, 0, 0, 0, 0");
      await page.getByRole("button", { name: "核对答案" }).click();
      await page.getByText("拆开算一次", { exact: true }).waitFor();
      await page.locator("#numeric-q11").fill("-10, -10, 8, 7.29, -9, 3");
      await page.getByRole("button", { name: "核对答案" }).click();
      await page.getByText("计算正确", { exact: true }).waitFor();

      await open(page, "q13");
      await page.getByRole("button", { name: "写完了，对照评分要点" }).click();
      assert.ok((await page.locator(".feedback").innerText()).includes("先写下"));
      await page.locator("#open-q13").fill(
        "持续任务两条路线回报不同；将目标改为自然终止后删去奖励尾部，偏好会改变。");
      await page.getByRole("button", { name: "写完了，对照评分要点" }).click();
      await page.locator(".rubric").waitFor();
      assert.ok((await page.locator(".reference-answer").innerText()).includes("0.729"));
      await page.locator("#score-q13").fill("6");
      await page.getByRole("button", { name: "记录自评" }).click();

      for (const id of ["q16", "q17", "q18"]) {
        await open(page, id);
        const reference = await api("/api/reveal", { chapter: chapterId, id });
        const question = bank.questions.find((entry) => entry.id === id);
        if (id === "q16") {
          await page.getByRole("button", { name: "运行公开测试" }).click();
          await page.getByText(/公开测试 [0-4] \/ 5 通过/).waitFor({ timeout: 90000 });
          await typeCode(page, "def discounted_returns(:\n    pass");
          await page.getByRole("button", { name: "运行公开测试" }).click();
          await page.locator(".syntax-diagnostic").waitFor({ timeout: 90000 });
        }
        await typeCode(page, reference.reference_solution);
        await page.getByRole("button", { name: "运行公开测试" }).click();
        const complete = `公开测试 ${question.tests.length} / ${question.tests.length} 通过`;
        await page.getByText(complete, { exact: true }).waitFor({ timeout: 90000 });
        await page.reload({ waitUntil: "networkidle" });
        assert.ok((await page.locator(".feedback").innerText()).includes(complete));
      }
      assert.equal(await page.locator("#progress-count").innerText(), "6 / 18");
      await page.locator("#chapter-select").selectOption("easyrl-2.1-2.2.2");
      await page.getByText("0 / 12", { exact: true }).waitFor();
      await page.locator("#chapter-select").selectOption(chapterId);
      await page.getByText("6 / 18", { exact: true }).waitFor();
      console.log("PASS: wrong/correct grading, open self-review, 17 Pyodide tests, syntax errors, refresh and chapter isolation.");
    } finally {
      await context.close();
    }

    const numericCases = [
      [chapterId, "q07", ".72, .8, 2"],
      [chapterId, "q08", "3, 2.8"],
      [chapterId, "q09", ".125, .3125"],
      [chapterId, "q10", ".04, .84, .04, .04, .04"],
      [chapterId, "q12", "2.71, 7.29"],
      ["easyrl-2.3", "q08", "2.8, 3.9"],
      ["easyrl-2.1-2.2.2", "q05", "3.25"],
    ];
    for (const [id, question, value] of numericCases) {
      const fresh = await browser.newContext({ viewport: { width: 1365, height: 900 } });
      try {
        const page = await fresh.newPage();
        observe(page, errors);
        await page.goto(`${base}?chapter=${id}&question=${question}`, { waitUntil: "networkidle" });
        await page.locator(`#numeric-${question}`).fill(value);
        await page.getByRole("button", { name: "核对答案" }).click();
        await page.getByText("计算正确", { exact: true }).waitFor();
      } finally {
        await fresh.close();
      }
    }
    console.log("PASS: every MC numeric question and legacy two-value/scalar formats submit correctly.");

    const mobile = await browser.newContext({
      viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
    });
    try {
      const page = await mobile.newPage();
      observe(page, errors);
      for (const question of bank.questions) {
        await open(page, question.id);
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - innerWidth);
        assert.ok(overflow <= 0, `mobile overflow: ${question.id}: ${overflow}`);
        if (question.id === "q11") await capture(page, "mc-mobile.png");
        if (question.id === "q17") await capture(page, "mc-code-mobile.png");
      }
      console.log("PASS: all 18 mobile views fit; formulas and code render without page overflow.");
    } finally {
      await mobile.close();
    }
    assert.deepEqual(errors, []);
    console.log("PASS: no browser page errors or missing application assets.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
