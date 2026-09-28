"use strict";

const STORAGE_KEY = "rl-learning-lab-progress-v2";
const LEGACY_STORAGE_KEY = "rl-checkpoint-easyrl-2-2-2-v1";
const CATEGORY_LABELS = {
  concept: "概念",
  "hand-calculation": "手算",
  application: "解释",
  coding: "编码",
};
const TYPE_LABELS = {
  choice: "概念选择",
  numeric: "纸笔计算",
  open: "文字解释 · 自评",
  code: "Python · 公开测试",
};
const FULL_FORMULAE = [
  ["V(s) = R(s) + γ·Σ_{s'} P(s'|s)·V(s')", String.raw`V(s)=R(s)+\gamma\sum_{s'}P(s'\mid s)V(s')`],
  ["G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + …", String.raw`G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots`],
  ["G_t = r_{t+1} + r_{t+2} + r_{t+3} + …", String.raw`G_t=r_{t+1}+r_{t+2}+r_{t+3}+\cdots`],
  ["G_t = γ·r_{t+1} + γ²·r_{t+2} + …", String.raw`G_t=\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots`],
  ["G_t = R(s_t) + γ·R(s_{t+1}) + …", String.raw`G_t=R(s_t)+\gamma R(s_{t+1})+\cdots`],
  ["G_0 = r_1 + γ·r_2 + γ²·r_3 + γ³·r_4", String.raw`G_0=r_1+\gamma r_2+\gamma^2r_3+\gamma^3r_4`],
  ["G_t = r_{t+1} + γ·G_{t+1}", String.raw`G_t=r_{t+1}+\gamma G_{t+1}`],
  ["V(s) = E[G_t | S_t = s]", String.raw`V(s)=\mathbb E[G_t\mid S_t=s]`],
  ["G_0 = Σ_k γ^k·rewards[k]", String.raw`G_0=\sum_k\gamma^k\,\mathrm{rewards}[k]`],
  ["Σ_{k=0}^{19} γ^k", String.raw`\sum_{k=0}^{19}\gamma^k`],
  ["V1 = R + γ·P·V0", String.raw`V_1=R+\gamma PV_0`],
  ["V' = R + γ·P·V", String.raw`V'=R+\gamma PV`],
  ["transitions[i][j] = P(s_j | s_i)", String.raw`\mathrm{transitions}[i][j]=P(s_j\mid s_i)`],
].sort((a, b) => b[0].length - a[0].length);
const $ = (selector) => document.querySelector(selector);
const dom = {
  loading: $("#loading-view"),
  error: $("#error-view"),
  errorMessage: $("#error-message"),
  question: $("#question-view"),
  summary: $("#summary-view"),
  nav: $("#question-nav"),
  count: $("#progress-count"),
  bar: $("#progress-track"),
  fill: $("#progress-fill"),
  kicker: $("#workspace-kicker"),
  status: $("#workspace-status"),
};
let data = null;
let config = { vendor_available: false };
let state = { current: "q01", responses: {} };
let chapterStates = {};
let chapterCatalog = { default_chapter_id: "easyrl-2.1-2.2.2", chapters: [] };
let currentChapterId = "";
let activeEditor = null;
let view = "question";
let busy = false;
let inlineError = "";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

// Only trusted question-bank text enters this formatter. User answers and
// Python output always remain plain textContent, never HTML.
function rich(tag, className, text) {
  const node = element(tag, className);
  const source = String(text);
  const token = /(?:[GgRrSsVv]_(?:\{[^}]+\}|[A-Za-z0-9]+))|(?:Σ_\{[^}]+\})|(?:γ[²³⁴])|(?:V[01](?:\([AB]\))?)/g;
  let cursor = 0;
  while (cursor < source.length) {
    let next = null;
    for (const [literal, tex] of FULL_FORMULAE) {
      const index = source.indexOf(literal, cursor);
      if (index !== -1 && (!next || index < next.index ||
          (index === next.index && literal.length > next.raw.length))) {
        next = { index, raw: literal, tex, full: true };
      }
    }
    token.lastIndex = cursor;
    const match = token.exec(source);
    if (match && (!next || match.index < next.index)) {
      const raw = match[0];
      let tex = raw.replaceAll("γ", "\\gamma ").replaceAll("Σ", "\\sum ");
      tex = tex.replaceAll("²", "^{2}").replaceAll("³", "^{3}").replaceAll("⁴", "^{4}");
      tex = tex.replace(/^V([01])/, "V_{$1}");
      next = { index: match.index, raw, tex, full: false };
    }
    if (!next) break;
    if (next.index > cursor) node.append(document.createTextNode(source.slice(cursor, next.index)));
    const math = element("span", next.full && next.raw.length > 31
      ? "inline-math math-equation" : "inline-math");
    if (window.katex) {
      window.katex.render(next.tex, math, { throwOnError: false, trust: false, strict: "ignore" });
    } else {
      math.textContent = next.raw;
    }
    node.append(math);
    cursor = next.index + next.raw.length;
  }
  if (cursor < source.length) node.append(document.createTextNode(source.slice(cursor)));
  return node;
}

function button(text, className, callback, disabled = false) {
  const node = element("button", className, text);
  node.type = "button";
  node.disabled = disabled;
  node.addEventListener("click", callback);
  return node;
}

function append(parent, ...children) {
  parent.append(...children);
  return parent;
}

function own(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function loadSaved() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (parsed?.version === 2 && parsed.chapters && typeof parsed.chapters === "object") {
      chapterStates = parsed.chapters;
      if (chapterCatalog.chapters.some((chapter) => chapter.id === parsed.selectedChapterId)) {
        currentChapterId = parsed.selectedChapterId;
      }
    } else {
      const legacy = JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY));
      if (legacy && typeof legacy.current === "string" && legacy.responses &&
          typeof legacy.responses === "object") {
        chapterStates[chapterCatalog.default_chapter_id] = legacy;
      }
    }
    state = validChapterState(chapterStates[currentChapterId]) ||
      { current: "q01", responses: {} };
  } catch (_) {
    // A disabled or damaged localStorage should not prevent studying.
  }
}

function validChapterState(candidate) {
  if (!candidate || typeof candidate !== "object" ||
      typeof candidate.current !== "string" ||
      !candidate.responses || typeof candidate.responses !== "object" ||
      Array.isArray(candidate.responses)) return null;
  return candidate;
}

function persist() {
  try {
    chapterStates[currentChapterId] = state;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      version: 2, selectedChapterId: currentChapterId, chapters: chapterStates,
    }));
  } catch (_) {
    // Storage is optional, and browser grading remains usable without it.
  }
}

function record(id) {
  if (!own(state.responses, id) || !state.responses[id] ||
      typeof state.responses[id] !== "object") {
    state.responses[id] = {};
  }
  return state.responses[id];
}

function safeScore(value, maximum) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(maximum, Math.max(0, number)) : 0;
}

function isComplete(question) {
  const answer = record(question.id);
  if (question.grading_mode === "auto") return answer.result && typeof answer.result.score === "number";
  if (question.grading_mode === "self") return own(answer, "selfScore") && Number.isFinite(Number(answer.selfScore));
  return answer.codeResult && typeof answer.codeResult.passed === "number";
}

function scored(question) {
  const answer = record(question.id);
  if (question.grading_mode === "auto") return safeScore(answer.result?.score, question.max_score);
  if (question.grading_mode === "self") return safeScore(answer.selfScore, question.max_score);
  return safeScore(answer.codeResult?.score, question.max_score);
}

async function api(path, body) {
  if (window.RL_SITE_DATA) return staticApi(path, body);
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败（${response.status}）`);
  return result;
}

function staticApi(path, body) {
  const site = window.RL_SITE_DATA;
  const url = new URL(path, window.location.href);
  if (url.pathname === "/api/chapters") {
    return {
      default_chapter_id: site.default_chapter_id,
      chapters: site.chapters.map(({ id, title, reference, count, total_score }) =>
        ({ id, title, reference, count, total_score })),
    };
  }
  if (url.pathname === "/api/config") return { vendor_available: true };
  const id = body?.chapter || url.searchParams.get("chapter") || site.default_chapter_id;
  const chapter = site.chapters.find((entry) => entry.id === id);
  if (!chapter) throw new Error(`未知章节：${id}`);
  if (url.pathname === "/api/questions") return chapter.payload;
  if (url.pathname === "/api/reveal") {
    const result = chapter.reveals[body?.id];
    if (!result) throw new Error("本题没有可展开的参考内容。");
    return result;
  }
  if (url.pathname === "/api/grade") {
    const rule = chapter.answers[body?.id];
    const question = chapter.payload.questions.find((item) => item.id === body?.id);
    if (!rule || !question) throw new Error("该题不能自动判分。");
    if (rule.type === "choice") {
      const chosen = Number(body.answer);
      if (!Number.isInteger(chosen) || chosen < 0 || chosen >= question.options.length) {
        throw new Error("请选择一个有效选项。");
      }
      const expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[rule.answer];
      const correct = chosen === rule.answer;
      return {
        correct, score: correct ? question.max_score : 0,
        max_score: question.max_score, expected,
        feedback: correct ? "回答正确。" : `回答不正确，正确答案是 ${expected}。`,
      };
    }
    const got = Array.isArray(body.answer) ? body.answer : [body.answer];
    const expectedValues = Array.isArray(rule.expected) ? rule.expected : [rule.expected];
    if (got.length !== expectedValues.length ||
        !got.every((item) => typeof item === "number" && Number.isFinite(item))) {
      throw new Error("请按题目要求输入有限数值及正确的个数。");
    }
    const correct = got.every((value, index) => {
      const expected = expectedValues[index];
      return Math.abs(value - expected) <=
        Math.max(rule.tolerance, rule.tolerance * Math.max(Math.abs(value), Math.abs(expected)));
    });
    return {
      correct, score: correct ? question.max_score : 0,
      max_score: question.max_score, expected: rule.expected,
      feedback: correct ? "数值正确。" : "数值不正确。",
    };
  }
  throw new Error(`未知接口：${url.pathname}`);
}

async function initialize() {
  dom.loading.hidden = false;
  dom.error.hidden = true;
  dom.question.hidden = true;
  dom.summary.hidden = true;
  try {
    chapterCatalog = await api("/api/chapters").catch(() => ({
      default_chapter_id: "easyrl-2.1-2.2.2",
      chapters: [{ id: "easyrl-2.1-2.2.2", title: "马尔可夫过程与贝尔曼方程" }],
    }));
    currentChapterId = chapterCatalog.default_chapter_id;
    loadSaved();
    const [loaded, available] = await Promise.all([
      api(`/api/questions?chapter=${encodeURIComponent(currentChapterId)}`),
      api("/api/config").catch(() => ({ vendor_available: false })),
    ]);
    if (!loaded || !Array.isArray(loaded.questions) || loaded.questions.length === 0) {
      throw new Error("题库返回了空内容");
    }
    data = loaded;
    config = available;
    if (!loaded.questions.some((question) => question.id === state.current)) {
      state.current = loaded.questions[0].id;
    }
    persist();
    updateChapterHeading();
    dom.loading.hidden = true;
    render();
  } catch (error) {
    dom.loading.hidden = true;
    dom.error.hidden = false;
    dom.errorMessage.textContent = error.message || String(error);
    dom.status.textContent = "连接失败";
  }
}

function updateChapterHeading() {
  const chapter = chapterCatalog.chapters.find((item) => item.id === currentChapterId);
  if (!chapter) return;
  const title = chapter.title.replace(/\s*·.*/, "");
  const accent = document.querySelector(".hero h1 em");
  if (accent) accent.textContent = title;
  const edition = document.querySelector(".edition");
  if (edition) edition.textContent = chapter.reference || title;
  document.title = `${title} · RL 学习检查点`;
}

function render() {
  if (!data) return;
  renderNavigation();
  if (view === "summary") {
    if (activeEditor) { activeEditor.destroy(); activeEditor = null; }
    dom.question.hidden = true;
    dom.summary.hidden = false;
    dom.kicker.textContent = "LEARNING REVIEW / 01";
    dom.status.textContent = "本地学习记录";
    renderSummary();
  } else {
    dom.summary.hidden = true;
    dom.question.hidden = false;
    renderQuestion(data.questions.find((question) => question.id === state.current));
  }
}

function renderNavigation() {
  const total = data.questions.length;
  const done = data.questions.filter(isComplete).length;
  dom.count.textContent = `${done} / ${total}`;
  dom.bar.setAttribute("aria-valuemax", String(total));
  dom.bar.setAttribute("aria-valuenow", String(done));
  dom.fill.style.width = `${(done / total) * 100}%`;
  dom.nav.replaceChildren();
  const chapterBlock = element("div", "chapter-picker");
  const chapterLabel = element("label", "answer-label", "学习章节");
  chapterLabel.htmlFor = "chapter-select";
  const chapterSelect = element("select", "chapter-select");
  chapterSelect.id = chapterLabel.htmlFor;
  chapterSelect.disabled = busy;
  for (const chapter of chapterCatalog.chapters) {
    const option = element("option", "", chapter.title);
    option.value = chapter.id;
    chapterSelect.append(option);
  }
  chapterSelect.value = currentChapterId;
  chapterSelect.addEventListener("change", () => selectChapter(chapterSelect.value));
  append(chapterBlock, chapterLabel, chapterSelect);
  dom.nav.append(chapterBlock);
  const categories = [...new Set(data.questions.map((question) => question.category))];
  for (const category of categories) {
    const label = CATEGORY_LABELS[category] || category;
    const questions = data.questions.filter((question) => question.category === category);
    if (!questions.length) continue;
    const group = element("div", "nav-group");
    append(group, element("p", "nav-group-heading",
      `${label} / ${questions[0].id.slice(1)}—${questions.at(-1).id.slice(1)}`));
    for (const question of questions) {
      const complete = isComplete(question);
      const active = view === "question" && state.current === question.id;
      const entry = button("", `nav-question${active ? " active" : ""}${complete ? " complete" : ""}`,
        () => selectQuestion(question.id));
      entry.setAttribute("aria-label", `${question.id.slice(1)} ${question.title}${complete ? "，已作答" : ""}`);
      if (active) entry.setAttribute("aria-current", "step");
      append(entry,
        element("span", "nav-index", question.id.slice(1)),
        rich("span", "nav-name", question.title),
        element("span", "nav-indicator", complete ? "●" : ""));
      group.append(entry);
    }
    dom.nav.append(group);
  }
}

async function selectChapter(id) {
  if (busy || id === currentChapterId ||
      !chapterCatalog.chapters.some((chapter) => chapter.id === id)) return;
  const previous = currentChapterId;
  chapterStates[previous] = state;
  dom.loading.hidden = false;
  dom.question.hidden = true;
  dom.summary.hidden = true;
  try {
    const loaded = await api(`/api/questions?chapter=${encodeURIComponent(id)}`);
    if (!loaded?.questions?.length) throw new Error("这个章节还没有题目。");
    currentChapterId = id;
    data = loaded;
    state = validChapterState(chapterStates[id]) ||
      { current: loaded.questions[0].id, responses: {} };
    if (!loaded.questions.some((question) => question.id === state.current)) {
      state.current = loaded.questions[0].id;
    }
    view = "question";
    persist();
    updateChapterHeading();
    render();
  } catch (error) {
    inlineError = error.message || "章节暂时无法打开。";
    currentChapterId = previous;
    state = chapterStates[previous];
    render();
  } finally {
    dom.loading.hidden = true;
  }
}

function selectQuestion(id) {
  if (!data.questions.some((question) => question.id === id)) return;
  state.current = id;
  view = "question";
  inlineError = "";
  persist();
  render();
  if (window.innerWidth < 781) dom.question.scrollIntoView({ block: "start" });
}

function sectionHeader(question) {
  const heading = element("div", "question-meta");
  append(heading,
    element("span", "", `${question.id.slice(1)} / ${String(data.questions.length).padStart(2, "0")}`),
    element("span", "meta-divider", "—"),
    element("span", "", TYPE_LABELS[question.type] || question.type),
    element("span", "meta-divider", "·"),
    element("span", "", `${question.max_score} 分`));
  return heading;
}

function renderQuestion(question) {
  if (!question) return;
  if (activeEditor) { activeEditor.destroy(); activeEditor = null; }
  const answer = record(question.id);
  dom.kicker.textContent = `KNOWLEDGE CHECK / ${question.id.slice(1)}`;
  dom.status.textContent = isComplete(question) ? "已记录 · 可以再试" : "进行中";
  const fragment = document.createDocumentFragment();
  append(fragment,
    sectionHeader(question),
    rich("h2", "question-title", question.title),
    rich("p", "question-prompt", question.prompt));

  if (data.industry_case && question.id === (data.industry_case_question_id || "q11")) {
    fragment.append(renderIndustryCase(data.industry_case));
  }
  if (question.type === "choice") fragment.append(renderChoice(question, answer));
  if (question.type === "numeric") fragment.append(renderNumeric(question, answer));
  if (question.type === "open") fragment.append(renderOpen(question, answer));
  if (question.type === "code") fragment.append(renderCode(question, answer));

  if (inlineError) fragment.append(feedback("尚未完成", inlineError, "warning"));
  const footer = element("div", "question-footer");
  const index = data.questions.findIndex((item) => item.id === question.id);
  append(footer,
    button("← 上一题", "button button-quiet", () => selectQuestion(data.questions[index - 1].id), index === 0),
    button(index === data.questions.length - 1 ? "查看复盘 ↗" : "下一题 →",
      "button", () => index === data.questions.length - 1
        ? showSummary()
        : selectQuestion(data.questions[index + 1].id)));
  fragment.append(footer);
  dom.question.replaceChildren(fragment);
}

// Mirrors the strict GSM8K extract_solution/compute_score branch in this
// checkout; this is a tiny scoring demo, NOT a browser inference/torch run.
function gsm8kResult(sample) {
  const text = sample.response.slice(-300);
  const matches = [...text.matchAll(/#### (-?[0-9.,]+)/g)];
  const extracted = matches.length ? matches[matches.length - 1][1].replaceAll(",", "") : null;
  return {
    extracted,
    score: extracted !== null && extracted === sample.ground_truth ? 1 : 0,
  };
}

function returnVector(sample, gamma) {
  const { extracted, score } = gsm8kResult(sample);
  const mask = sample.positions.map((_, index) => index < sample.valid_length ? 1 : 0);
  const rewards = mask.map((valid, index) => valid && index === sample.valid_length - 1
    ? score : 0);
  const returns = Array(mask.length).fill(0);
  let running = 0;
  for (let index = mask.length - 1; index >= 0; index--) {
    if (!mask[index]) {
      running = 0;
      continue;
    }
    running = rewards[index] + gamma * running;
    returns[index] = running;
  }
  return { extracted, score, mask, rewards, returns };
}

function displayedNumber(value) {
  return String(Number(value.toFixed(4)));
}

function caseTable(labels, rows, className) {
  const wrapper = element("div", "case-table-wrap");
  const scroller = element("div", "case-table-scroll");
  scroller.tabIndex = 0;
  scroller.setAttribute("role", "region");
  scroller.setAttribute("aria-label", `${labels[0]}到${labels[labels.length - 1]}的批次数据，可横向滚动`);
  const table = element("table", `case-table ${className}`);
  const head = element("thead");
  const header = element("tr");
  for (const label of labels) header.append(element("th", "", label));
  head.append(header);
  table.append(head);
  const body = element("tbody");
  for (const cells of rows) {
    const row = element("tr");
    for (const [index, cell] of cells.entries()) {
      row.append(element(index === 0 ? "th" : "td", "", String(cell)));
    }
    body.append(row);
  }
  table.append(body);
  scroller.append(table);
  wrapper.append(scroller, element("p", "case-scroll-hint", "← 横向滚动表格可看完全部列 →"));
  return wrapper;
}

function renderIndustryCase(caseData) {
  const section = element("section", "industry-case");
  section.setAttribute("aria-label", "真实源码与教学数据案例");
  const header = element("div", "industry-heading");
  append(header, element("span", "case-eyebrow", `CASE / ${caseData.project.toUpperCase()} · ${caseData.commit}`),
    element("h3", "", caseData.title),
    element("p", "", caseData.scope));
  section.append(header);

  const steps = element("div", "case-steps");
  for (const stage of caseData.stages) {
    const detail = element("details", "case-step");
    if (stage.number === "01" || stage.number === "05") detail.open = true;
    append(detail, element("summary", "", `${stage.number} / ${stage.title}`),
      element("p", "", stage.explanation),
      element("code", "case-path", `${stage.path}:${stage.lines}`),
      element("pre", "case-code", stage.code),
      element("p", "case-code-note", stage.code_note));
    steps.append(detail);
  }
  section.append(steps);

  const lab = element("div", "case-lab");
  append(lab, element("span", "case-eyebrow", "DATA WALKTHROUGH / B = 4, T = 4"),
    element("h4", "", "把四条回答放进同一个 batch"),
    element("p", "case-disclaimer", caseData.data_notice));

  let gamma = Number(state.caseGamma);
  if (!Number.isFinite(gamma) || gamma < 0 || gamma > 1 ||
      state.caseGamma === undefined) gamma = 0.9;
  let selected = caseData.samples.find((item) => item.id === state.caseSelected)
    || caseData.samples[0];
  const control = element("div", "case-control");
  const label = element("label", "", "折扣因子 γ");
  label.htmlFor = "case-gamma";
  const slider = element("input");
  slider.id = "case-gamma";
  slider.type = "range";
  slider.min = "0";
  slider.max = "1";
  slider.step = "0.1";
  slider.value = String(gamma);
  const gammaValue = element("output", "", gamma.toFixed(1));
  gammaValue.htmlFor = slider.id;
  const configNote = element("span", "case-config", "演示起点 0.9；verl 源码默认 1.0。");
  append(control, label, slider, gammaValue, configNote);
  lab.append(control);

  const batchFlow = element("div", "case-batch");
  lab.append(batchFlow);
  const overview = element("div", "case-overview");
  const detailHost = element("div", "case-detail");
  function updateCase() {
    gammaValue.textContent = gamma.toFixed(1);
    const inputRows = caseData.samples.map((sample) => {
      const { extracted, score } = gsm8kResult(sample);
      return [
        sample.id, sample.prompt, sample.ground_truth, sample.response,
        extracted ?? "无匹配", String(score),
      ];
    });
    const outputRows = caseData.samples.map((sample) => {
      const vectors = returnVector(sample, gamma);
      const asVector = (values) => `[${values.map(displayedNumber).join(", ")}]`;
      return [
        sample.id,
        String(sample.valid_length),
        asVector(vectors.mask),
        asVector(vectors.rewards),
        asVector(vectors.rewards),
        asVector(vectors.returns),
      ];
    });
    batchFlow.replaceChildren(
      element("h5", "", "① 数据行 → 按源码规则给整条回答打分"),
      element("p", "case-batch-note",
        "四条记录共享 data_source=openai/gsm8k；ground_truth 对应 reward_model.ground_truth。"
        + "提取器只认最后一处“#### 数字”；D 行的数字正确但格式不匹配。"),
      caseTable(["样本", "prompt", "ground_truth", "生成 response", "提取答案", "score"], inputRows, "case-input-table"),
      element("h5", "", "② 右侧补齐 → 标量落点 → 逐 token 回报"),
      element("p", "case-batch-note",
        "所有向量形状 [B=4,T=4]；rm_scores 是末个有效位置承载标量分数。"
        + "此配置中 token_level_rewards=rm_scores；returns 自末尾倒序递推，padding 列为 0。"),
      caseTable(
        ["样本", "有效长", "response_mask", "rm_scores", "token_level_rewards", `returns · γ=${gamma.toFixed(1)}`],
        outputRows, "case-output-table"),
    );
    overview.replaceChildren();
    for (const sample of caseData.samples) {
      const vectors = returnVector(sample, gamma);
      const choice = button("", `case-sample${selected.id === sample.id ? " selected" : ""}`, () => {
        selected = sample;
        state.caseSelected = sample.id;
        persist();
        updateCase();
      });
      choice.setAttribute("aria-pressed", String(selected.id === sample.id));
      append(choice,
        element("span", "case-sample-id", `样本 ${sample.id}`),
        element("span", "case-sample-response", `“${sample.response}” · 判分 ${vectors.score}`),
        element("strong", "", `G₀ = ${displayedNumber(vectors.returns[0])}`));
      overview.append(choice);
    }
    const vectors = returnVector(selected, gamma);
    detailHost.replaceChildren();
    const caption = element("p", "case-caption",
      `样本 ${selected.id}：prompt「${selected.prompt}」→ response「${selected.response}」；`
      + `真值 ${selected.ground_truth}，规则提取 ${vectors.extracted ?? "无匹配"}，评分 ${vectors.score}。`);
    detailHost.append(caption);
    const scroll = element("div", "case-table-scroll");
    const table = element("table", "case-table");
    const head = element("thead");
    const headRow = element("tr");
    append(headRow, element("th", "", "位置"));
    for (let index = 0; index < selected.positions.length; index++) {
      headRow.append(element("th", "", `t=${index}`));
    }
    head.append(headRow);
    table.append(head);
    const body = element("tbody");
    for (const [labelText, values] of [
      ["回答片段*", selected.positions],
      ["response_mask", vectors.mask],
      ["rm_scores", vectors.rewards],
      ["token_level_rewards", vectors.rewards],
      ["回报 Gₜ", vectors.returns.map(displayedNumber)],
    ]) {
      const row = element("tr");
      row.append(element("th", "", labelText));
      for (const value of values) row.append(element("td", "", value));
      body.append(row);
    }
    table.append(body);
    scroll.append(table);
    detailHost.append(scroll);
    const calculations = element("ol", "case-calculations");
    for (let index = selected.valid_length - 1; index >= 0; index--) {
      const next = index === selected.valid_length - 1 ? 0 : vectors.returns[index + 1];
      calculations.append(element("li", "",
        `t=${index}：G${index} = ${vectors.rewards[index]} + ${gamma.toFixed(1)} × ${displayedNumber(next)} = ${displayedNumber(vectors.returns[index])}`));
    }
    detailHost.append(calculations);
  }
  slider.addEventListener("input", () => {
    gamma = Number(slider.value);
    state.caseGamma = gamma;
    persist();
    updateCase();
  });
  lab.append(element("h5", "case-drill-title", "③ 点选一条回答：看回报怎样倒序填回每个位置"),
    overview, detailHost);
  updateCase();
  append(lab,
    element("p", "case-footnote", "*片段仅用于显示四个位置，不是 tokenizer 真实切分；不同模型的 token 数、有效长和补齐宽度会变化。本页无 GPU 训练，交互只按所列源码规则计算。"),
    element("p", "case-exercise-bridge",
      "直接用于本题：把 A 行有效奖励 [0, 0, 0, 1] 输入 compute_return(..., 0.9)，"
      + "应得到 0.729；B 行先按 mask 去掉 padding，输入 [0, 1]，应得到 0.9。"
      + "你写的函数只算一个 G₀；源码按 batch 倒序算出全部位置的 returns。"),
    rich("p", "case-boundary",
      "本例的 G_0 是某条已生成回答的采样回报，不是 V(s)；§2.2.2 的贝尔曼方程还要对可能后继状态取期望。此处先不学习 KL、优势标准化和策略更新。"));
  section.append(lab);
  return section;
}

function feedback(title, text, kind) {
  const panel = element("div", `feedback ${kind || ""}`);
  panel.setAttribute("role", "status");
  append(panel, element("strong", "", title), element("p", "", text));
  return panel;
}

function renderChoice(question, answer) {
  const wrapper = element("div", "answer-zone");
  append(wrapper, element("span", "answer-label", "选出最准确的一项"));
  const options = element("div", "options");
  for (const [index, option] of question.options.entries()) {
    const label = element("label", "option");
    const input = element("input");
    input.type = "radio";
    input.name = `answer-${question.id}`;
    input.value = String(index);
    input.checked = String(answer.draft) === String(index);
    input.addEventListener("change", () => {
      answer.draft = index;
      delete answer.result;
      inlineError = "";
      persist();
      renderNavigation();
    });
    append(label, input, element("span", "option-letter", "ABCD"[index] || String(index + 1)),
      rich("span", "option-text", option));
    options.append(label);
  }
  wrapper.append(options);
  wrapper.append(button(busy ? "正在判分…" : "提交判断", "button button-primary",
    () => submitAuto(question, answer.draft), busy));
  if (answer.result) wrapper.append(feedback(
    answer.result.correct ? "判断正确" : "再对照定义看一眼",
    `${answer.result.feedback}  本题 ${answer.result.score} / ${question.max_score} 分。`,
    answer.result.correct ? "success" : "warning"));
  return wrapper;
}

function renderNumeric(question, answer) {
  const wrapper = element("div", "answer-zone");
  const label = element("label", "answer-label", question.answer_type === "list" ? "输入两个数值" : "输入计算结果");
  label.htmlFor = `numeric-${question.id}`;
  const input = element("input", "answer-field numeric-field");
  input.id = label.htmlFor;
  input.type = "text";
  input.autocomplete = "off";
  input.inputMode = "decimal";
  input.placeholder = question.answer_type === "list" ? "例如：2.5, 3.75" : "例如：3.25";
  input.value = typeof answer.draft === "string" ? answer.draft : "";
  input.addEventListener("input", () => {
    answer.draft = input.value;
    delete answer.result;
    inlineError = "";
    persist();
    renderNavigation();
  });
  append(wrapper, label, input,
    element("p", "field-hint", question.answer_type === "list"
      ? "按 [V₁(A), V₁(B)] 的顺序，用逗号分隔；也可输入 [2.5, 3.75]。"
      : "允许小数；先手算，再交给程序核对。"));
  wrapper.append(button(busy ? "正在核对…" : "核对答案", "button button-primary",
    () => submitAuto(question, answer.draft), busy));
  if (answer.result) {
    const expected = Array.isArray(answer.result.expected)
      ? `[${answer.result.expected.join(", ")}]` : answer.result.expected;
    wrapper.append(feedback(answer.result.correct ? "计算正确" : "拆开算一次",
      `${answer.result.feedback}  参考结果：${expected}。本题 ${answer.result.score} / ${question.max_score} 分。`,
      answer.result.correct ? "success" : "warning"));
  }
  return wrapper;
}

function parseNumeric(question, raw) {
  const text = String(raw ?? "").trim();
  if (!text) throw new Error("请先输入你算出的结果。");
  if (question.answer_type === "number") {
    const number = Number(text);
    if (!Number.isFinite(number)) throw new Error("请输入一个有限数值。");
    return number;
  }
  const pieces = text.replace(/^\s*\[/, "").replace(/\]\s*$/, "").split(/[,，;；\s]+/).filter(Boolean);
  if (pieces.length !== 2) throw new Error("请按顺序输入两个数，例如 [2.5, 3.75]。");
  const numbers = pieces.map(Number);
  if (!numbers.every(Number.isFinite)) throw new Error("列表中每一项都必须是有限数值。");
  return numbers;
}

async function submitAuto(question, raw) {
  if (busy) return;
  let answer;
  try {
    if (question.type === "choice") {
      if (raw === undefined || raw === null || raw === "") throw new Error("请先选择一个选项。");
      answer = Number(raw);
    } else {
      answer = parseNumeric(question, raw);
    }
  } catch (error) {
    inlineError = error.message;
    renderQuestion(question);
    return;
  }
  busy = true;
  inlineError = "";
  renderQuestion(question);
  try {
    record(question.id).result = await api("/api/grade", { id: question.id, answer });
    persist();
  } catch (error) {
    inlineError = error.message;
  } finally {
    busy = false;
    render();
  }
}

function renderRubric(question, answer) {
  const rubric = answer.revealed;
  const section = element("section", "rubric");
  append(section, element("h3", "", question.type === "open" ? "对照评分要点" : "参考实现与检查点"));
  const list = element("ol");
  for (const item of rubric.rubric || []) {
    list.append(element("li", "", `${item.criterion}（${item.points} 分）`));
  }
  section.append(list);
  const detail = element("p", "reference-answer", question.type === "open"
    ? rubric.model_answer : rubric.reference_solution);
  section.append(detail);
  if (question.type === "open") {
    const scoreRow = element("div", "self-score");
    const label = element("label", "answer-label", "我的自评分");
    label.htmlFor = `score-${question.id}`;
    const input = element("input", "answer-field");
    input.id = label.htmlFor;
    input.type = "number";
    input.min = "0";
    input.max = String(question.max_score);
    input.step = "1";
    input.value = own(answer, "selfScore") ? String(answer.selfScore) : "";
    input.placeholder = `0—${question.max_score}`;
    append(scoreRow, label, input,
      button("记录自评", "button", () => {
        const number = Number(input.value);
        if (input.value.trim() === "" || !Number.isInteger(number) ||
            number < 0 || number > question.max_score) {
          inlineError = `请填写 0—${question.max_score} 的整数分数。`;
          renderQuestion(question);
          return;
        }
        answer.selfScore = number;
        inlineError = "";
        persist();
        render();
      }));
    section.append(scoreRow);
    if (own(answer, "selfScore")) {
      section.append(feedback("已记录自评", `${answer.selfScore} / ${question.max_score} 分。这是你的自我判断，不是机器判分。`, "success"));
    }
  }
  return section;
}

async function reveal(question, answer) {
  if (busy) return;
  if (question.type === "open" && !String(answer.draft || "").trim()) {
    inlineError = "先写下自己的解释，再展开参考要点。";
    renderQuestion(question);
    return;
  }
  busy = true;
  inlineError = "";
  renderQuestion(question);
  try {
    answer.revealed = await api("/api/reveal", { id: question.id });
    persist();
  } catch (error) {
    inlineError = error.message;
  } finally {
    busy = false;
    renderQuestion(question);
  }
}

function renderOpen(question, answer) {
  const wrapper = element("div", "answer-zone");
  const label = element("label", "answer-label", "先写出自己的解释");
  label.htmlFor = `open-${question.id}`;
  const input = element("textarea", "answer-field");
  input.id = label.htmlFor;
  input.placeholder = "不用追求标准术语，先用自己的话说明因果关系……";
  input.value = typeof answer.draft === "string" ? answer.draft : "";
  input.addEventListener("input", () => {
    answer.draft = input.value;
    if (own(answer, "selfScore")) delete answer.selfScore;
    persist();
    renderNavigation();
  });
  append(wrapper, label, input);
  if (!answer.revealed) {
    const row = element("div", "action-row");
    row.append(button(busy ? "正在加载…" : "写完了，对照评分要点", "button button-primary",
      () => reveal(question, answer), busy));
    wrapper.append(row);
  } else {
    wrapper.append(renderRubric(question, answer));
  }
  return wrapper;
}

function renderCodeResults(question, answer) {
  const result = answer.codeResult;
  const passed = Math.min(result.total, Math.max(0, result.passed));
  const panel = feedback(`公开测试 ${passed} / ${result.total} 通过`,
    `暂记 ${safeScore(result.score, question.max_score)} / ${question.max_score} 分。公开用例只能检查已知情况，不等于完整正确性。`,
    passed === result.total ? "success" : "warning");
  if (result.diagnostic) {
    panel.append(element("p", "syntax-diagnostic",
      `第 ${result.diagnostic.line} 行、第 ${result.diagnostic.column} 列：${result.diagnostic.message}。请检查缩进、冒号、括号或赋值符号。`));
  }
  const list = element("ul");
  for (const test of result.results || []) {
    const item = element("li", test.passed ? "test-pass" : "test-fail");
    if (test.passed) {
      item.append(element("span", "", `✓ ${test.name}：通过`));
    } else {
      const detail = element("details", "test-detail");
      append(detail, element("summary", "", `× ${test.name}：未通过`),
        element("pre", "", test.error || "测试未通过"));
      item.append(detail);
    }
    list.append(item);
  }
  panel.append(list);
  if (result.stdout) {
    const out = element("pre", "test-output", `运行输出\n${result.stdout.slice(0, 1800)}`);
    panel.append(out);
  }
  return panel;
}

function renderCode(question, answer) {
  const wrapper = element("div", "answer-zone");
  const label = element("label", "answer-label", "在这里完成函数（仅在浏览器运行）");
  label.htmlFor = `code-${question.id}`;
  const initial = typeof answer.draft === "string" ? answer.draft : question.starter_code;
  const toolbar = element("div", "code-toolbar");
  append(toolbar,
    element("span", "code-filename", `● ${question.entry_point}.py`),
    element("span", "code-runtime", "PYTHON · LOCAL WORKER"));
  const frame = element("div", "code-frame");
  const host = element("div", "code-editor-host");
  function onChange(value) {
    answer.draft = value;
    delete answer.codeResult;
    inlineError = "";
    persist();
    renderNavigation();
  }
  let fallback = null;
  if (window.RLCodeEditor) {
    activeEditor = window.RLCodeEditor.mount(host, initial, onChange,
      () => runCode(question, answer), `${question.title} Python 代码编辑区`, label.htmlFor);
    if (answer.codeResult?.diagnostic) activeEditor.setDiagnostic(answer.codeResult.diagnostic);
    frame.append(host);
  } else {
    const lines = element("div", "code-lines");
    lines.setAttribute("aria-hidden", "true");
    fallback = element("textarea", "answer-field code-editor");
    fallback.id = label.htmlFor;
    fallback.spellcheck = false;
    fallback.setAttribute("aria-label", `${question.title} Python 代码编辑区`);
    fallback.value = initial;
    function updateLines() {
      lines.textContent = Array.from({ length: fallback.value.split("\n").length },
        (_, index) => index + 1).join("\n");
    }
    updateLines();
    fallback.addEventListener("input", () => { onChange(fallback.value); updateLines(); });
    fallback.addEventListener("scroll", () => { lines.scrollTop = fallback.scrollTop; });
    append(frame, lines, fallback);
  }
  append(wrapper, label, toolbar, frame);
  const editorHint = element("p", "field-hint",
    window.RLCodeEditor
      ? "Tab 缩进，Shift+Tab 减缩进，⌘/Ctrl+Enter 运行。提示只补全通用 Python 词汇，不会写出题目答案。"
      : "Tab 可跳转到下一控件；点击“四空格”插入缩进。运行只使用公开测试。");
  const indentButton = button("四空格", "code-indent", () => {
    if (activeEditor) activeEditor.indent();
    else if (fallback) {
      fallback.setRangeText("    ", fallback.selectionStart, fallback.selectionEnd, "end");
      fallback.dispatchEvent(new Event("input"));
      fallback.focus();
    }
  });
  const editorHelp = element("div", "editor-help");
  append(editorHelp, editorHint, indentButton);
  wrapper.append(editorHelp);
  const testList = element("div", "test-list");
  append(testList, element("h3", "", `${question.tests.length} 个公开测试`));
  const list = element("ul");
  for (const test of question.tests) list.append(rich("li", "", `${test.name} · ${test.description}`));
  testList.append(list);
  wrapper.append(testList);
  if (!config.vendor_available) {
    wrapper.append(feedback("代码运行环境不可用",
      "本地 Pyodide 文件尚未就绪。你仍可保存代码与阅读参考实现；概念和手算题不受影响。", "warning"));
  }
  const actions = element("div", "action-row");
  actions.append(button(busy ? "正在运行…" : "运行公开测试", "button button-primary",
    () => runCode(question, answer), busy || !config.vendor_available));
  if (!answer.revealed) {
    actions.append(button("看参考实现", "button button-quiet", () => reveal(question, answer), busy));
  }
  wrapper.append(actions);
  if (answer.codeResult) wrapper.append(renderCodeResults(question, answer));
  if (answer.revealed) wrapper.append(renderRubric(question, answer));
  return wrapper;
}

function evaluateInWorker(code, tests) {
  return new Promise((resolve, reject) => {
    const worker = new Worker("./worker.js");
    let finished = false;
    let timer = setTimeout(() => fail("Python 环境加载超时，请稍后重试。"), 60000);
    function cleanup() {
      if (finished) return false;
      finished = true;
      clearTimeout(timer);
      worker.terminate();
      return true;
    }
    function fail(message) {
      if (cleanup()) reject(new Error(message));
    }
    worker.onerror = (event) => fail(event.message || "代码运行环境出错。");
    worker.onmessage = ({ data: message }) => {
      if (message.type === "ready") {
        clearTimeout(timer);
        timer = setTimeout(() => fail("运行超过 10 秒，已终止这次代码测试。"), 10000);
        worker.postMessage({ type: "run", code, tests });
      } else if (message.type === "result") {
        if (cleanup()) resolve(message.result);
      } else if (message.type === "error") {
        fail(message.error || "Python 运行失败。");
      }
    };
  });
}

async function runCode(question, answer) {
  if (busy || !config.vendor_available) return;
  busy = true;
  inlineError = "";
  renderQuestion(question);
  try {
    const result = await evaluateInWorker(
      String(answer.draft ?? question.starter_code),
      question.tests.map(({ name, code }) => ({ name, code })));
    const total = question.tests.length;
    const passed = result.results.filter((test) => test.passed).length;
    answer.codeResult = {
      passed,
      total,
      score: Math.round((passed / total) * question.max_score * 10) / 10,
      results: result.results,
      stdout: result.stdout || "",
      diagnostic: result.diagnostic || null,
    };
    persist();
  } catch (error) {
    inlineError = error.message;
  } finally {
    busy = false;
    render();
  }
}

function showSummary() {
  view = "summary";
  inlineError = "";
  render();
  dom.summary.scrollIntoView({ block: "start" });
}

function topicFor(question) {
  if (["q03", "q10"].includes(question.id)) return "回看 EasyRL §2.1.1：马尔可夫性质与状态信息。";
  if (["q01", "q02", "q04", "q05", "q06", "q11"].includes(question.id)) {
    return "回看 §2.2.1：折扣回报与状态价值。";
  }
  return "回看 §2.2.2：即时奖励、转移概率与贝尔曼方程。";
}

function exportProgress() {
  persist();
  const file = new Blob([JSON.stringify({
    version: 2, selectedChapterId: currentChapterId, chapters: chapterStates,
  }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = "rl-learning-progress.json";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

async function importProgress(file) {
  if (!file) return;
  try {
    if (file.size > 1024 * 1024) throw new Error("进度文件不能超过 1 MB。");
    const saved = JSON.parse(await file.text());
    if (saved?.version !== 2 || !saved.chapters ||
        typeof saved.chapters !== "object" || Array.isArray(saved.chapters) ||
        !Object.values(saved.chapters).every(validChapterState)) {
      throw new Error("这不是本学习站导出的有效进度文件。");
    }
    if (!window.confirm("导入将覆盖此浏览器中的现有学习进度。确定继续吗？")) return;
    const previous = { currentChapterId, state, chapterStates, data };
    const nextId = chapterCatalog.chapters.some((item) => item.id === saved.selectedChapterId)
      ? saved.selectedChapterId : chapterCatalog.default_chapter_id;
    try {
      const loaded = await api(`/api/questions?chapter=${encodeURIComponent(nextId)}`);
      chapterStates = saved.chapters;
      currentChapterId = nextId;
      data = loaded;
      state = validChapterState(chapterStates[nextId]) ||
        { current: loaded.questions[0].id, responses: {} };
      if (!loaded.questions.some((question) => question.id === state.current)) {
        state.current = loaded.questions[0].id;
      }
      persist();
      updateChapterHeading();
      view = "question";
      inlineError = "";
      render();
    } catch (error) {
      ({ currentChapterId, state, chapterStates, data } = previous);
      throw error;
    }
  } catch (error) {
    window.alert(error.message || "导入失败，请检查文件。");
  }
}

function renderSummary() {
  const questions = data.questions;
  const total = questions.reduce((sum, question) => sum + scored(question), 0);
  const done = questions.filter(isComplete).length;
  const buckets = [
    ["客观题", "auto", data.scoring.objective],
    ["开放题自评", "self", data.scoring.open_self_review],
    ["代码公开测试", "browser", data.scoring.code_browser_tests],
  ];
  const root = document.createDocumentFragment();
  const header = element("div", "summary-header");
  append(header, element("h2", "", "这一节，你掌握了多少？"),
    element("div", "summary-total", `${Number(total.toFixed(1))} `));
  header.lastChild.append(element("small", "", `/ ${data.total_score} 分`));
  root.append(header);
  root.append(element("p", "summary-copy",
    `已完成 ${done} / ${questions.length} 题。分数混合了客观判分、你自己的开放题评分与代码公开测试，请分开理解；尚未作答的题目暂记 0 分。`));
  const breakdown = element("div", "summary-breakdown");
  for (const [label, mode, maximum] of buckets) {
    const card = element("div");
    append(card, element("span", "", label),
      element("strong", "", `${Number(questions.filter((q) => q.grading_mode === mode)
        .reduce((sum, q) => sum + scored(q), 0).toFixed(1))} / ${maximum}`));
    breakdown.append(card);
  }
  root.append(breakdown);
  root.append(element("h3", "review-heading", "下一步，回到哪里？"));
  const review = questions.filter((question) => !isComplete(question) || scored(question) < question.max_score);
  const list = element("ul", "review-list");
  if (review.length) {
    for (const question of review) {
      const item = element("li");
      const link = button(`${question.id.slice(1)} · ${question.title}`, "", () => selectQuestion(question.id));
      link.append(element("span", "", !isComplete(question) ? "未作答" : topicFor(question)));
      item.append(link);
      list.append(item);
    }
  } else {
    const item = element("li", "", "当前题目已全部完成。下一节仍需亲手验证概念，而非只看分数。");
    list.append(item);
  }
  root.append(list);
  const actions = element("div", "action-row");
  actions.append(button("返回当前题", "button button-primary", () => selectQuestion(state.current)));
  actions.append(button("导出学习进度", "button button-quiet", exportProgress));
  const importInput = element("input", "progress-file-input");
  importInput.type = "file";
  importInput.accept = ".json,application/json";
  importInput.setAttribute("aria-label", "选择学习进度 JSON 文件");
  importInput.addEventListener("change", () => {
    importProgress(importInput.files?.[0]);
    importInput.value = "";
  });
  actions.append(button("导入学习进度", "button button-quiet", () => importInput.click()));
  actions.append(importInput);
  root.append(actions);
  root.append(element("p", "field-hint",
    "进度只保存在当前浏览器；从本地地址换到公开网址时，先在旧站导出，再到新站导入。"));
  dom.summary.replaceChildren(root);
}

$("#retry-button").addEventListener("click", initialize);
$("#summary-button").addEventListener("click", showSummary);
initialize();
