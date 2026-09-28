# 编写与发布新的学习检查点

当前目录是一套可复用的「概念 → 手算 → 解释 → Python 公开测试」练习站。
第一个章节的题库在 `bank.py`；以后每章单独建一个 Python 模块，不要往原章节
硬塞 PPO、DPO 等未读内容。

## 1. 先核对来源，再设计题

记录教材版本、节号、印刷页与 PDF 页的差别。若结合真实代码，记录仓库提交、
文件位置、输入输出、分支配置和可复核样例；模拟数据必须标为教学构造。
每道题写明学习目标、评分规则和典型错误。代码题给出签名、starter、
公开测试及边界情况；自动测试只能证明公开用例通过。

## 2. 新增一个独立题库模块

仿照 `bank.py`，例如新建 `chapter_mdp.py`，提供以下接口：

```python
REFERENCE = "具体教材版本和章节"

def list_questions():
    return QUESTIONS  # 内部题目，含 _answer / _expected / _solution

def public_payload():
    return {
        "reference": REFERENCE,
        "conventions": "...",
        "industry_case": None,
        "total_score": sum(q["max_score"] for q in QUESTIONS),
        "count": len(QUESTIONS),
        "scoring": {
            "objective": ...,
            "open_self_review": ...,
            "code_browser_tests": ...,
        },
        "questions": [public_question(q) for q in QUESTIONS],
    }

def grade(question_id, answer):
    """只评选择和数值题。未知题/输入不合法时使用 bank.py 的异常类型。"""

def reveal(question_id):
    """仅开放题和代码题返回 rubric、model_answer/reference_solution。"""
```

可直接复制 `bank.py` 的题型结构和公开字段白名单实现，不要在
`public_payload()` 中输出 `_answer`、`_expected`、`_solution`。本地 API
对错误类型使用 `bank.UnknownQuestion`、`bank.InvalidAnswer`、
`bank.NotAutoGradable` 和 `bank.NotRevealable`；新模块应沿用这些异常。
题号 `q01` 等可在不同章节复用，存储与判分会按章节隔离。

在 `chapters.py` 中导入模块，并添加一行：

```python
register_bank("mdp-basics", "MDP：策略与价值", chapter_mdp)
```

真实章节若有行业案例，先查明运行路径与条件再增加内容，不要把当前
`verl` 示例当作所有章节通用案例。首页以注册顺序显示章节。

## 3. 验证与发布

```bash
python3 -m unittest -q test_server.py test_site_build.py
python3 build_site.py --output /tmp/rl-learning-preview/checkpoint
```

再以本地静态服务器打开导出目录，逐项检查选择题、计算题、开放题、
代码编译错误、公开测试、刷新后进度与窄屏。导出的 `site-data.js` 包含
**自学用途**的答案和参考实现；任何人都能查看，不能用于防作弊考试。
浏览器中的 Pyodide 运行学员代码；Python HTTP 服务从不执行学员代码。

GitHub Pages 部署从 `main` 分支的 `/docs` 目录发布；在仓库根目录执行：

```bash
python3 checkpoint/publish_pages.py
git add checkpoint docs
git commit -m "Update learning checkpoint"
git push origin main
```

发布脚本会运行单测、检查 JS、执行 `npm ci` 和编辑器构建，然后重新生成
`docs/checkpoint/`。对外路径是 `/rl-learning/checkpoint/`。静态站不运行
`server.py`；客户端加载生成的数据，因此要核对本地和导出站的评分行为一致。
不要把教材 PDF、`verl` 工作树、量化数据或凭据放进发布目录。公开仓库不包含
外部 `verl` 源码时，针对它的**源码交叉核对测试**会明确跳过，其余题库测试仍需通过。

本机 `127.0.0.1` 和正式域名是不同的浏览器存储来源。换域名之前先在旧站
导出学习进度，再到新站导入；切勿重置旧浏览器的 `localStorage`。
