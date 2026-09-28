# RL Learning

A hands-on learning workspace covering classical reinforcement learning, recommender systems, RLHF, quantitative-finance RL, and small-model post-training.

## 在线练习站

访问 [RL Learning Lab](https://yonggodlikean.github.io/rl-learning/checkpoint/)：现有 EasyRL §2.1–§2.2.2 和 [§2.3 MDP：策略评估与控制](https://yonggodlikean.github.io/rl-learning/checkpoint/?chapter=easyrl-2.3) 两章，各 12 道概念、手算、解释和 Python 代码题。§2.3 含两段同步贝尔曼备份代码题、合成推荐曝光矩阵演算，以及明确注明条件分支的 `verl` GAE 真实源码对照；合成数据不是真实用户日志或模型训练结果。代码用浏览器内 Pyodide Worker 执行公开测试，编辑器支持 Python 高亮、行号、缩进和基础补全。当前章节只用 Python 标准库；不需要安装 NumPy，也不依赖本机 `127.0.0.1` 服务。它是自学站，不是保密考试：静态发布数据包含答案。

浏览器进度不会在设备或域名之间自动同步。从旧版本地站迁移：在旧站打开「学习复盘」→「导出学习进度」，再到公开站「学习复盘」→「导入学习进度」。不要清除旧站浏览器存储。

题库、独立章节注册方法、来源核验和本地开发步骤见 [`checkpoint/AUTHORING.md`](checkpoint/AUTHORING.md)。新增章节可注册题库；尚未学过的章节不会自动生成题目。运行 `python3 checkpoint/publish_pages.py` 后将 `checkpoint/` 和生成的 `docs/` 一起提交推送到 `main`，GitHub Pages 从 `main /docs` 自动发布。发布脚本会先运行测试并构建编辑器，无须在 GitHub CLI 增加 `workflow` 权限。

## Structure

- `classic_rl/` — bandits, MDPs, TD learning, policy gradients, and PPO
- `recommender/` — BPR, sequential recommendation, and preference optimization
- `rlhf/` — reward modeling, PPO-based RLHF, DPO, and GRPO
- `quant_rl/` — reinforcement-learning experiments using quantitative-finance data
- `post_training/` — SFT and preference/post-training experiments for small models
- `papers/` — paper notes and reproductions
- `reports/` — experiment reports, plots, and learning summaries
- `checkpoint/` — 可持续新增章节的浏览器练习站与题库

## Learning plan

The initial plan is a 30–45 day curriculum with classical RL and RLHF as the main track, plus quantitative RL and small-model post-training as applied projects.
