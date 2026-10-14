# RL 学习中枢 (RL Learning Hub)

强化学习学习中枢（RL Learning Hub）是依托 `rl-learning` 仓库构建的本地与静态学习导航系统。它通过构建期生成的结构化数据快照（`catalog.json`），实现理论章节梳理、随堂题目检验、浏览器代码演练、真实实验索引与本地学习反思的闭环。

---

## 核心设计原则

1. **目录随仓库自然生长 (Repository-Driven)**  
   课程章节完全由题库与章节注册表（`checkpoint/chapters.py`）驱动；实验案例完全由仓库实际物理目录与清单文件（`lab.json` / `README.md`）驱动。**仓库新增章节或实验无需修改任何前端代码**。

2. **仓库收录 ≠ 个人掌握 (Availability ≠ Progress)**  
   仓库目录或实验代码的增加仅代表学习资源的丰富，不代表学习者的掌握度。系统不提供虚假的“全书完成百分比”或强制关卡解锁，而以真实的**最近访问（Last Visited）**记录作为“继续学习”的锚点。

3. **构建期生成，严禁运行时轮询 (Build-Time Generation)**  
   所有元数据在构建期由编译脚本扫描生成并固化为 `catalog.json`（Schema v1）。前端完全基于该静态快照运行，**杜绝在前端运行时向 GitHub API 进行高频或实时轮询**。

4. **源码指引 ≠ 线上验证 (Source Guidance ≠ Verified Runtime)**  
   实验案例的入口文件与执行命令仅作为本地运行的元数据参考，中枢界面如实标注“运行未经在线验证 (Runtime NOT verified)”，严禁在前端伪造运行通过状态或在线执行任意系统命令。

5. **本地反思私有持久化**  
   章末与案例的反思笔记严格保存在浏览器本地（`localStorage` 的 `rl_hub_state_v1`），不上传远端，且严格与 Checkpoint 判分存储隔离。

---

## 本地开发与生成命令

### 1. 生成数据目录 (Build Catalog)
在仓库根目录下运行目录编译器，扫描当前注册的章节与领域目录中的实验：

```bash
python3 hub/build_catalog.py
```

该命令将在 `hub/catalog.json` 生成符合 Schema v1 的最新数据快照。

### 2. 本地启动服务测试 (Local HTTP Server)
在仓库根目录依次执行目录编译与本地安全回环服务启动：

```bash
# 1. 扫描生成数据目录
python3 hub/build_catalog.py

# 2. 启动安全回环 HTTP 服务器（仅限本地 127.0.0.1，仅开放 hub/ 目录）
python3 -m http.server 8899 --bind 127.0.0.1 --directory hub
```

启动后在浏览器中访问：  
[http://127.0.0.1:8899/](http://127.0.0.1:8899/)

### 3. 构建与发布说明 (Build & Publish)
静态页面发布由 `checkpoint/publish_pages.py` 统筹管理：

```bash
python3 checkpoint/publish_pages.py
```

- 该脚本会将 Checkpoint 判分页面打包到 `docs/checkpoint/`，同时将 Hub 前端（`index.html`、`styles.css`、`app.js` 及生成的 `catalog.json`）打包到 `docs/hub/`。
- `docs/` 根目录将重定向至 `./hub/`。
- **当前任务为本地实现与测试阶段，不执行远端 GitHub 部署或推送。**

---

## 扩展指引：如何新增章节与实验

### 1. 注册新章节示例
在 `checkpoint/chapters.py` 中注册新章节并定义其关联题库：

```python
# checkpoint/chapters.py 示例
from checkpoint.bank import QuestionBank

new_bank = QuestionBank("easyrl-3.0", "无模型预测与控制 (Model-Free Prediction & Control)")
# 添加选择题、手算题、简答题或代码题...
new_bank.add_choice(
    "q01",
    "时序差分 (TD) 学习与蒙特卡洛 (MC) 方法的核心区别是什么？",
    category="概念辨析",
    options=[...],
    answer="...",
    max_score=5
)

register_chapter(
    chapter_id="easyrl-3.0",
    title="无模型预测与控制",
    reference="EasyRL §3.0–§3.3",
    bank=new_bank,
    order=3
)
```

**更新流程**：保存后执行 `python3 hub/build_catalog.py`。刷新前端页面，新章节将自动出现在全景总览、侧边栏导航、检索索引中，各题型自动归类，无需修改任何前端 JavaScript 文件。

### 2. 新增实验案例清单 (`lab.json`) 示例
在 `hub/catalog_config.json` 声明的 7 大实验领域之一（例如 `classic_rl/`）中创建实验子目录，并添加 `lab.json` 清单文件：

```json
{
  "title": "网格世界策略迭代与价值迭代对比",
  "description": "基于经典 4x4 GridWorld 环境，对比同步策略迭代与贝尔曼最优价值迭代的收敛步数与值函数曲面。",
  "chapter_ids": [
    "easyrl-2.3"
  ],
  "prerequisites": [
    "easyrl-2.1-2.2.2"
  ],
  "entrypoint": "run_comparison.py",
  "command": "python3 run_comparison.py --gamma 0.95 --theta 1e-4"
}
```

- `chapter_ids`: 关联的已注册章节 ID（前端章节页将自动关联展示）。若关联未知章节，构建脚本将输出警告并在关系导航中安全剔除。
- `prerequisites`: 前置掌握章节 ID。
- `entrypoint`: 相对于实验目录的主执行脚本文件。
- `command`: 推荐的本地命令行启动参数（仅作文本指引展示）。

*(若实验目录仅包含 `README.md` 与源码文件而无 `lab.json`，构建脚本亦会自动发现并归类为该领域下的待关联案例。)*

---

## 领域配置 (`hub/catalog_config.json`)

领域根目录通过 `hub/catalog_config.json` 维护，默认收录 7 大方向：
- `classic_rl`: 经典强化学习算法（动态规划、表格型 TD、Q-Learning、DQN 等）
- `recommender`: 推荐系统中的强化学习应用（多臂老虎机、Slate RL、序列决策）
- `rlhf`: 基于人类反馈的强化学习（奖励模型训练、PPO、DPO 对齐）
- `quant_rl`: 量化金融与高频交易决策强化学习
- `post_training`: 大模型 Post-Training 阶段的大规模 RL
- `papers`: 经典与前沿论文复现实验
- `reports`: 实验分析报告与消融对比实验

目录编译器会递归遍历各领域目录，但会自动跳过 `.git`、隐藏目录、虚拟环境（`venv`、`.venv`）、数据缓存与模型权重等敏感/巨型目录。

---

## 本地状态字典 (`rl_hub_state_v1`)

中枢在浏览器本地存储中维护 `rl_hub_state_v1` 键：
- `reflections`: 键值映射表，支持 `chapter:<id>`、`case:<id>` 及历史遗留的 `topic:<id>` 键。
- `reflectionTimestamps`: 反思内容最后一次保存的时间戳。
- `lastVisited`: `{ route: string, title: string, timestamp: number }`，真实反映学习者的最近访问轨迹。
- 所有非本系统新增的自定义字段均在读写时予以完整保留，绝不主动清空或跨越边界访问 Checkpoint 的答题进度。
