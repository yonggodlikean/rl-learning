# 实战案例 2：verl 框架大模型推理链路的规则打分与 GAE 优势计算

对应框架：**字节跳动开源大模型 RL 框架 `verl`（commit a2ad9f6）**  
对应教材章节：**《EasyRL》§2.2（回报与 MRP）与 §2.3（状态价值评估与贝尔曼自举）**

---

## 业务背景与问题定义

在像 GSM8K 这类大模型数学推理任务中，模型生成一个包含多步推理的 Token 序列。
我们面临现代大模型强化学习（PPO / RLHF）的核心挑战：
1. **即时奖励极度稀疏**：只有在最后一步得出正确答案时，环境才给予 $+1.0$ 的奖励，中间每一步没有即时反馈；
2. **如何评估每个 Token 的贡献（Credit Assignment）**：
   - 使用 Critic 模型预测每个 Token 状态的基准内在价值 $V(s_t)$；
   - 利用广义优势估计（GAE, Generalized Advantage Estimation）从后向前倒序计算时序差分残差 $\delta_t$ 与累积优势 $A_t$，指导策略网络（Actor）更新。

---

## 文件结构与数据流

```text
verl_token_gae_pipeline/
├── README.md        # 业务全景图与 verl 源码映射解析（当前文件）
├── schema.py        # 极简模拟 verl 的 DataProto 数据协议 (tokens, masks, critic_values, ground_truth)
├── core_solver.py   # 【核心考卷】待你手写的：规则奖励路由与 GAE 倒序递推算子
└── run.py           # 一键运行驱动器：包含严格 Padding 保护断言与漂亮的 Token 价值流转报表
```

### 数据流拓扑图：
```
 +--------------------+
 |     schema.py      | (加载推理轨迹样本与 Token 掩码)
 +---------+----------+
           |
           v
 +--------------------+
 |   core_solver.py   | <--- 【待你手写实现】
 |  - extract_rewards |      1. 从文本末尾提取 #### 答案并打标稀疏奖励
 |  - compute_gae()   |      2. 倒序递推 TD 残差与 GAE 优势值 (支持 Mask 保护)
 +---------+----------+
           |
           v
 +--------------------+
 |       run.py       | (打印每个 Token 的 Critic V、TD δ、GAE Adv 与 Return G)
 +--------------------+
```

---

## 真实源码对应清单（对应你桌面上的 `verl` 源码）

| 待手写的算子 | 对应 `verl` 源码位置 | 真实行号 | 核心作用 |
| :--- | :--- | :--- | :--- |
| `extract_and_assign_rewards` | `verl/utils/reward_score/gsm8k.py`<br>`verl/experimental/agent_loop/agent_loop.py` | 20–36<br>997–1027 | 规则提取答案比对，将标量赋给末位有效 token |
| `compute_gae_and_returns` | `verl/trainer/ppo/core_algos.py` | 216–262 | 倒序计算时序差分残差与 GAE：$\delta_t = r_t + \gamma V_{t+1} - V_t$ |

---

## 你的任务

请打开 `core_solver.py`，根据详细注释补齐两大核心算子：
1. `extract_and_assign_rewards(traj)`：规则打分并赋给最后一个有效 token；
2. `compute_gae_and_returns(token_rewards, critic_values, response_mask, gamma=0.5, lam=0.95)`：倒序计算 GAE。

完成后，在终端运行：
```bash
python rlhf/scenarios/verl_token_gae_pipeline/run.py
```
你将看到控制台直接输出对齐工业级框架的 Token 级价值报表！
