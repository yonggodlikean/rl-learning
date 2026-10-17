"""Value function methods: balanced theoretical intuition and practical RL engineering.

Grounded in:
- Zhao Shiyu《Mathematical Foundations of Reinforcement Learning》Chapter 8 (pp. 151-190)
- EasyRL v1.0.6 Chapter 6 (pp. 93-105) & Chapter 7 (pp. 106-119)
- Local CartPole-v1 DQN experimental code & logs in experiments/cartpole-dqn/
"""

import math
import numpy as np

import bank


CHAPTER_ID = "value-function-methods"
REFERENCE = (
    "Zhao《Mathematical Foundations of Reinforcement Learning》第8章"
    "（印刷页151–190 / PDF页164–203）；"
    "EasyRL v1.0.6 第6章与第7章"
    "（印刷页93–119 / PDF页101–127）；"
    "工作区实验代码：experiments/cartpole-dqn/"
)
CONVENTIONS = (
    "题目紧密贴合 CartPole 与通用深度强化学习工程实战，平衡理论本质与代码落地。"
    "在 DQN 中，经验元组采用 (s, a, r, s', done)，当 done=True 时未来价值必须截断为 0。"
    "半梯度对应 PyTorch 中使用 torch.no_grad() 或 detach() 切断目标项的计算图梯度传播。"
    "Double DQN 采用当前网络选择贪心动作索引，目标网络评估该动作分值以消除高估偏差。"
    "手算题所有参数与转移数据均已自包含，无需翻找前后文。"
)

PRESENTATION = {
    "eyebrow": "VALUE FUNCTION METHODS / 理论与实战平衡特辑",
    "intro": (
        "摒弃脱离实际的纯矩阵推导，紧扣 DQN 训练精髓与 CartPole 工程实践："
        "从半梯度在 PyTorch 计算图中的实现、终止状态截断的致命 Bug、"
        "目标网络与经验池超参数权衡，到 Double DQN 与 Dueling 架构的代码落地，"
        "真正搞懂强化学习中的移动靶、分布偏移与稳定收敛机理。"
    ),
    "field_notes": [
        {"label": "01 / NO-GRAD TARGET",
         "formula": "y = r + (1 - done) * gamma * max_a' Q(s', a'; theta^-)",
         "tex": r"y_t = r_{t+1} + (1 - \text{done}) \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)",
         "description": "目标值必须用 torch.no_grad() 阻断梯度回传，且 done=True 时必须截断未来。"},
        {"label": "02 / DOUBLE DECOUPLING",
         "formula": "a* = argmax Q(s', a; theta), y = r + gamma * Q(s', a*; theta^-)",
         "tex": r"a^* = \arg\max_a Q(s', a; \theta),\quad y = r + \gamma Q(s', a^*; \theta^-)",
         "description": "选动作与评价值彻底解耦，两行代码消除 max 算子的正向高估偏差。"},
        {"label": "03 / DUELING IDENTIFIABILITY",
         "formula": "Q(s, a) = V(s) + [A(s, a) - mean_a'(A(s, a'))]",
         "tex": r"Q(s, a) = V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\Big)",
         "description": "优势流均值中心化保证唯一可辨识性，不增加采样开销即可加速价值学习。"},
    ],
    "note_footer": (
        "理论指导调参，代码印证理论：深入理解强化学习与常规监督学习在数据分布与梯度目标上的根本差异。"
    ),
}


def _source(zhao, printed, pdf, easy, easy_printed, easy_pdf, note=""):
    return {"text": (
        f"Zhao §{zhao}，印刷页 {printed} / PDF 页 {pdf}；"
        f"EasyRL v1.0.6 §{easy}，印刷页 {easy_printed} / PDF 页 {easy_pdf}。"
        + note
    )}


VREP_SOURCE = _source("8.1", "152–155", "165–168",
                      "6.1", "93–96", "101–104",
                      "函数近似下的状态泛化与特征干扰；对应 CartPole 状态表示。")
SEMIGRAD_SOURCE = _source("8.2.1–8.2.2", "156–162", "169–175",
                          "6.2", "96–99", "104–107",
                          "半梯度在代码中的体现：为什么必须用 torch.no_grad() 切断目标梯度。")
DONE_SOURCE = _source("8.4.1", "181–184", "194–197",
                      "6.2", "96–99", "104–107",
                      "终止状态截断 (1 - done) 的物理含义与漏写灾难。")
TARGET_NET_SOURCE = _source("8.4.1", "182–184", "195–197",
                            "6.2–6.3", "96–101", "104–109",
                            "目标网络同步周期与非稳态移动靶的工程权衡；结合 train_dqn_500epochs.py。")
REPLAY_SOURCE = _source("8.4.1", "182–184", "195–197",
                        "6.2–6.3", "96–101", "104–109",
                        "经验回放池容量、Warm-up 策略与时序相关性打破。")
DDQN_SOURCE = _source("8.4.1", "182–184", "195–197",
                      "7.1", "106–109", "114–117",
                      "Double DQN 解耦机制与过估计在训练曲线中的诊断。")
DUELING_SOURCE = _source("8.4.1", "182–184", "195–197",
                        "7.2", "109–112", "117–120",
                        "Dueling DQN 架构与均值中心化前向传播。")
DEADLY_SOURCE = _source("8.2.5、8.6", "176–178、187–190", "189–191、200–203",
                       "6.2–6.3", "97–101", "105–109",
                       "死亡铁三角发散机理与现代工程防线。")


def _q(qid, qtype, category, title, prompt, score, goal, source, **extra):
    if qtype == "numeric" and extra.get("answer_type") == "list":
        extra["answer_count"] = len(extra["_expected"])
    if qtype == "code":
        if "language" not in extra:
            extra["language"] = "python"
        if "public_tests" in extra and "tests" not in extra:
            extra["tests"] = extra.pop("public_tests")
    return {
        "id": qid, "type": qtype, "category": category,
        "title": title, "prompt": prompt, "max_score": score,
        "grading_mode": {"choice": "auto", "numeric": "auto",
                         "open": "self", "code": "browser"}[qtype],
        "learning_goal": goal, "source_note": source,
        "review_hint": source["text"], **extra,
    }


# =========================================================================== #
# Reference solutions for code questions
# =========================================================================== #

_SELECT_ACTION_SOLUTION = """def select_action(q_values, epsilon, num_actions, random_val=None):
    import numpy as np
    import random
    r = random.random() if random_val is None else random_val
    if r < epsilon:
        return 0 if random_val is not None else random.randrange(num_actions)
    return int(np.argmax(q_values))
"""

_COMPUTE_DQN_TARGETS_SOLUTION = """def compute_dqn_targets(rewards, next_q_max, dones, gamma):
    import numpy as np
    r = np.array(rewards, dtype=float).reshape(-1, 1)
    q_max = np.array(next_q_max, dtype=float).reshape(-1, 1)
    d = np.array(dones, dtype=float).reshape(-1, 1)
    return r + (1.0 - d) * gamma * q_max
"""

_COMPUTE_DDQN_TARGETS_SOLUTION = """def compute_ddqn_targets(rewards, q_eval_next, q_target_next, dones, gamma):
    import numpy as np
    r = np.array(rewards, dtype=float).reshape(-1, 1)
    d = np.array(dones, dtype=float).reshape(-1, 1)
    q_eval = np.array(q_eval_next, dtype=float)
    q_target = np.array(q_target_next, dtype=float)
    best_actions = np.argmax(q_eval, axis=1)
    selected_q = q_target[np.arange(len(q_eval)), best_actions].reshape(-1, 1)
    return r + (1.0 - d) * gamma * selected_q
"""

_DUELING_FORWARD_SOLUTION = """def dueling_forward(v_s, a_sa):
    import numpy as np
    v = np.array(v_s, dtype=float)
    a = np.array(a_sa, dtype=float)
    a_mean = np.mean(a, axis=-1, keepdims=True)
    return v + (a - a_mean)
"""

_SOFT_UPDATE_SOLUTION = """def soft_update(target_weights, policy_weights, tau):
    import numpy as np
    t = np.array(target_weights, dtype=float)
    s = np.array(policy_weights, dtype=float)
    return tau * s + (1.0 - tau) * t
"""


# =========================================================================== #
# Questions definitions (21 questions total, 108 pts)
# =========================================================================== #

QUESTIONS = [
    # ----------------------------------------------------------------------- #
    # Part 1: Concept choices (q01 - q06, 4 pts each = 24 pts)
    # ----------------------------------------------------------------------- #
    _q("q01", "choice", "concept", "从表格到神经网络：泛化性与跨状态干扰的权衡",
       r"在 CartPole 实验中，小车在倾斜角度 \(\theta = 0.05\) 时采取推力获得良好回报，网络反向传播更新了所有共享权重。"
       r"这种更新不可避免地连带改变了小车在严重倾斜（如 \(\theta = 0.20\)）状态下的 Q 值估值。"
       r"下列关于这种‘函数逼近’现象的评价，哪项最为客观准确？",
       4, "透彻理解神经网络参数共享带来的泛化优势与表示干扰代价。", VREP_SOURCE,
       options=[
           "这是一种严重缺陷，说明神经网络无法应用于强化学习，必须改回表格型 Q 表格。",
           "这是函数逼近的核心优势（使得未见状态能获得合理初值），但若采样分布严重偏倚，也会导致先前学到的高价值状态估值被冲刷（灾难性遗忘与表示干扰）。",
           "这说明全连接网络层数太少，只要增加到 100 层以上，各状态之间的参数更新就会完全相互独立。",
           r"这说明学习率 \(\alpha\) 设置过小，只要将学习率调大到 10 以上即可实现状态隔离。"],
       _answer=1, feedback_hint="参数共享带来泛化，但也会导致频繁采样的经验冲刷未频繁访问状态的记忆。"),

    _q("q02", "choice", "concept", "PyTorch 计算图实操：为什么 Target Q 必须用 torch.no_grad()？",
       r"在 `train_dqn_500epochs.py` 中，计算目标 Q 值时编写了如下代码："
       r"\n```python\nwith torch.no_grad():\n    max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)\n    target_q = rewards + (1 - dones) * self.gamma * max_next_q\n```"
       r"\n如果移除了 `with torch.no_grad():`（且未调用 `.detach()`），在训练中会发生什么？",
       4, "从深度学习计算图与半梯度理论结合的角度理解 no_grad 的必要性。", SEMIGRAD_SOURCE,
       options=[
           "代码会因为变量类型不匹配而在第一步立即崩溃，抛出 TypeError。",
           "反向传播会将梯度同时回传给目标项，算法在数学上退化为残差梯度法（引入双采样偏差），且历史计算图常驻显存引发内存/显存泄露。",
           "训练完全不受影响，因为 PyTorch 的优化器会自动识别并忽略目标网络的参数。",
           "网络会强制进行二阶牛顿法迭代，使得训练速度提高 10 倍。"],
       _answer=1, feedback_hint="不阻断目标计算图会导致梯度反传到 target_net 甚至历史图，破坏半梯度更新的单向性。"),

    _q("q03", "choice", "concept", "致命代码 Bug 诊断：为什么终止状态必须有 (1 - dones) 掩码？",
       r"在 DQN 目标计算公式 \(y = r + (1 - \text{done}) \gamma \max_{a'} Q(s', a')\) 中，`done` 标志用于标识回合终止（如 CartPole 杆子倒地或越界）。"
       r"如果某位工程师疏忽写成了 `target_q = rewards + self.gamma * max_next_q`（漏掉了 `1 - dones`），会导致什么灾难性后果？",
       4, "深刻理解马尔可夫决策过程中吸收终止状态的价值定义与代码防错。", DONE_SOURCE,
       options=[
           "小车会永远处于静止状态，动作全被强制置为 0。",
           "小车倒地失败后，算法依然将假想的未来步数折现加到当前回报中，导致失败动作的 Q 值严重虚高，智能体彻底失去对倒地惩罚的敏感性而无法收敛。",
           "没有任何影响，因为倒地后环境会重置，下一状态的 Q 值必然为 0。",
           "只影响代码运行速度，不会影响策略最终学会平衡。"],
       _answer=1, feedback_hint="终止态的未来真实价值严格为 0。漏乘 (1-done) 会向倒地终止转移错误注入虚假未来期望。"),

    _q("q04", "choice", "concept", "目标网络（Target Network）同步周期的工程权衡",
       r"在 CartPole 脚本中，目标网络每隔 `target_sync_freq = 10` 个 episode 将策略网络的权重硬拷贝一次（`sync_target`）。"
       r"关于同步周期大小的设定，下列哪项分析是正确的？",
       4, "掌握目标网络在缓解移动靶震荡与保证学习时效之间的折中调参法则。", TARGET_NET_SOURCE,
       options=[
           "同步周期越短越好，每步同步（freq=1）效果最佳，因为此时目标网络最懂最新策略。",
           "周期太短（如步步同步）会导致目标成为剧烈移动的靶子，丧失稳定性；周期太长（如数千 episode）则导致目标信息极度滞后，训练迟缓不前。",
           "目标网络必须在训练前由随机种子初始化后永久冻结，绝不允许进行任何形式的更新。",
           "同步周期必须严格等于经验回放池的容量，否则数学上不满足 Markov 性。"],
       _answer=1, feedback_hint="目标网络的核心价值在于在一个短窗口内提供固定的回归靶心，既不能频繁震荡也不能过度陈旧。"),

    _q("q05", "choice", "concept", "经验回放池（Replay Buffer）的 Warm-up 与容量权衡",
       r"在 DQN 实现中，主循环开头通常包含：`if len(buffer) < batch_size: return None`（或预热收集数千步）。"
       r"此外经验池设置了固定容量（如 `capacity=10000`）。关于其工程设计，下列说法正确的是？",
       4, "透彻理解经验回放池消除时序相关性、支持均匀离线采样与容量设定的工程逻辑。", REPLAY_SOURCE,
       options=[
           "经验池容量设为 64（等于 batch_size）最理想，因为这样可以保证每次只学最新鲜的一批数据。",
           "经验池过小会导致数据退化为高度自相关的前后连续帧，网络迅速过拟合失稳；但容量过大且不加优先级（如存入 100 万步早期随机数据），会导致高价值稀有样本被大量低效数据稀释。",
           "Warm-up 只是为了等待 GPU 温度上升，对算法数学更新没有任何影响。",
           "经验池必须按时间严格从旧到新顺序抽取，严禁随机打乱。"],
       _answer=1, feedback_hint="经验回放的核心是打破时序相关性并提升样本利用率，容量需要与探索进程相适应。"),

    _q("q06", "choice", "concept", "Double DQN 在训练监控中的表征与实战诊断",
       r"在训练标准 DQN 时，如果在 Tensorboard 中监控各状态动作的最大预测 Q 值曲线，常发现预测 Q 值持续飙升甚至远超实际环境能达到的理论上限。"
       r"下列关于该现象与 Double DQN 的描述，正确的是？",
       4, "掌握 Q 值高估在工程曲线中的诊断方法与 Double DQN 极简改动的核心机理。", DDQN_SOURCE,
       options=[
           "这说明策略学得特别好，预测 Q 值越高代表小车平衡时间越长。",
           "这是 max 操作符引入的正向高估偏差；Double DQN 仅需用当前策略网络选出动作索引，再用目标网络查出对应分值，即可用极小的改动有效压平虚高估值。",
           "Double DQN 需要将神经网络的参数量增加 4 倍，且必须使用双倍的经验回放池。",
           "预测 Q 值过高是因为学习率太小，只要加大学习率就会自动下降。"],
       _answer=1, feedback_hint="DDQN 通过解耦 argmax（当前网）与 evaluate（目标网），打破了对含噪最大值的盲目乐观。"),

    # ----------------------------------------------------------------------- #
    # Part 2: Hand calculations (q07 - q12, 5 pts each = 30 pts)
    # ----------------------------------------------------------------------- #
    _q("q07", "numeric", "calculation", "CartPole 满分 500 步的理论折扣回报上限手算",
       r"CartPole-v1 环境中，只要小车未倾倒，每前进一步获得即时奖励 \(r = 1.0\)。"
       r"假设某优秀智能体从初始状态 \(t=0\) 开始一直稳定平衡至满分终止步数 \(T = 500\)。"
       r"在折扣因子 \(\gamma = 0.9\) 下，该轨迹在初始时刻 \(t=0\) 的无偏累计折扣回报"
       r" \(G_0 = \sum_{t=0}^{499} \gamma^t \cdot 1.0\) 为多少？"
       r"（已知 \(0.9^{500} \approx 1.32 \times 10^{-23} \approx 0\)，请计算精确到小数点后一位的数值，如 `10.0`）：",
       5, "从强化学习回报定义出发，掌握任务理论最大回报上界的计算，避免盲目调参。", VREP_SOURCE,
       answer_type="number", _expected=10.0, tolerance=1e-2,
       feedback_hint="等比数列求和 G0 = (1 - 0.9^500) / (1 - 0.9) = 1.0 / 0.1 = 10.0。"),

    _q("q08", "numeric", "calculation", "CartPole 单步 Bellman Target 与 TD Loss 手算",
       r"从 CartPole 经验池中抽取一条真实转移：小车从状态 \(s\) 采取向右推（\(a=1\)），获得奖励 \(r=1.0\)，"
       r"进入安全下一状态 \(s'\)（此时未倒地，`done = 0`），折扣因子 \(\gamma = 0.9\)。"
       r"\n已知：\n"
       r"- 当前策略网络对当前状态动作的估值为：\(Q(s, a=1; \theta) = 3.2\)；\n"
       r"- 目标网络在下一状态对两个动作的输出为：\(Q(s', \cdot; \theta^-) = [2.0, 4.0]\)（0:推左, 1:推右）。\n"
       r"请按顺序计算：[该样本的 TD 目标值 \(y\), 该样本的均方误差损失 \(\text{Loss} = (y - Q)^2\)]（如 `4.6, 1.96`）：",
       5, "掌握 DQN 核心 Bellman 目标与单步均方误差损失的完整数值推导。", SEMIGRAD_SOURCE,
       answer_type="list", _expected=[4.6, 1.96], tolerance=1e-3,
       feedback_hint="max Q(s')=4.0; y = 1.0 + 0.9 * 4.0 = 4.6; delta = 4.6 - 3.2 = 1.4; Loss = 1.4^2 = 1.96。"),

    _q("q09", "numeric", "calculation", "漏写 `(1 - done)` 掩码的数值偏差量化手算",
       r"小车在某一步严重倾角倒地，触发回合终止：即时奖励 \(r=1.0\)，`done = 1`，折扣因子 \(\gamma = 0.99\)。"
       r"此时目标网络对倒地状态 \(s'\) 预测的最大 Q 值为 \(\max_{a'} Q(s', a'; \theta^-) = 8.0\)。"
       r"\n请按顺序计算：\n"
       r"1. 带有正确 `(1 - done)` 截断的真实目标值 \(y_{\text{correct}}\)；\n"
       r"2. 错误漏乘 `(1 - done)` 时算出的虚假目标值 \(y_{\text{err}}\)；\n"
       r"3. 漏写此代码给该样本注入的额外 Target 误差值 \(|y_{\text{err}} - y_{\text{correct}}|\)。（如 `1.0, 8.92, 7.92`）：",
       5, "从直观数值上深刻体会漏写终止掩码对 Q 值学习造成的毁灭性偏差。", DONE_SOURCE,
       answer_type="list", _expected=[1.0, 8.92, 7.92], tolerance=1e-3,
       feedback_hint="y_correct = 1.0; y_err = 1.0 + 0.99 * 8.0 = 8.92; 误差差值 = 7.92。"),

    _q("q10", "numeric", "calculation", "Double DQN vs 标准 DQN 目标值实战对比手算",
       r"在 CartPole 下一状态 \(s'\) 中，动作空间为 \(\{0:\text{推左}, 1:\text{推右}\}\)，即时奖励 \(r=1.0\)，\(\gamma = 0.9\)，`done = 0`。"
       r"\n各网络输出如下：\n"
       r"- 当前策略网络（Policy Net）：\(Q(s', \cdot; \theta) = [3.5, 4.2]\)（策略网络倾向推右）；\n"
       r"- 目标网络（Target Net）：\(Q(s', \cdot; \theta^-) = [4.8, 3.8]\)（推左被偶发高估为 4.8）。\n"
       r"请按顺序计算：[标准 DQN 计算出的目标值 \(y_{\text{DQN}}\), Double DQN 计算出的目标值 \(y_{\text{DDQN}}\)]（如 `5.32, 4.42`）：",
       5, "动手验证 Double DQN 如何避免盲目贪婪抓取高估动作，计算解耦目标。", DDQN_SOURCE,
       answer_type="list", _expected=[5.32, 4.42], tolerance=1e-3,
       feedback_hint="DQN: max=4.8, y=1+0.9*4.8=5.32; DDQN: argmax=1, target net evaluate action 1=3.8, y=1+0.9*3.8=4.42。"),

    _q("q11", "numeric", "calculation", "Dueling DQN 均值中心化前向张量数值还原手算",
       r"在小车接近竖直的状态 \(s\) 下，Dueling DQN 网络的双头分支输出如下：\n"
       r"- 标量状态价值流：\(V(s) = 15.0\)；\n"
       r"- 未中心化的优势向量：\(A(s, \cdot) = [2.0, 4.0]\)（对应推左、推右两个动作）。\n"
       r"按照均值中心化前向传播公式：\(Q(s, a) = V(s) + \big(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\big)\)。"
       r"请计算最终输出的两个动作的 Q 值 \([Q(s, a=0), Q(s, a=1)]\)（如 `14.0, 16.0`）：",
       5, "掌握 Dueling 架构均值中心化的计算逻辑，验证其相对优势保持不变。", DUELING_SOURCE,
       answer_type="list", _expected=[14.0, 16.0], tolerance=1e-3,
       feedback_hint="mean(A) = (2 + 4)/2 = 3.0; 中心化优势 = [-1.0, 1.0]; Q = 15.0 + [-1, 1] = [14.0, 16.0]。"),

    _q("q12", "numeric", "calculation", r"CartPole 训练中的 \(\epsilon\)-贪心按轮衰减手算",
       r"在 `train_dqn_500epochs.py` 中，智能体在每个 episode 结束后调用 `agent.decay_epsilon()` 执行指数衰减："
       r"\n\[ \epsilon_{k+1} = \max(\epsilon_{\min}, \epsilon_k \times \text{decay}) \]"
       r"\n设初始探索率 \(\epsilon_0 = 1.0\)，衰减率 \(\text{decay} = 0.90\)，探索下限 \(\epsilon_{\min} = 0.05\)。"
       r"请计算经历完整 3 个 episode 训练并衰减后，在第 4 个 episode 开始时的 \(\epsilon\) 值（精确到小数后三位，如 `0.729`）：",
       5, "掌握探索率退火规律的计算，理解前期充分探索与后期平稳收敛的转换节点。", REPLAY_SOURCE,
       answer_type="number", _expected=0.729, tolerance=1e-3,
       feedback_hint="eps1 = 1.0 * 0.9 = 0.9; eps2 = 0.9 * 0.9 = 0.81; eps3 = 0.81 * 0.9 = 0.729。"),

    # ----------------------------------------------------------------------- #
    # Part 3: Open review & debugging questions (q13 - q16, 6 pts each = 24 pts)
    # ----------------------------------------------------------------------- #
    _q("q13", "open", "explanation", "【代码审查实战】指出这段 DQN 训练循环中的 3 个致命 Bug",
       r"某同学编写了如下一段 PyTorch DQN 的 `update` 方法，测试发现智能体不仅学不会平衡，还报显存不足且 Q 值发散。"
       r"请审查该段代码，指出其中存在的 **3 个致命 Bug** 并简述修改方案：\n"
       r"```python\ndef update(self, buffer, batch_size):\n"
       r"    states, actions, rewards, next_states, dones = buffer.sample(batch_size)\n"
       r"    curr_q = self.policy_net(states).gather(1, actions)\n"
       r"    max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)\n"
       r"    target_q = rewards + self.gamma * max_next_q\n"
       r"    loss = nn.MSELoss()(curr_q, target_q)\n"
       r"    loss.backward()\n"
       r"    self.optimizer.step()\n```",
       6, "具备工业级代码审查能力，精准识别强化学习特有的计算图与截断陷阱。", DONE_SOURCE,
       rubric=[
           {"criterion": "指出漏写 (1 - dones) 截断：target 计算未考虑终止态，倒地后依然累加未来，应改为 rewards + (1 - dones) * self.gamma * max_next_q。", "points": 2},
           {"criterion": "指出目标项缺少 torch.no_grad() 或 .detach()：导致梯度反向传播给 target_net 并驻留显存破坏半梯度，应包在 with torch.no_grad(): 中或 max_next_q.detach()。", "points": 2},
           {"criterion": "指出缺少 self.optimizer.zero_grad()：PyTorch 默认梯度累加，每个 batch 必须在反向传播前清零梯度，否则梯度爆炸。", "points": 2},
       ],
       model_answer=(
           "Bug 1：目标计算漏写终止截断掩码 (1 - dones)。\n"
           "后果：当小车倒下 (done=1) 时，依然累加了下一状态的 Q 值，导致倒地惩罚失效。必须改为：\n"
           "  target_q = rewards + (1 - dones) * self.gamma * max_next_q\n\n"
           "Bug 2：计算 max_next_q 时没有阻断计算图梯度。\n"
           "后果：没有使用 with torch.no_grad() 或 .detach()，反向传播会将梯度回传到 target_net 并在显存中堆积历史计算图，引发显存溢出并破坏半梯度稳定性。应改为：\n"
           "  with torch.no_grad():\n"
           "      max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)\n\n"
           "Bug 3：在 loss.backward() 之前忘记调用 self.optimizer.zero_grad()。\n"
           "后果：PyTorch 默认累加每次 backward 的梯度，导致参数更新幅度急剧膨胀发散。必须在 backward 前清零梯度。"
       )),

    _q("q14", "open", "explanation", "【实验现象复盘】CartPole 训练中 Loss 上升而 Reward 也在上升，如何科学评估策略？",
       r"在 `train_dqn_500epochs.py` 训练曲线中，经常发现第 100 轮左右平均得分从 50 猛增到 400，但同时 MSE Loss 却从 0.05 剧烈上升到 0.8 以上。"
       r"请回答：\n"
       r"1. 为什么在强化学习中，TD Loss 上升并不代表模型在变差？（结合移动靶与分布漂移回答）\n"
       r"2. 实际工程中，为什么不能把 TD Loss 作为早停（Early Stopping）指标？我们应当如何严谨评估当前策略的真实水平？",
       6, "破除监督学习定势，掌握强化学习中动态目标与策略独立评估法则。", TARGET_NET_SOURCE,
       rubric=[
           {"criterion": "从移动靶与状态分布漂移角度解释 Loss 上升：策略变强探索到未见过的高价值长轨迹新状态，标签本身由目标网络生成非固定真值。", "points": 3},
           {"criterion": "指出策略好坏取决于 argmax 相对排序而非绝对 MSE，并给出工程评估方案：必须周期性固定/冻结网络并在 epsilon=0 的纯贪婪模式下运行独立评测环境。", "points": 3},
       ],
       model_answer=(
           "1. 为什么 TD Loss 上升而策略变好：\n"
           "(a) 状态分布剧烈漂移（Distribution Shift）：初始策略极弱（20步就倒下），经验池里全是平庸的短轨迹样本，网络很容易将这些简单的低价值状态拟合到极小 Loss；随着策略进步，小车能坚持到 300~500 步，经验池涌入大量从未见过的高回报新状态，短期内大幅推高了拟合误差。\n"
           "(b) 目标网络的阶跃移动靶：target_q 是由定期同步的目标网络生成的。每当 sync_target 同步时，目标网络的估值大幅跃迁，导致计算出的 MSE 发生脉冲式跳跃。\n\n"
           "2. 为什么不能用 Loss 做早停及如何正确评估：\n"
           "- 智能体的控制决策只取决于 argmax Q(s, a) 的相对排序，只要最优动作比次优动作大 0.1，哪怕全局 Q 值预测有很大的绝对偏差（导致大 Loss），小车依然能拿到 500 分满分。\n"
           "- 正确的评估规范：绝不能看训练 batch 的 Loss，也不能看带探索噪声（epsilon > 0）的训练回报；必须定期在独立的测试环境中，强制将探索率置为 epsilon=0（纯贪心决策），跑 10~20 个完整的 Episode，以平均总回报和稳定性方差作为评估策略好坏的唯一客观金标准。"
       )),

    _q("q15", "open", "explanation", "【架构演进剖析】从 DQN 到 Double DQN 与 Dueling DQN 的工程改造精髓",
       r"在实际开发中，从标准 DQN 升级到 Double DQN 和 Dueling DQN 是最通用的优化路线。"
       r"1. 为什么说 Double DQN 是‘极高性价比’的改造（改动几行代码？解决了什么痛点？）\n"
       r"2. Dueling DQN 将网络拆解为标量 \(V(s)\) 和优势向量 \(A(s, a)\)，为什么在类似 CartPole 这样的大多数状态下，能大幅加速学习？",
       6, "透彻理解两代进阶算法的核心设计哲学与代码改造成本。", DDQN_SOURCE,
       rubric=[
           {"criterion": "说明 Double DQN 仅需两三行代码将 max 改为利用当前网络 argmax 寻优、目标网络打分，以极低改造成本根除了 Q 值虚高问题。", "points": 3},
           {"criterion": "说明 Dueling DQN 能够独立学好状态本身的固有价值 V(s)，无需对每个细微动作重复采样评估，极大提升了动作对状态影响微弱时的样本效率。", "points": 3},
       ],
       model_answer=(
           "1. Double DQN 的极高性价比：\n"
           "- 改造成本：几乎零成本，代码上仅仅修改了 2 行。不再直接调用 target_net.max(1)，而是：\n"
           "    best_a = policy_net(next_states).argmax(dim=1, keepdim=True)\n"
           "    max_next_q = target_net(next_states).gather(1, best_a)\n"
           "- 解决的痛点：彻底切断了传统 DQN 中‘同一套有偏网络既当裁判又当选手’的正向高估恶性循环，使估值曲线平稳真实，避免智能体陷入盲目自大的次优陷阱。\n\n"
           "2. Dueling DQN 加速学习的原理：\n"
           "- 在很多状态下（例如小车非常端正地处于轨道中央），无论选‘推左’还是‘推右’，小车的存活价值其实相差无几，决定当前回报的主要因素是状态本身的安全程度 V(s)。\n"
           "- 传统 DQN 必须对推左和推右分别更新才能学好各自的 Q 值；而 Dueling 架构将 V(s) 独立提取出来，每经历一个样本，V(s) 就会得到更新，使得所有动作的基准价值同时受益，极大节省了在无关紧要状态下探索不同动作的样本量。"
       )),

    _q("q16", "open", "explanation", "【理论与工程闭环】死亡铁三角在深度强化学习中是如何被这三道防线驯服的？",
       r"强化学习理论中著名的‘死亡铁三角’指出：函数逼近 + 自举（TD）+ 离线策略（Off-policy）同时存在时，算法可能发散。"
       r"然而现代 DQN 恰好集齐了这三者（神经网络 + 单步 TD + 经验回放离线数据）。"
       r"请阐述深度强化学习是通过哪 **三道工程防线** 成功抑制发散、将算法拉回稳定收敛轨道的？",
       6, "将深层数学收敛理论与实际系统工程架构完整融会贯通。", DEADLY_SOURCE,
       rubric=[
           {"criterion": "防线一（目标网络）：延迟/冻结自举目标，打碎正反馈自迭代闭环，使非稳态回归转化为阶段性稳态拟合。", "points": 2},
           {"criterion": "防线二（大容量经验回放）：打散时间序列自相关，平滑由于策略变动带来的极端样本分布漂移，保证数据近乎 i.i.d.。", "points": 2},
           {"criterion": "防线三（损失与梯度约束）：采用 Huber 损失（Smooth L1）对大误差梯度截断，配合梯度范数裁剪（clip_grad_norm_），杜绝数值爆炸。", "points": 2},
       ],
       model_answer=(
           "DQN 集齐了死亡铁三角的三大要素，但通过以下三道坚固的工程防线将发散概率降至最低：\n\n"
           "1. 第一道防线：目标网络（Target Network）——打碎自举的正反馈闭环。\n"
           "理论上自举发散是因为参数 w 更新后立即改变了目标 y，形成自我抬升的恶性反馈。DQN 引入目标网络并周期性冻结，使当前几百步内的目标在短时间内保持静止，将危险的自举迭代降级为‘准监督学习’。\n\n"
           "2. 第二道防线：大容量经验回放（Replay Buffer）——平滑离线采样分布。\n"
           "理论上离线策略发散是因为行为策略与目标策略分布不匹配导致投影算子膨胀。通过 10000 步以上的大回放池随机均匀采样，混合了历史多种策略的经验，不仅打破了连续时间序列的强自相关，还大幅平滑了极端状态分布漂移。\n\n"
           "3. 第三道防线：Huber 损失与梯度裁剪（Gradient Clipping）——抑制数值溢出。\n"
           "在神经网络中，偶发的大 TD 误差可能导致极其剧烈的梯度更新，击穿网络权重。通过将 MSE 替换为 Huber Loss，在大误差区间将梯度强制截断为常数（|g| <= 1），配合 torch.nn.utils.clip_grad_norm_，彻底筑牢了数值防线。"
       )),

    # ----------------------------------------------------------------------- #
    # Part 4: Practical coding questions (q17 - q21, 6 pts each = 30 pts)
    # ----------------------------------------------------------------------- #
    _q("q17", "code", "implementation", r"实现 \(\epsilon\)-贪心动作选择器",
       r"实现函数 `select_action(q_values, epsilon, num_actions, random_val=None)`。\n"
       r"参数说明：\n"
       r"- `q_values`: 当前状态下各离散动作的 Q 值数组（NumPy array 或列表）；\n"
       r"- `epsilon`: 探索概率 \(\epsilon\)；\n"
       r"- `num_actions`: 可用动作总数（如 CartPole 为 2）；\n"
       r"- `random_val`: 传入浮点数时（测试用），若 `random_val < epsilon` 则执行探索返回 `0`；为 `None` 时通过 `random.random()` 抽样。\n"
       r"逻辑要求：以概率 \(\epsilon\) 探索（返回随机动作索引），以概率 \(1 - \epsilon\) 贪心（返回 \(\arg\max\) 动作索引）。",
       6, "熟练掌握强化学习探索与利用基础控制算子的矢量化实现。", REPLAY_SOURCE,
       entry_point="select_action",
       starter_code="""def select_action(q_values, epsilon, num_actions, random_val=None):
    # return: integer action index
    pass
""",
       public_tests=[
           {"name": "greedy_selection", "description": "验证当 random_val >= epsilon 时严格执行 argmax 贪心决策。",
            "code": """q = [2.1, 5.8, 1.4]
a = select_action(q, epsilon=0.1, num_actions=3, random_val=0.5)
assert a == 1, f"Expected greedy action 1, got {a}"
"""},
           {"name": "explore_selection", "description": "验证当 random_val < epsilon 时触发探索分支。",
            "code": """q = [2.1, 5.8, 1.4]
a = select_action(q, epsilon=0.8, num_actions=3, random_val=0.2)
assert a == 0, f"Expected exploration branch action 0, got {a}"
"""},
       ],
       rubric=[
           {"criterion": "根据 epsilon 阈值准确分流随机探索分支与 argmax 贪心分支。", "points": 3},
           {"criterion": "正确处理数组与列表输入并返回标准整数动作索引。", "points": 3},
       ],
       _solution=_SELECT_ACTION_SOLUTION),

    _q("q18", "code", "implementation", "实现带 `done` 掩码的 DQN 目标计算函数",
       r"实现函数 `compute_dqn_targets(rewards, next_q_max, dones, gamma)`。\n"
       r"参数说明：\n"
       r"- `rewards`: 批次即时奖励，shape 为 `[B]` 或 `[B, 1]`；\n"
       r"- `next_q_max`: 目标网络在下一状态的最大 Q 值，shape 为 `[B]` 或 `[B, 1]`；\n"
       r"- `dones`: 终止布尔标志（0 或 1），shape 为 `[B]` 或 `[B, 1]`；\n"
       r"- `gamma`: 折扣因子标量。\n"
       r"计算公式：\(y = r + (1 - \text{done}) \cdot \gamma \cdot \max_{a'} Q(s', a')\)。\n"
       r"返回 shape 为 `[B, 1]` 的二维列向量（NumPy array）。",
       6, "严密实现带终止截断掩码的标准 DQN 目标张量计算。", DONE_SOURCE,
       entry_point="compute_dqn_targets",
       starter_code="""def compute_dqn_targets(rewards, next_q_max, dones, gamma):
    # return: numpy array of shape [B, 1]
    pass
""",
       public_tests=[
           {"name": "check_hand_calc_q08", "description": "验证第q08题手算算例，非终止转移目标值。",
            "code": """import numpy as np
y = compute_dqn_targets([1.0], [4.0], [0.0], 0.9)
assert np.isclose(y[0, 0], 4.6, atol=1e-5), f"Expected 4.6, got {y[0,0]}"
"""},
           {"name": "check_done_masking_q09", "description": "验证第q09题手算算例，当 done=1 时未来价值严格截断为 0。",
            "code": """import numpy as np
y = compute_dqn_targets([1.0], [8.0], [1.0], 0.99)
assert np.isclose(y[0, 0], 1.0, atol=1e-5), f"Expected 1.0 with done=1, got {y[0,0]}"
"""},
       ],
       rubric=[
           {"criterion": "使用 (1 - dones) 掩码准确处理终止态，切断未来价值传播。", "points": 3},
           {"criterion": "严谨处理单维与多维广播，返回标准的 [B, 1] 列向量格式。", "points": 3},
       ],
       _solution=_COMPUTE_DQN_TARGETS_SOLUTION),

    _q("q19", "code", "implementation", "实现 Double DQN 交叉解耦目标计算函数",
       r"实现函数 `compute_ddqn_targets(rewards, q_eval_next, q_target_next, dones, gamma)`。\n"
       r"参数说明：\n"
       r"- `rewards`: 批次即时奖励，shape `[B]` 或 `[B, 1]`；\n"
       r"- `q_eval_next`: 当前策略网络在下一状态输出的各动作估值，shape `[B, num_actions]`；\n"
       r"- `q_target_next`: 目标网络在下一状态输出的各动作估值，shape `[B, num_actions]`；\n"
       r"- `dones`: 终止标志批次，shape `[B]` 或 `[B, 1]`；\n"
       r"- `gamma`: 折扣因子。\n"
       r"Double DQN 计算逻辑：\n"
       r"1. 由策略网络选取最优动作：\(a^* = \arg\max_a Q_{\text{eval}}(s', a)\)；\n"
       r"2. 由目标网络评估该动作分值：\(Q_{\text{target}}(s', a^*)\)；\n"
       r"3. 组合目标值：\(y = r + (1 - \text{done}) \cdot \gamma \cdot Q_{\text{target}}(s', a^*)\)。\n"
       r"返回 shape 为 `[B, 1]` 的 NumPy 数组。",
       6, "熟练实现 Double DQN 动作选择与估值解耦的矢量化逻辑。", DDQN_SOURCE,
       entry_point="compute_ddqn_targets",
       starter_code="""def compute_ddqn_targets(rewards, q_eval_next, q_target_next, dones, gamma):
    # return: numpy array of shape [B, 1]
    pass
""",
       public_tests=[
           {"name": "check_hand_calc_q10", "description": "检验第q10题手算算例，验证 DDQN 消除高估。",
            "code": """import numpy as np
rewards = [1.0]
q_eval = [[3.5, 4.2]]
q_targ = [[4.8, 3.8]]
dones = [0.0]
y = compute_ddqn_targets(rewards, q_eval, q_targ, dones, 0.9)
assert np.isclose(y[0, 0], 4.42, atol=1e-5), f"Expected 4.42, got {y[0,0]}"
"""},
           {"name": "batch_matrix_test", "description": "验证批次数据下多样本矢量化解耦计算。",
            "code": """import numpy as np
rewards = [1.0, 2.0]
q_eval = [[1.0, 5.0], [8.0, 2.0]]
q_targ = [[3.0, 4.0], [7.0, 9.0]]
dones = [0.0, 1.0]
y = compute_ddqn_targets(rewards, q_eval, q_targ, dones, 0.5)
assert np.isclose(y[0, 0], 1.0 + 0.5 * 4.0, atol=1e-5)
assert np.isclose(y[1, 0], 2.0, atol=1e-5)
"""},
       ],
       rubric=[
           {"criterion": "使用 q_eval 沿动作轴准确求出 argmax 动作索引并在 q_target 中按行提取分值。", "points": 3},
           {"criterion": "正确结合奖励、折扣与 done 掩码返回形状严谨的二维列向量。", "points": 3},
       ],
       _solution=_COMPUTE_DDQN_TARGETS_SOLUTION),

    _q("q20", "code", "implementation", "实现 Dueling DQN 均值中心化前向传播",
       r"实现函数 `dueling_forward(v_s, a_sa)`。\n"
       r"参数说明：\n"
       r"- `v_s`: 状态价值分支输出，shape 为 `[B, 1]`；\n"
       r"- `a_sa`: 动作优势分支输出，shape 为 `[B, num_actions]`；\n"
       r"计算公式：\n"
       r"\[ Q(s, a) = V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'=1}^{|\mathcal{A}|} A(s, a')\Big) \]\n"
       r"返回 shape 为 `[B, num_actions]` 的二维 Q 值数组。",
       6, "掌握 Dueling 架构均值中心化的向量化广播实现与物理意义。", DUELING_SOURCE,
       entry_point="dueling_forward",
       starter_code="""def dueling_forward(v_s, a_sa):
    # return: numpy array of shape [B, num_actions]
    pass
""",
       public_tests=[
           {"name": "check_hand_calc_q11", "description": "检验第q11题手算数值，验证中心化优势与价值叠加。",
            "code": """import numpy as np
v_s = np.array([[15.0]])
a_sa = np.array([[2.0, 4.0]])
q = dueling_forward(v_s, a_sa)
assert np.allclose(q, [[14.0, 16.0]], atol=1e-5), f"Expected [[14, 16]], got {q}"
"""},
           {"name": "advantage_mean_property", "description": "验证输出 Q 值沿动作维度的平均值在数值上严格等于 V(s)。",
            "code": """import numpy as np
v_s = np.array([[10.0], [5.0]])
a_sa = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
q = dueling_forward(v_s, a_sa)
assert np.allclose(np.mean(q, axis=1, keepdims=True), v_s, atol=1e-5)
"""},
       ],
       rubric=[
           {"criterion": "正确计算优势向量沿动作维度的均值，并执行广播减法完成中心化。", "points": 3},
           {"criterion": "将状态价值与中心化优势正确广播相加，输出维度与原优势矩阵完全一致。", "points": 3},
       ],
       _solution=_DUELING_FORWARD_SOLUTION),

    _q("q21", "code", "implementation", "实现目标网络 Polyak 软更新函数",
       r"实现函数 `soft_update(target_weights, policy_weights, tau)`。\n"
       r"参数说明：\n"
       r"- `target_weights`: 目标网络参数数组（NumPy array 或列表）；\n"
       r"- `policy_weights`: 当前策略网络参数数组；\n"
       r"- `tau`: 平滑更新系数（浮点数，如 0.05）；\n"
       r"更新公式：\n"
       r"\[ \theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^- \]\n"
       r"返回更新后的目标参数 NumPy 数组副本。",
       6, "实现连续平滑跟踪目标网络更新算子。", TARGET_NET_SOURCE,
       entry_point="soft_update",
       starter_code="""def soft_update(target_weights, policy_weights, tau):
    # return: updated target weights array
    pass
""",
       public_tests=[
           {"name": "soft_update_interpolation", "description": "验证指数移动平均平滑插值。",
            "code": """import numpy as np
t = np.array([2.0, 4.0])
s = np.array([10.0, 20.0])
out = soft_update(t, s, tau=0.1)
assert np.allclose(out, [2.8, 5.6], atol=1e-5)
"""},
           {"name": "extreme_tau_cases", "description": "验证 tau=0.0 和 tau=1.0 边界行为。",
            "code": """import numpy as np
t = np.array([1.0])
s = np.array([5.0])
assert np.isclose(soft_update(t, s, 0.0)[0], 1.0, atol=1e-5)
assert np.isclose(soft_update(t, s, 1.0)[0], 5.0, atol=1e-5)
"""},
       ],
       rubric=[
           {"criterion": "严谨实现线性平滑插值公式 tau * s + (1 - tau) * t。", "points": 3},
           {"criterion": "正确处理数组运算并返回无副效应的独立 NumPy 数组副本。", "points": 3},
       ],
       _solution=_SOFT_UPDATE_SOLUTION),
]

TOTAL_SCORE = sum(q["max_score"] for q in QUESTIONS)

SCORING = {
    "objective": sum(q["max_score"] for q in QUESTIONS if q["grading_mode"] == "auto"),
    "open_self_review": sum(q["max_score"] for q in QUESTIONS if q["grading_mode"] == "self"),
    "code_browser_tests": sum(q["max_score"] for q in QUESTIONS if q["grading_mode"] == "browser"),
}

_INDEX = {q["id"]: q for q in QUESTIONS}


def list_questions():
    return list(QUESTIONS)


def _get(question_id):
    try:
        return _INDEX[question_id]
    except (KeyError, TypeError):
        raise bank.UnknownQuestion("unknown question id: %r" % (question_id,))


def public_payload():
    return {
        "reference": REFERENCE, "conventions": CONVENTIONS,
        "presentation": PRESENTATION, "industry_case": None,
        "count": len(QUESTIONS), "total_score": TOTAL_SCORE,
        "scoring": dict(SCORING),
        "questions": [bank.public_question(q) for q in QUESTIONS],
    }


def grade(question_id, answer):
    question = _get(question_id)
    if question["type"] == "choice":
        index = bank._normalize_choice(answer, question["options"])
        correct = index == question["_answer"]
        expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[question["_answer"]]
    elif question["type"] == "numeric":
        try:
            values = bank._as_float_list(answer)
        except OverflowError as exc:
            raise bank.InvalidAnswer("numeric answer must be finite") from exc
        expected = question["_expected"]
        wanted = list(expected) if question["answer_type"] == "list" else [expected]
        if len(values) != len(wanted):
            raise bank.InvalidAnswer(f"答案个数不符，期望 {len(wanted)} 个数")
        correct = all(math.isclose(a, b, rel_tol=question["tolerance"],
                                   abs_tol=question["tolerance"])
                      for a, b in zip(values, wanted))
    else:
        raise bank.NotAutoGradable("解释题请自评；代码题只在浏览器运行公开测试。")
    return {
        "correct": correct, "score": question["max_score"] if correct else 0,
        "max_score": question["max_score"], "expected": expected,
        "feedback": ("回答正确。" if correct else
                     "回答不正确。" + question["feedback_hint"]),
    }


def reveal(question_id):
    question = _get(question_id)
    if question["type"] not in ("open", "code"):
        raise bank.NotRevealable("选择或计算题请先提交答案。")
    result = {
        "id": question["id"], "type": question["type"],
        "max_score": question["max_score"],
        "self_assessment": question["type"] == "open",
        "rubric": [dict(item) for item in question["rubric"]],
    }
    if question["type"] == "open":
        result["model_answer"] = question["model_answer"]
    else:
        result["entry_point"] = question["entry_point"]
        result["reference_solution"] = question["_solution"]
    return result
