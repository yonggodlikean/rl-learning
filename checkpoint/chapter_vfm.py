"""Value function methods: complete textbook-grounded curriculum and engineering practice.

Grounded in:
- Zhao Shiyu《Mathematical Foundations of Reinforcement Learning》Chapter 8 (pp. 151-190)
  - §8.1 Motivation & State value function approximation (pp. 151–155)
  - §8.2 TD learning of state values with function approximation (pp. 156–178)
  - §8.3 TD learning of action values: Sarsa vs Q-learning (pp. 179–181)
  - §8.4 Deep Q-Learning (DQN) (pp. 181–186)
  - §8.6 Summary & The Deadly Triad (pp. 187–190)
- EasyRL v1.0.6:
  - Chapter 6: Deep Q-Network foundations & training (pp. 93–105)
  - Chapter 7: DQN improvements (Double DQN, Dueling DQN, PER) (pp. 106–119)
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
    "本题库严格按照赵世钰第8章大框架与 EasyRL 第6–7章教学逻辑递进设计："
    "§8.1 范式跃迁 → §8.2 状态价值半梯度TD与PBE投影 → §8.3 动作价值逼近与控制 → "
    "§8.4 深度Q网络基石与工程实现 → EasyRL Ch7 DQN进阶优化技巧。"
    "在 DQN 与通用强化学习中，经验元组采用 (s, a, r, s', done)，当 done=True 时未来价值必须截断为 0。"
    "半梯度对应 PyTorch 中使用 torch.no_grad() 或 .detach() 切断目标项的计算图梯度传播。"
    "Double DQN 采用当前网络选择贪心动作索引，目标网络评估该动作分值以消除高估偏差。"
    "手算题所有参数与转移数据均已自包含，无需翻找前后文。"
)

PRESENTATION = {
    "eyebrow": "VALUE FUNCTION METHODS / 教材大框架与工程实践系统精通",
    "intro": (
        "严格遵循赵世钰《强化学习的数学原理》第8章知识演进逻辑，深度融合 EasyRL 第6–7章工程实战：\n"
        "从为什么必须从表格跨越到函数逼近（§8.1），到状态价值的半梯度 TD 学习、PBE 投影与死亡铁三角（§8.2）；\n"
        "从动作价值逼近 Sarsa 与 Q-learning 的收敛性分野（§8.3），到深度 Q 网络（DQN）两大基石机制与 CartPole 避坑实战（§8.4）；\n"
        "再到 Double DQN、Dueling DQN、PER 与 Huber 损失等高阶工程演进（EasyRL Ch 7）。\n"
        "理论层层递进，手算步步印证，代码落地生根，构建对值函数方法全景的深刻理解。"
    ),
    "field_notes": [
        {"label": "01 / SEMI-GRADIENT & NO-GRAD",
         "formula": "w_{t+1} = w_t + alpha * [r + gamma * v_hat(s', w_t) - v_hat(s, w_t)] * grad_w v_hat",
         "tex": r"w_{t+1} = w_t + \alpha_t \big[r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t)\big] \nabla_w \hat{v}(s_t, w_t)",
         "description": "半梯度将自举目标视为常数靶子；在 PyTorch 中使用 torch.no_grad() 阻断目标回传。"},
        {"label": "02 / DEADLY TRIAD",
         "formula": "Function Approx + Bootstrapping + Off-policy => Divergence",
         "tex": r"\text{Function Approximation} + \text{Bootstrapping} + \text{Off-policy} \implies \text{Divergence}",
         "description": "投影算子在非稳态分布下可能膨胀；DQN 依靠经验回放与目标网络筑起工程防线。"},
        {"label": "03 / DOUBLE DQN DECOUPLING",
         "formula": "a* = argmax Q(s', a; theta), y = r + (1 - done) * gamma * Q(s', a*; theta^-)",
         "tex": r"a^* = \arg\max_a Q(s', a; \theta),\quad y = r + (1 - \text{done}) \gamma Q(s', a^*; \theta^-)",
         "description": "当前策略网络选动作，目标网络评价值，两行代码消除 max 算子的正向高估偏差。"},
        {"label": "04 / DUELING IDENTIFIABILITY",
         "formula": "Q(s, a) = V(s) + [A(s, a) - mean_a'(A(s, a'))]",
         "tex": r"Q(s, a) = V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\Big)",
         "description": "优势流均值中心化保证唯一可辨识性，不增加采样开销即可加速状态价值共享学习。"},
    ],
    "note_footer": (
        "理论指导调参，代码印证理论：全面掌握值函数逼近的数学逻辑底色与现代工程实现细节。"
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
                      "从表格到函数逼近：连续状态空间、维度灾难、参数共享泛化与特征干扰。")
SEMIGRAD_SOURCE = _source("8.2.1–8.2.2", "156–162", "169–175",
                          "6.2", "96–99", "104–107",
                          "平稳分布加权均方误差、半梯度与残差梯度的区别、PyTorch no_grad。")
LINEAR_FA_SOURCE = _source("8.2.3", "162–167", "175–180",
                           "6.1–6.2", "93–99", "101–107",
                           "线性函数逼近更新、特征工程（多项式、傅里叶、RBF、状态聚合）。")
THEORY_PBE_SOURCE = _source("8.2.4", "168–175", "181–188",
                            "6.2", "96–99", "104–107",
                            "投影 Bellman 误差（PBE）、投影算子 Pi 与几何不动点收敛分析。")
DEADLY_SOURCE = _source("8.2.5、8.6", "176–178、187–190", "189–191、200–203",
                        "6.2–6.3", "97–101", "105–109",
                        "死亡铁三角发散机理、Baird 反例与现代工程防线。")
ACTION_VALUE_SOURCE = _source("8.3", "179–181", "192–194",
                             "6.2", "96–99", "104–107",
                             "动作价值逼近：为什么控制必须逼近 q(s,a)；Sarsa 与 Q-learning 的收敛性对比。")
DQN_CORE_SOURCE = _source("8.4.1", "181–184", "194–197",
                          "6.2–6.3", "96–101", "104–109",
                          "DQN 经验回放打破时序相关性、目标网络消除移动靶、done 掩码与 CartPole 实战。")
DDQN_SOURCE = _source("8.4.1", "182–184", "195–197",
                      "7.1", "106–109", "114–117",
                      "高估偏差数学成因（Jensen 不等式与 max）及 Double DQN 解耦机制。")
DUELING_SOURCE = _source("8.4.1", "182–184", "195–197",
                        "7.2", "109–112", "117–120",
                        "Dueling DQN 架构、可辨识性与均值中心化前向传播。")
PER_HUBER_SOURCE = _source("8.4.1", "182–184", "195–197",
                           "7.3–7.4", "112–119", "120–127",
                           "优先经验回放（PER）重要性采样校正、Huber 损失与梯度裁剪。")


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

_LINEAR_Q_UPDATE_SOLUTION = """def linear_q_learning_update(w, phi_sa, reward, next_phi_list, gamma, alpha):
    import numpy as np
    w_arr = np.array(w, dtype=float)
    phi_cur = np.array(phi_sa, dtype=float)
    next_phis = np.array(next_phi_list, dtype=float)
    q_cur = float(np.dot(phi_cur, w_arr))
    q_next = np.dot(next_phis, w_arr)
    q_next_max = float(np.max(q_next))
    td_target = reward + gamma * q_next_max
    td_error = td_target - q_cur
    return w_arr + alpha * td_error * phi_cur
"""

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
# Questions definitions: 40 questions total, 190 pts
# Organized strictly along the textbook curriculum:
# Stage 1: 表格到函数逼近的范式跃迁 (§8.1) -> q01-q04
# Stage 2: 状态价值的半梯度 TD 学习与理论投影 (§8.2) -> q05-q11
# Stage 3: 动作价值逼近：Sarsa vs Q-learning 与控制 (§8.3) -> q12-q16
# Stage 4: 深度 Q 网络（DQN）两大基石与工程实现 (§8.4 & EasyRL Ch6) -> q17-q29
# Stage 5: DQN 进阶技巧：Double DQN、Dueling DQN、PER 与 Huber 损失 (EasyRL Ch7) -> q30-q40
# =========================================================================== #

QUESTIONS = [
    # ----------------------------------------------------------------------- #
    # Stage 1: 表格到函数逼近的范式跃迁 (§8.1 & EasyRL §6.1)
    # ----------------------------------------------------------------------- #
    _q("q01", "choice", "concept", "强化学习为何必须跨越到函数逼近？表格型方法的维度诅咒",
       r"在前几章（MC、TD、Sarsa、Q-learning）中，价值函数均存储在二维表格 \(V(s)\) 或 \(Q(s, a)\) 中。"
       r"但在围棋（约 \(10^{170}\) 状态）或连续控制任务（如 CartPole 的位置、速度、摆角、角速度）中，表格法彻底失效。"
       r"关于从表格型向函数逼近（Function Approximation）转变的根本动力，下列哪项说法最准确？",
       4, "深刻理解从离散表格到连续函数逼近的理论必然性与泛化核心价值。", VREP_SOURCE,
       options=[
           "仅仅是因为现代计算机内存不足，若内存无限大，表格型方法就能够完美且高效地解决连续状态控制问题。",
           "连续状态空间包含无限多状态，在有限步采样内绝大多数状态永远无法被精确访问；函数逼近不仅压缩了存储，更核心的是利用共享参数实现了从未见状态到相似状态的泛化（Generalization）。",
           "函数逼近可以将强化学习直接转化为标准的无监督学习聚类任务，不再需要与环境进行交互采样。",
           r"表格型方法在数学上无法计算带折扣因子的累积回报，只有引入参数化函数后才能定义 \(\gamma\)。"],
       _answer=1, feedback_hint="函数逼近的核心不仅是节省空间，更在于利用参数共享赋予智能体对未见状态的归纳泛化能力。"),

    _q("q02", "choice", "concept", "函数逼近的双刃剑：参数共享泛化 vs 跨状态表示干扰",
       r"在线性逼近 \(\hat{v}(s, w) = \phi(s)^T w\) 或深度神经网络 \(Q(s, a; \theta)\) 中，所有状态共享同一套权重 \(w\) 或 \(\theta\)。"
       r"当小车在状态 \(s_1\)（角度 0.05 弧度）执行动作获得奖励并更新权重后，小车在状态 \(s_2\)（角度 0.20 弧度）的 Q 值估值也会连带改变。"
       r"下列关于这种“状态间关联耦合”现象的评价，哪项最为客观准确？",
       4, "掌握参数共享带来的泛化收益与跨状态干扰/遗忘代价之间的本质权衡。", VREP_SOURCE,
       options=[
           "这是函数逼近的核心优势（让相似状态自动获得合理解释），但若采样分布严重偏倚，高频访问的状态会冲刷低频稀有关键状态的参数，导致灾难性遗忘与表示干扰（Representation Interference）。",
           "这是一种严重的代码 Bug，说明全连接网络层数太少，只要增加到 100 层以上，各状态之间的参数更新就会完全相互独立。",
           "这说明神经网络无法应用于强化学习，工业界遇到连续状态任务必须全部退回分桶表格法。",
           r"这种状态耦合可以通过将学习率 \(\alpha\) 调大到 10 以上完全消除。"],
       _answer=0, feedback_hint="参数共享是一把双刃剑：它赋予了未见状态合理的先验初值，但也带来了频繁经验冲刷冷门状态的表示干扰。"),

    _q("q03", "choice", "concept", r"状态特征工程：特征向量 \(\phi(s)\) 的设计哲学",
       r"在深度学习普及之前，经典的函数逼近主要依赖人工设计的特征向量 \(\phi(s) = [\phi_1(s), \dots, \phi_d(s)]^T\)。"
       r"常见的特征构造包括多项式特征、傅里叶基底（Fourier Basis）、径向基函数（RBF）与状态聚合（State Aggregation）。"
       r"关于特征向量 \(\phi(s)\)，下列说法错误的是？",
       4, "理解经典特征工程构造方法及其与线性函数逼近非线性表达能力的关系。", LINEAR_FA_SOURCE,
       options=[
           "状态聚合（State Aggregation）本质上是将相邻的状态划分到同一个集合/桶中，属于一种特殊的分段常数线性特征。",
           "径向基函数（RBF）通过高斯核度量状态与预设中心点的距离，能提供局部平滑的非线性表示能力。",
           r"线性函数逼近的输出关于参数 \(w\) 是线性的，因此即使使用高阶多项式或高斯 RBF 特征，也绝对无法拟合非线性价值函数曲面。",
           "深度神经网络可以被视为端到端的自适应特征提取器，免去了人工繁琐设计特征基底的过程。"],
       _answer=2, feedback_hint="线性函数逼近指的是关于参数 w 是线性的，但特征基底 phi(s) 可以是高度非线性的，因此完全能拟合复杂的非线性曲面。"),

    _q("q04", "numeric", "calculation", "线性价值函数特征加权估值手算",
       r"某连续一维走廊环境，设计了 3 维特征向量 \(\phi(s) = [1.0, s, s^2]^T\)（分别包含常数偏置项、线性项与二次项）。"
       r"当前参数权重向量为 \(w = [2.0, -1.0, 0.5]^T\)。"
       r"请计算在状态 \(s = 2.0\) 处，线性逼近器预测的状态价值 \(\hat{v}(s, w) = \phi(s)^T w\)（如 `2.0`）：",
       5, "掌握线性价值函数内积计算的基本步骤。", LINEAR_FA_SOURCE,
       answer_type="number", _expected=2.0, tolerance=1e-3,
       feedback_hint="phi(2.0) = [1.0, 2.0, 4.0]^T; v_hat = 1.0*2.0 + 2.0*(-1.0) + 4.0*0.5 = 2.0 - 2.0 + 2.0 = 2.0。"),

    # ----------------------------------------------------------------------- #
    # Stage 2: 状态价值的半梯度 TD 学习与理论投影 (§8.2 & EasyRL §6.2)
    # ----------------------------------------------------------------------- #
    _q("q05", "choice", "concept", r"为什么是平稳分布加权均方误差？目标函数 \(J(w)\) 的物理意义",
       r"在对真实价值 \(v_\pi\) 进行函数逼近时，Zhao §8.2.1 定义了目标函数："
       r"\n\[ J(w) = \mathbb{E}_{S \sim d_\pi} [(v_\pi(S) - \hat{v}(S, w))^2] = \sum_{s \in \mathcal{S}} d_\pi(s) (v_\pi(s) - \hat{v}(s, w))^2 \]"
       r"\n其中 \(d_\pi(s)\) 为策略 \(\pi\) 下的马尔可夫链平稳状态分布（Stationary Distribution）。为什么要引入权重 \(d_\pi(s)\)？",
       4, "透彻理解平稳分布在状态价值均方误差加权中的物理与采样意义。", SEMIGRAD_SOURCE,
       options=[
           r"因为平稳分布保证了在策略 \(\pi\) 下频繁访问的高价值/关键状态被赋予更高的拟合优先级，且使得期望可以直接通过与环境交互的在线轨迹采样进行无偏估计。",
           r"纯粹为了数学上凑出好看的导数形式，在实际算法采样与参数更新中没有任何物理意义。",
           r"\(d_\pi(s)\) 表示当前状态下智能体选择动作的概率分布，仅在离散动作空间中有效。",
           r"如果不乘 \(d_\pi(s)\)，均方误差在凸优化理论中会导致梯度永远为 0。"],
       _answer=0, feedback_hint="d_pi(s) 使得频繁访问的状态优先学准，且环境采样轨迹的自然出现频率恰好服从 d_pi，天然支持蒙特卡洛/TD无偏估计。"),

    _q("q06", "choice", "concept", "半梯度（Semi-gradient）与残差梯度（Residual Gradient）的本质区别",
       r"在推导 TD(0) 结合函数逼近的参数更新时，目标函数 \(J(w)\) 的真实梯度需要未知的真值 \(v_\pi(s)\)。"
       r"若将 \(v_\pi(s)\) 替换为单步自举目标 \(y_t = r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)\)：\n"
       r"- 半梯度方法：在对 \(w\) 求偏导时，将 \(y_t\) 视为常数靶子，仅对预测项 \(\hat{v}(s_t, w)\) 求导；\n"
       r"- 残差梯度法（Residual Gradient）：对 \(y_t\) 中的 \(w\) 也同时应用链式法则求导。\n"
       r"关于为什么主流强化学习普遍采用半梯度而不是残差梯度法，下列解释最严谨的是？",
       4, "从双采样障碍与收敛目标本质辨析半梯度与残差梯度法的关键差异。", SEMIGRAD_SOURCE,
       options=[
           "残差梯度法计算图过于简单，无法利用 GPU 的并行加速能力。",
           "残差梯度法在更新时需要对同一个状态同时采样两个独立的下一状态转移（双采样问题），在真实环境交互中通常不可行；且残差梯度法最小化的是 Bellman 误差，其极小值点并不等于真实的 Bellman 不动点。",
           "半梯度的更新方向与目标函数的真实负梯度完全重合，属于标准的凸优化随机梯度下降（SGD）。",
           "残差梯度法只能用于单层感知机，深度神经网络无法计算其链式法则。"],
       _answer=1, feedback_hint="残差梯度求导包含 E[target - v] * E[grad_target - grad_v]，展开需要两套独立采样的转移样本，且收敛点偏离真实值。"),

    _q("q07", "choice", "concept", "线性半梯度 TD(0) 更新公式与梯度展开",
       r"对于线性逼近 \(\hat{v}(s, w) = \phi(s)^T w\)，其关于参数 \(w\) 的梯度为 \(\nabla_w \hat{v}(s, w) = \phi(s)\)。"
       r"在经历了转移 \((s_t, r_{t+1}, s_{t+1})\) 后，线性半梯度 TD(0) 的单步参数更新公式为：",
       4, "熟记线性半梯度 TD(0) 的代数形式与特征向量更新方向。", LINEAR_FA_SOURCE,
       options=[
           r"\(w_{t+1} = w_t + \alpha_t \big[r_{t+1} + \gamma \phi(s_{t+1})^T w_t - \phi(s_t)^T w_t\big] \phi(s_t)\)",
           r"\(w_{t+1} = w_t + \alpha_t \big[r_{t+1} + \gamma \phi(s_t)^T w_t - \phi(s_{t+1})^T w_t\big] \phi(s_{t+1})\)",
           r"\(w_{t+1} = w_t - \alpha_t \big[r_{t+1} + \gamma \phi(s_{t+1})^T w_t\big] \phi(s_t)\)",
           r"\(w_{t+1} = w_t + \alpha_t \big[r_{t+1} + \gamma \phi(s_{t+1})^T w_t - \phi(s_t)^T w_t\big]^2 \phi(s_{t+1})\)"],
       _answer=0, feedback_hint="更新项为 alpha * delta_t * grad_w v_hat = alpha * [r + gamma * phi(s')^T w - phi(s)^T w] * phi(s)。"),

    _q("q08", "numeric", "calculation", "线性半梯度 TD(0) 单步参数更新手算",
       r"设环境状态由 2 维特征表示，当前权重向量为 \(w_0 = [1.0, 2.0]^T\)。"
       r"智能体在状态 \(s_t\)（特征 \(\phi(s_t) = [1.0, 0.0]^T\)）采取动作，获得即时奖励 \(r_{t+1} = 2.0\)，"
       r"转移到下一状态 \(s_{t+1}\)（特征 \(\phi(s_{t+1}) = [0.0, 1.0]^T\)）。"
       r"折扣因子 \(\gamma = 0.5\)，学习率 \(\alpha = 0.1\)。"
       r"请按顺序计算更新后的一步新权重向量 \(w_1 = [w_{1, 1}, w_{1, 2}]\)（如 `1.2, 2.0`）：",
       5, "亲手演算线性半梯度 TD(0) 的价值预测、TD 误差与权重更新全过程。", LINEAR_FA_SOURCE,
       answer_type="list", _expected=[1.2, 2.0], tolerance=1e-3,
       feedback_hint="v(s_t)=1.0, v(s_{t+1})=2.0; y=2.0+0.5*2.0=3.0; delta=3.0-1.0=2.0; w1 = w0 + 0.1*2.0*[1.0, 0.0]^T = [1.2, 2.0]^T。"),

    _q("q09", "choice", "concept", "几何投影视角：投影 Bellman 误差（PBE）与线性 TD 的不动点",
       r"在 Zhao §8.2.4 中，真实 Bellman 算子更新后的值 \(T_\pi \hat{v}\) 通常会溢出特征子空间。"
       r"加权正交投影算子 \(\Pi = \Phi (\Phi^T D \Phi)^{-1} \Phi^T D\) 将任意价值向量投影回特征空间 \(\mathcal{S}_w = \{\Phi w\}\)。"
       r"关于线性半梯度 TD 算法最终收敛到的解 \(w^*\)，下列哪项数学表述是正确的？",
       4, "从函数空间正交投影与不动点方程角度掌握线性 TD 的几何收敛机理。", THEORY_PBE_SOURCE,
       options=[
           r"它是无约束 Bellman 误差的绝对极小点：\(\|T_\pi (\Phi w^*) - \Phi w^*\|_D = 0\)",
           r"它是复合投影 Bellman 算子的不动点：\(\Pi T_\pi (\Phi w^*) = \Phi w^*\)（即投影 Bellman 误差 PBE 为 0）",
           r"它是环境全局最优策略的精确真实值：\(\Phi w^* = v^*\)",
           r"它是均方值误差（Mean Squared Value Error）的全局极值点"],
       _answer=1, feedback_hint="线性 TD 并不是在找 T_pi 的不动点（往往不可行），而是收敛到投影算子与 Bellman 算子复合映射 Pi * T_pi 的不动点。"),

    _q("q10", "choice", "concept", "理论界的噩梦：死亡铁三角（Deadly Triad）的发散机理与 Baird 反例",
       r"Sutton 与 Barto 指出的“死亡铁三角（Deadly Triad）”是强化学习理论中最著名的不稳定定理。"
       r"当且仅当哪三项要素同时存在时，自举算法可能会面临估计值无界发散与参数爆炸？",
       4, "精准牢记死亡铁三角的三大构成要素与系统失稳的边界条件。", DEADLY_SOURCE,
       options=[
           "策略梯度（Policy Gradient） + 连续动作空间 + 确定性策略网络",
           "函数逼近（Function Approximation） + 自举更新（Bootstrapping / TD） + 离线策略学习（Off-policy learning）",
           "经验回放（Experience Replay） + 优先采样（PER） + 目标网络（Target Network）",
           "表格型存储 + 蒙特卡洛多步回报采样 + 探索率退火"],
       _answer=1, feedback_hint="死亡铁三角的三大要素：函数逼近 + 自举（TD）+ 离线策略（Off-policy）。三者齐聚时投影算子可能膨胀导致发散。"),

    _q("q11", "open", "explanation", "【核心理论辨析】为什么同策略线性 TD 保证收敛，而离线策略线性 TD 会发散？",
       r"在 Zhao §8.2.4 与 §8.2.5 中详细证明了：在同策略（On-policy）下，线性半梯度 TD 保证以概率 1 收敛；"
       r"但一旦切换到离线策略（Off-policy），哪怕依然采用最简单的线性函数逼近，也可能彻底发散（如经典的 Baird 反例）。"
       r"请从 **加权范数 \(\|\cdot\|_D\)、正交投影算子 \(\Pi\) 的非扩张性（Non-expansion）、以及行为策略分布与目标策略分布的不匹配** 角度，"
       r"详细阐述导致这一收敛与发散分水岭的数学机理。",
       6, "从投影几何、压缩映射与分布失配的深层数学本质剖析算法稳定性边界。", DEADLY_SOURCE,
       rubric=[
           {"criterion": "指出在同策略下，投影权重矩阵 D 为目标策略 pi 的稳态分布，此时投影算子 Pi 在 D 范数下是非扩张的（||Pi v||_D <= ||v||_D），与 Bellman 压缩映射 T_pi（压缩因子 gamma < 1）复合后，复合算子 Pi * T_pi 依然是严格压缩映射，由 Banach 不动点定理必收敛。", "points": 3},
           {"criterion": "指出在离线策略下，数据采样分布由行为策略 beta 决定，投影算子是在 D_beta 范数下正交，但 Bellman 算子 T_pi 的压缩性是针对目标策略 pi 的。分布失配导致复合算子 Pi * T_pi 丧失压缩性质，范数可能被放大（算子谱半径 > 1），从而引发迭代发散。", "points": 3},
       ],
       model_answer=(
           "1. 同策略（On-policy）线性 TD 收敛机理：\n"
           "- 在同策略下，状态采样分布严格服从目标策略 pi 的平稳分布 d_pi。以对角矩阵 D = diag(d_pi) 定义的加权范数下，正交投影算子 Pi 具有非扩张性（Non-expansion），即对任意向量 v 均有 ||Pi v||_D <= ||v||_D。\n"
           "- 同时，Bellman 算子 T_pi 在该分布加权范数下是关于折扣因子 gamma 的严格压缩映射（||T_pi u - T_pi v||_D <= gamma ||u - v||_D）。\n"
           "- 两个非扩张算子的复合映射 Pi * T_pi 的压缩因子为 gamma < 1，依然是严格压缩映射。根据 Banach 不动点定理，迭代序列必以几何速率收敛到唯一不动点 Phi * w*。\n\n"
           "2. 离线策略（Off-policy）线性 TD 发散机理：\n"
           "- 离线策略中，经验轨迹由行为策略 beta 产生，采样分布由 d_beta 决定。因此正交投影算子 Pi 是在 D_beta 范数下定义的。\n"
           "- 然而，Bellman 期望算子 T_pi 依赖于目标策略 pi 的状态转移概率。由于 d_beta 与 d_pi 存在严重的分布失配（Distribution Mismatch），T_pi 在 D_beta 范数下不再保证具有严格压缩性。\n"
           "- 复合算子 Pi * T_pi 的诱导范数可能严格大于 1（算子矩阵的最大特征值模长大于 1），在连续的自举自反馈迭代中，误差向量被持续放大，导致权重参数 w 和价值估计值呈指数级无界爆炸（如 Baird 反例）。"
       )),

    # ----------------------------------------------------------------------- #
    # Stage 3: 动作价值逼近：Sarsa vs Q-learning 与控制 (§8.3 & EasyRL §6.2)
    # ----------------------------------------------------------------------- #
    _q("q12", "choice", "concept", r"从状态价值到动作价值：为什么无模型控制必须逼近 \(q(s, a)\)？",
       r"在纯粹的策略评估中逼近状态价值 \(\hat{v}(s, w)\) 就足够了，但在无模型（Model-free）策略改善与最优控制中，我们必须逼近动作价值 \(\hat{q}(s, a, w)\)。"
       r"其根本原因是什么？",
       4, "透彻理解无模型控制中动作价值函数对策略改进的关键支撑作用。", ACTION_VALUE_SOURCE,
       options=[
           r"因为状态价值函数在数学上无法定义均方误差损失函数。",
           r"在无模型环境中状态转移概率 \(P(s'|s, a)\) 完全未知，若仅知道状态价值 \(\hat{v}(s)\)，无法在不依赖模型的情况下通过 \(\arg\max_a \sum_{s'} P(s'|s,a)\hat{v}(s')\) 选取贪婪动作；而有了 \(\hat{q}(s,a)\)，无需转移概率即可直接通过 \(\arg\max_a \hat{q}(s,a)\) 决策。",
           "因为神经网络只能接受动作作为输入，无法直接输入状态。",
           "因为动作价值函数的参数量永远小于状态价值函数的参数量。"],
       _answer=1, feedback_hint="Model-free 的核心在于无法预知下一状态分布，必须通过 q(s,a) 直接评估当前执行某动作的长远回报以支持贪心选择。"),

    _q("q13", "choice", "concept", "线性 Sarsa(0) vs 线性 Q-learning 的收敛稳定性分野",
       r"在结合线性函数逼近时，Sarsa(0) 的更新目标为 \(r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w)\)，"
       r"而 Q-learning 的更新目标为 \(r_{t+1} + \gamma \max_a \hat{q}(s_{t+1}, a, w)\)。"
       r"关于二者在线性函数逼近下的性质对比，下列说法正确的是？",
       4, "掌握同策略与异策略在线性逼近框架下的收敛性差异与潜在发散风险。", ACTION_VALUE_SOURCE,
       options=[
           "线性 Q-learning 永远绝对收敛，因为 max 算子能自动平滑掉数值波动的误差。",
           r"线性 Sarsa(0) 是同策略算法（使用实际采样的下一个动作 \(a_{t+1}\)），避开了死亡铁三角中的 Off-policy 要素，在满足标准步长条件下保证收敛；而线性 Q-learning 由于引入了带 max 的异策略目标，可能出现发散。",
           "线性 Sarsa(0) 必须使用大容量经验回放池才能训练，而 Q-learning 不能使用回放池。",
           "二者的更新梯度推导完全相同，在任何情况下收敛轨迹均严格重合。"],
       _answer=1, feedback_hint="Sarsa 是 on-policy，避免了死亡铁三角的 off-policy 陷阱；而 Q-learning 取 max 属于 off-policy，在线性逼近下也有发散反例。"),

    _q("q14", "numeric", "calculation", "线性 Sarsa(0) 单步参数更新手算",
       r"设状态-动作特征向量为 2 维，当前权重向量为 \(w_0 = [0.5, 1.0]^T\)。"
       r"智能体在状态-动作对 \((s_t, a_t)\)（特征为 \(\phi(s_t, a_t) = [1.0, 1.0]^T\)）执行动作，获得即时奖励 \(r_{t+1} = 1.0\)；"
       r"根据当前策略选取了下一动作，得到下一对 \((s_{t+1}, a_{t+1})\)（特征为 \(\phi(s_{t+1}, a_{t+1}) = [0.0, 2.0]^T\)）。"
       r"折扣因子 \(\gamma = 0.8\)，学习率 \(\alpha = 0.2\)。"
       r"请按顺序计算更新后的一步权重向量 \(w_1 = [w_{1, 1}, w_{1, 2}]\)（如 `0.72, 1.22`）：",
       5, "亲手计算线性 Sarsa 动作价值预测、同策略自举目标与权重更新。", ACTION_VALUE_SOURCE,
       answer_type="list", _expected=[0.72, 1.22], tolerance=1e-3,
       feedback_hint="q_cur = [1,1][0.5,1]^T = 1.5; q_next = [0,2][0.5,1]^T = 2.0; y = 1.0 + 0.8*2.0 = 2.6; delta = 2.6 - 1.5 = 1.1; w1 = [0.5, 1.0]^T + 0.2*1.1*[1, 1]^T = [0.72, 1.22]^T。"),

    _q("q15", "numeric", "calculation", "线性 Q-learning 单步更新与 max 目标手算",
       r"紧接上一场景，若采用 Q-learning 规则：在下一状态 \(s_{t+1}\)，智能体有两个候选动作 \(a^{(1)}\) 与 \(a^{(2)}\)，"
       r"其特征分别为 \(\phi(s_{t+1}, a^{(1)}) = [0.0, 2.0]^T\) 与 \(\phi(s_{t+1}, a^{(2)}) = [5.0, 0.0]^T\)。"
       r"当前权重依然为 \(w_0 = [0.5, 1.0]^T\)，\(\gamma = 0.8\)，\(\alpha = 0.2\)。"
       r"即时奖励 \(r_{t+1} = 1.0\)，当前 \((s_t, a_t)\) 特征为 \([1.0, 1.0]^T\)。"
       r"请按顺序计算：[下一状态的最大预测 Q 值 \(\max_a \hat{q}(s_{t+1}, a)\), 更新后的权重向量 \(w_1\)]（三个数，如 `2.5, 0.8, 1.3`）：",
       5, "掌握 Q-learning 在多个候选动作中取 max 构造自举目标并更新权重的全过程。", ACTION_VALUE_SOURCE,
       answer_type="list", _expected=[2.5, 0.8, 1.3], tolerance=1e-3,
       feedback_hint="q(a1)=2.0, q(a2)=5.0*0.5=2.5 => max=2.5; y=1.0+0.8*2.5=3.0; delta=3.0-1.5=1.5; w1 = [0.5, 1.0]^T + 0.2*1.5*[1, 1]^T = [0.8, 1.3]^T。"),

    _q("q16", "code", "implementation", "实现线性 Q-learning 估值器与单步更新函数",
       r"实现函数 `linear_q_learning_update(w, phi_sa, reward, next_phi_list, gamma, alpha)`。\n"
       r"参数说明：\n"
       r"- `w`: 当前权重向量，1D NumPy 数组或列表，长度为 \(d\)；\n"
       r"- `phi_sa`: 当前状态-动作特征向量，长度为 \(d\)；\n"
       r"- `reward`: 即时奖励标量 \(r\)；\n"
       r"- `next_phi_list`: 包含下一状态下所有候选动作特征向量的二维数组/列表，形状为 `[num_actions, d]`；\n"
       r"- `gamma`: 折扣因子；\n"
       r"- `alpha`: 学习率。\n"
       r"计算逻辑：\n"
       r"1. 当前估值：\(q_{\text{cur}} = \phi(s, a)^T w\)；\n"
       r"2. 下一状态各动作估值并取最大：\(q_{\text{next\_max}} = \max_{a'} \phi(s', a')^T w\)；\n"
       r"3. TD 目标与误差：\(y = r + \gamma q_{\text{next\_max}}\)，\(\delta = y - q_{\text{cur}}\)；\n"
       r"4. 参数更新：\(w_{\text{new}} = w + \alpha \delta \phi(s, a)\)。\n"
       r"返回更新后的一维 NumPy 权重数组。",
       6, "严谨实现经典线性动作价值 Q-learning 矢量化单步更新。", ACTION_VALUE_SOURCE,
       entry_point="linear_q_learning_update",
       starter_code="""def linear_q_learning_update(w, phi_sa, reward, next_phi_list, gamma, alpha):
    # return: updated 1D numpy array w
    pass
""",
       public_tests=[
           {"name": "check_hand_calc_q15", "description": "检验第q15题手算算例，验证线性 Q-learning 权重更新。",
            "code": """import numpy as np
w = [0.5, 1.0]
phi_sa = [1.0, 1.0]
reward = 1.0
next_phi_list = [[0.0, 2.0], [5.0, 0.0]]
gamma = 0.8
alpha = 0.2
w_new = linear_q_learning_update(w, phi_sa, reward, next_phi_list, gamma, alpha)
assert np.allclose(w_new, [0.8, 1.3], atol=1e-5), f"Expected [0.8, 1.3], got {w_new}"
"""},
           {"name": "single_action_boundary", "description": "验证仅有单动作分支时的更新正确性。",
            "code": """import numpy as np
w = [1.0, 0.0]
phi_sa = [1.0, 0.0]
reward = 2.0
next_phi = [[1.0, 0.0]]
w_new = linear_q_learning_update(w, phi_sa, reward, next_phi, gamma=0.5, alpha=0.1)
# q_cur = 1.0, q_next = 1.0, y = 2.0 + 0.5*1.0 = 2.5, delta = 1.5, w_new = [1.0+0.15, 0.0] = [1.15, 0.0]
assert np.allclose(w_new, [1.15, 0.0], atol=1e-5)
"""},
       ],
       rubric=[
           {"criterion": "准确计算当前状态动作特征内积以及下一状态所有动作内积并选取最大值。", "points": 3},
           {"criterion": "使用半梯度更新公式正确更新权重并返回标准的一维 NumPy 数组。", "points": 3},
       ],
       _solution=_LINEAR_Q_UPDATE_SOLUTION),

    # ----------------------------------------------------------------------- #
    # Stage 4: 深度 Q 网络（DQN）两大基石与工程实现 (§8.4 & EasyRL Ch6)
    # ----------------------------------------------------------------------- #
    _q("q17", "choice", "concept", "深度强化学习的非稳态与相关性困境",
       r"当我们尝试直接用深度神经网络替代 Q 表格进行在线 Q-learning 时（如 2013 年 DQN 提出之前），网络几乎无法稳定训练。"
       r"造成训练剧烈震荡发散的两个根本原因是什么？",
       4, "深刻理解深度神经网络与强化学习相结合时的两大固有阻碍。", DQN_CORE_SOURCE,
       options=[
           "神经网络参数量太大导致显存溢出，且单精度浮点数无法支持 Bellman 方程的迭代。",
           "连续时间步采集的经验样本高度时序自相关（违背深度学习 SGD 的独立同分布 i.i.d. 假设），且同一个网络既预测 Q 值又计算 TD Target 导致严重的“动态移动靶”正反馈震荡。",
           "强化学习环境在底层物理引擎中不支持自动微分，反向传播无法传递梯度。",
           "梯度下降算法无法求解包含离散动作空间的优化问题。"],
       _answer=1, feedback_hint="样本时序强相关破坏了 i.i.d. 假设；自举目标随自身权重频繁变动形成了恶性正反馈移动靶。"),

    _q("q18", "choice", "concept", "经验回放池（Replay Buffer）的四大工程机制与容量权衡",
       r"DQN 引入了经验回放池（Replay Buffer）。关于经验回放池的工程设计与机理，下列分析错误的是？",
       4, "透彻理解经验回放池在消除时序相关性、提升数据利用率等方面的核心机理。", DQN_CORE_SOURCE,
       options=[
           "随机均匀采样打破了连续时间步样本之间的强自相关性，使 mini-batch 数据更符合 i.i.d. 分布。",
           "能够多次重复利用历史转移样本，显著提高了与环境交互采样数据的样本利用效率。",
           "回放池容量设置得越小越好（例如容量设为 32，等于 batch_size），这样可以保证每次只学最新鲜的一步数据。",
           "训练初期通常需要进行 Warm-up 预热，收集足够数量的随机探索经验后再启动网络反向传播。"],
       _answer=2, feedback_hint="回放池过小会导致数据退化为高度自相关的前后连续帧，网络迅速过拟合失稳；通常容量需设置为 10000 甚至更大。"),

    _q("q19", "choice", "concept", "目标网络（Target Network）与移动靶阻断",
       r"DQN 引入了独立的目标网络 \(Q(s, a; \theta^-)\)。在固定周期 \(C\) 内保持其权重 \(\theta^-\) 冻结不变，仅周期性从策略网络同步（\(\theta^- \leftarrow \theta\)）。"
       r"关于目标网络的工程价值，下列说法正确的是？",
       4, "掌握目标网络在缓解移动靶震荡与保证学习稳定性之间的核心工程逻辑。", DQN_CORE_SOURCE,
       options=[
           "目标网络必须在初始化后永久冻结，绝不允许进行任何更新。",
           "目标网络在一段固定时间窗口内为回归任务提供了一个相对静止的靶心，切断了参数更新对目标值的即时正反馈放大效应，大幅提升了训练稳定性。",
           "目标网络的更新频率越高越好，每步同步（C=1）效果最佳。",
           "目标网络只能用于连续动作环境，离散环境无法使用。"],
       _answer=1, feedback_hint="目标网络的核心价值在于在一个短窗口内提供固定的回归靶心，既不能步步震荡也不能过度陈旧。"),

    _q("q20", "choice", "concept", "致命代码 Bug 诊断：为什么终止状态必须有 (1 - dones) 掩码？",
       r"在 DQN 目标计算公式 \(y = r + (1 - \text{done}) \gamma \max_{a'} Q(s', a'; \theta^-)\) 中，`done` 标志用于标识回合终止（如 CartPole 杆子倒地或越界）。"
       r"如果某位工程师疏忽写成了 `target_q = rewards + self.gamma * max_next_q`（漏掉了 `1 - dones`），会导致什么灾难性后果？",
       4, "深刻理解马尔可夫决策过程中吸收终止状态的价值定义与代码防错。", DQN_CORE_SOURCE,
       options=[
           "小车倒地失败后，算法依然将假想的未来步数折现加到当前回报中，导致失败动作的 Q 值严重虚高，智能体彻底失去对倒地惩罚的敏感性而无法收敛。",
           "小车会永远处于静止状态，动作全被强制置为 0。",
           "没有任何影响，因为倒地后环境会重置，下一状态的 Q 值必然为 0。",
           "只影响代码运行速度，不会影响策略最终学会平衡。"],
       _answer=0, feedback_hint="终止态的未来真实价值严格为 0。漏乘 (1-done) 会向倒地终止转移错误注入虚假未来期望。"),

    _q("q21", "choice", "concept", "PyTorch 计算图实操：为什么 Target Q 必须用 torch.no_grad()？",
       r"在 `train_dqn_500epochs.py` 中，计算目标 Q 值时编写了如下代码："
       r"\n```python\nwith torch.no_grad():\n    max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)\n    target_q = rewards + (1 - dones) * self.gamma * max_next_q\n```"
       r"\n如果移除了 `with torch.no_grad():`（且未调用 `.detach()`），在训练中会发生什么？",
       4, "从深度学习计算图与半梯度理论结合的角度理解 no_grad 的必要性。", SEMIGRAD_SOURCE,
       options=[
           "反向传播会将梯度同时回传给目标项，算法在数学上退化为残差梯度法（引入双采样偏差），且历史计算图常驻显存引发内存/显存泄露。",
           "代码会因为变量类型不匹配而在第一步立即崩溃，抛出 TypeError。",
           "训练完全不受影响，因为 PyTorch 的优化器会自动识别并忽略目标网络的参数。",
           "网络会强制进行二阶牛顿法迭代，使得训练速度提高 10 倍。"],
       _answer=0, feedback_hint="不阻断目标计算图会导致梯度反传到 target_net 甚至历史图，破坏半梯度更新的单向性并导致显存泄漏。"),

    _q("q22", "numeric", "calculation", "CartPole 满分 500 步的理论折扣回报上限手算",
       r"CartPole-v1 环境中，只要小车未倾倒，每前进一步获得即时奖励 \(r = 1.0\)。"
       r"假设某优秀智能体从初始状态 \(t=0\) 开始一直稳定平衡至满分终止步数 \(T = 500\)。"
       r"在折扣因子 \(\gamma = 0.9\) 下，该轨迹在初始时刻 \(t=0\) 的无偏累计折扣回报"
       r" \(G_0 = \sum_{t=0}^{499} \gamma^t \cdot 1.0\) 为多少？"
       r"（已知 \(0.9^{500} \approx 1.32 \times 10^{-23} \approx 0\)，请计算精确到小数点后一位的数值，如 `10.0`）：",
       5, "从强化学习回报定义出发，掌握任务理论最大回报上界的计算，避免盲目调参。", VREP_SOURCE,
       answer_type="number", _expected=10.0, tolerance=1e-2,
       feedback_hint="等比数列求和 G0 = (1 - 0.9^500) / (1 - 0.9) = 1.0 / 0.1 = 10.0。"),

    _q("q23", "numeric", "calculation", "CartPole 单步 Bellman Target 与 TD Loss 手算",
       r"从 CartPole 经验池中抽取一条真实转移：小车从状态 \(s\) 采取向右推（\(a=1\)），获得奖励 \(r=1.0\)，"
       r"进入安全下一状态 \(s'\)（此时未倒地，`done = 0`），折扣因子 \(\gamma = 0.9\)。\n"
       r"已知：\n"
       r"- 当前策略网络对当前状态动作的估值为：\(Q(s, a=1; \theta) = 3.2\)；\n"
       r"- 目标网络在下一状态对两个动作的输出为：\(Q(s', \cdot; \theta^-) = [2.0, 4.0]\)（0:推左, 1:推右）。\n"
       r"请按顺序计算：[该样本的 TD 目标值 \(y\), 该样本的均方误差损失 \(\text{Loss} = (y - Q)^2\)]（如 `4.6, 1.96`）：",
       5, "掌握 DQN 核心 Bellman 目标与单步均方误差损失的完整数值推导。", SEMIGRAD_SOURCE,
       answer_type="list", _expected=[4.6, 1.96], tolerance=1e-3,
       feedback_hint="max Q(s')=4.0; y = 1.0 + 0.9 * 4.0 = 4.6; delta = 4.6 - 3.2 = 1.4; Loss = 1.4^2 = 1.96。"),

    _q("q24", "numeric", "calculation", "漏写 `(1 - done)` 掩码的数值偏差量化手算",
       r"小车在某一步严重倾角倒地，触发回合终止：即时奖励 \(r=1.0\)，`done = 1`，折扣因子 \(\gamma = 0.99\)。"
       r"此时目标网络对倒地状态 \(s'\) 预测的最大 Q 值为 \(\max_{a'} Q(s', a'; \theta^-) = 8.0\)。\n"
       r"请按顺序计算：\n"
       r"1. 带有正确 `(1 - done)` 截断的真实目标值 \(y_{\text{correct}}\)；\n"
       r"2. 错误漏乘 `(1 - done)` 时算出的虚假目标值 \(y_{\text{err}}\)；\n"
       r"3. 漏写此代码给该样本注入的额外 Target 误差值 \(|y_{\text{err}} - y_{\text{correct}}|\)。（如 `1.0, 8.92, 7.92`）：",
       5, "从直观数值上深刻体会漏写终止掩码对 Q 值学习造成的毁灭性偏差。", DQN_CORE_SOURCE,
       answer_type="list", _expected=[1.0, 8.92, 7.92], tolerance=1e-3,
       feedback_hint="y_correct = 1.0; y_err = 1.0 + 0.99 * 8.0 = 8.92; 误差差值 = 7.92。"),

    _q("q25", "numeric", "calculation", r"CartPole 训练中的 \(\epsilon\)-贪心按轮衰减手算",
       r"在 `train_dqn_500epochs.py` 中，智能体在每个 episode 结束后调用 `agent.decay_epsilon()` 执行指数衰减："
       r"\n\[ \epsilon_{k+1} = \max(\epsilon_{\min}, \epsilon_k \times \text{decay}) \]"
       r"\n设初始探索率 \(\epsilon_0 = 1.0\)，衰减率 \(\text{decay} = 0.90\)，探索下限 \(\epsilon_{\min} = 0.05\)。"
       r"请计算经历完整 3 个 episode 训练并衰减后，在第 4 个 episode 开始时的 \(\epsilon\) 值（精确到小数后三位，如 `0.729`）：",
       5, "掌握探索率退火规律的计算，理解前期充分探索与后期平稳收敛的转换节点。", DQN_CORE_SOURCE,
       answer_type="number", _expected=0.729, tolerance=1e-3,
       feedback_hint="eps1 = 1.0 * 0.9 = 0.9; eps2 = 0.9 * 0.9 = 0.81; eps3 = 0.81 * 0.9 = 0.729。"),

    _q("q26", "open", "explanation", "【代码审查实战】指出这段 DQN 训练循环中的 3 个致命 Bug",
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
       6, "具备工业级代码审查能力，精准识别强化学习特有的计算图与截断陷阱。", DQN_CORE_SOURCE,
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

    _q("q27", "open", "explanation", "【实验现象复盘】CartPole 训练中 Loss 上升而 Reward 也在上升，如何科学评估策略？",
       r"在 `train_dqn_500epochs.py` 训练曲线中，经常发现第 100 轮左右平均得分从 50 猛增到 400，但同时 MSE Loss 却从 0.05 剧烈上升到 0.8 以上。"
       r"请回答：\n"
       r"1. 为什么在强化学习中，TD Loss 上升并不代表模型在变差？（结合移动靶与分布漂移回答）\n"
       r"2. 实际工程中，为什么不能把 TD Loss 作为早停（Early Stopping）指标？我们应当如何严谨评估当前策略的真实水平？",
       6, "破除监督学习定势，掌握强化学习中动态目标与策略独立评估法则。", DQN_CORE_SOURCE,
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

    _q("q28", "code", "implementation", r"实现 \(\epsilon\)-贪心动作选择器",
       r"实现函数 `select_action(q_values, epsilon, num_actions, random_val=None)`。\n"
       r"参数说明：\n"
       r"- `q_values`: 当前状态下各离散动作的 Q 值数组（NumPy array 或列表）；\n"
       r"- `epsilon`: 探索概率 \(\epsilon\)；\n"
       r"- `num_actions`: 可用动作总数（如 CartPole 为 2）；\n"
       r"- `random_val`: 传入浮点数时（测试用），若 `random_val < epsilon` 则执行探索返回 `0`；为 `None` 时通过 `random.random()` 抽样。\n"
       r"逻辑要求：以概率 \(\epsilon\) 探索（返回随机动作索引），以概率 \(1 - \epsilon\) 贪心（返回 \(\arg\max\) 动作索引）。",
       6, "熟练掌握强化学习探索与利用基础控制算子的矢量化实现。", DQN_CORE_SOURCE,
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

    _q("q29", "code", "implementation", "实现带 `done` 掩码的 DQN 目标计算函数",
       r"实现函数 `compute_dqn_targets(rewards, next_q_max, dones, gamma)`。\n"
       r"参数说明：\n"
       r"- `rewards`: 批次即时奖励，shape 为 `[B]` 或 `[B, 1]`；\n"
       r"- `next_q_max`: 目标网络在下一状态的最大 Q 值，shape 为 `[B]` 或 `[B, 1]`；\n"
       r"- `dones`: 终止布尔标志（0 或 1），shape 为 `[B]` 或 `[B, 1]`；\n"
       r"- `gamma`: 折扣因子标量。\n"
       r"计算公式：\(y = r + (1 - \text{done}) \cdot \gamma \cdot \max_{a'} Q(s', a')\)。\n"
       r"返回 shape 为 `[B, 1]` 的二维列向量（NumPy array）。",
       6, "严密实现带终止截断掩码的标准 DQN 目标张量计算。", DQN_CORE_SOURCE,
       entry_point="compute_dqn_targets",
       starter_code="""def compute_dqn_targets(rewards, next_q_max, dones, gamma):
    # return: numpy array of shape [B, 1]
    pass
""",
       public_tests=[
           {"name": "check_hand_calc_q23", "description": "验证非终止转移的目标计算。",
            "code": """import numpy as np
y = compute_dqn_targets([1.0], [4.0], [0.0], 0.9)
assert np.isclose(y[0, 0], 4.6, atol=1e-5), f"Expected 4.6, got {y[0,0]}"
"""},
           {"name": "check_done_masking_q24", "description": "验证当 done=1 时未来价值严格截断为 0。",
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

    # ----------------------------------------------------------------------- #
    # Stage 5: DQN 进阶技巧：Double DQN、Dueling DQN、PER 与 Huber 损失 (EasyRL Ch7)
    # ----------------------------------------------------------------------- #
    _q("q30", "choice", "concept", "动作过估计偏差（Overestimation Bias）：Jensen 不等式与 max 算子的数学宿命",
       r"传统 Q-learning 与 DQN 普遍存在高估（Overestimation）现象。"
       r"设某状态下各动作的真实 Q 值均为 0，但网络预测带有零均值独立随机噪声 \(\epsilon_a \sim \mathcal{N}(0, \sigma^2)\)。"
       r"关于 \(\mathbb{E}[\max_a (Q(s, a) + \epsilon_a)]\) 与 \(\max_a \mathbb{E}[Q(s, a)]\) 的关系，下列数学分析正确的是？",
       4, "从数学期望与凸函数性质（Jensen 不等式）理解 Q 值高估的不可避免性。", DDQN_SOURCE,
       options=[
           r"由于 \(\max\) 是严格的凸函数（Convex Function），根据 Jensen 不等式，\(\mathbb{E}[\max_a X_a] \ge \max_a \mathbb{E}[X_a]\)，含噪估值的最大值期望必然大于等于真实最大值的期望，产生系统性正向高估偏差。",
           r"噪声是对称的（均值为 0），因此取最大值后的期望依然严格等于 0，绝不存在高估。",
           r"取最大值会导致严重的欠估计（Underestimation），预测值会越来越低。",
           r"只有当动作空间维度超过 1000 时才会出现高估，CartPole 仅有 2 个动作在数学上绝不会出现高估。"],
       _answer=0, feedback_hint="max 是凸函数，E[max(X)] >= max(E[X])；对含噪估计取最大值在数学期望上必然产生正向高估偏差。"),

    _q("q31", "choice", "concept", "Double DQN 的核心解耦哲学：两套网络如何分工？",
       r"Double DQN（Van Hasselt et al.）通过动作选择（Action Selection）与动作评估（Action Evaluation）的解耦，有效消除了 max 算子的高估偏差。"
       r"其计算下一状态目标值的公式是？",
       4, "熟记 Double DQN 选动作与评价值彻底解耦的公式与实现逻辑。", DDQN_SOURCE,
       options=[
           r"\(y = r + (1 - \text{done})\gamma \max_{a'} Q(s', a'; \theta^-)\)",
           r"\(a^* = \arg\max_{a'} Q(s', a'; \theta),\quad y = r + (1 - \text{done})\gamma Q(s', a^*; \theta^-)\)",
           r"\(a^* = \arg\max_{a'} Q(s', a'; \theta^-),\quad y = r + (1 - \text{done})\gamma Q(s', a^*; \theta)\)",
           r"\(y = r + (1 - \text{done})\gamma \frac{1}{|\mathcal{A}|} \sum_{a'} Q(s', a'; \theta^-)\)"],
       _answer=1, feedback_hint="Double DQN 用当前策略网络 theta 挑出最大动作索引，再用目标网络 theta^- 评估该动作的分值。"),

    _q("q32", "choice", "concept", "Dueling DQN 架构与可辨识性（Identifiability）：为什么要减去优势均值？",
       r"Dueling DQN 将网络拆解为状态价值分支 \(V(s)\) 与优势分支 \(A(s, a)\)。"
       r"若直接写为 \(Q(s, a) = V(s) + A(s, a)\)，存在不可辨识性（Identifiability）问题（给 \(V(s)\) 加常数 \(c\) 同时给所有 \(A(s, a)\) 减 \(c\)，\(Q\) 值不变）。"
       r"论文中实际采用的均值中心化公式为："
       r"\n\[ Q(s, a) = V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\Big) \]"
       r"\n关于采用该公式的核心原因与性质，下列说法正确的是？",
       4, "掌握 Dueling 架构解耦思想、均值中心化对唯一可辨识性的保证及其优化平滑性。", DUELING_SOURCE,
       options=[
           r"它强制让各动作输出 Q 值的平均值严格等于状态价值 \(V(s)\)，不仅解决了不可辨识性，而且相对于减去最大值（max），均值中心化使得优化过程更加平滑稳定。",
           r"它彻底删除了动作空间，使得网络只能输出单个标量。",
           r"减去均值后，动作优势的相对排序发生了翻转，原本最好的动作变成了最差的动作。",
           r"仅仅是为了让全连接层的梯度恰好恒等于 0。"],
       _answer=0, feedback_hint="均值中心化保证了 mean(Q) = V(s)，使 V 与 A 具有唯一解，且均值求导梯度的分散性比 max 更稳定。"),

    _q("q33", "choice", "concept", "优先经验回放（PER）与重要性采样校正",
       r"在优先经验回放（Prioritized Experience Replay, PER）中，样本被抽中的概率正比于其 TD 误差：\(P(i) \propto |\delta_i|^\alpha + \epsilon\)。"
       r"但非均匀采样改变了状态访问分布，引入了参数更新偏差。PER 是如何从数学上消除这一分布偏差的？",
       4, "理解非均匀采样引入的分布偏移以及重要性采样权重校正机制。", PER_HUBER_SOURCE,
       options=[
           r"在损失函数中对每个样本乘以重要性采样权重 \(w_i = (N \cdot P(i))^{-\beta}\)（并除以最大权重归一化），且让退火指数 \(\beta\) 随训练进程逐渐退火至 1.0。",
           r"将回放池容量强制缩小到 10 个样本以内以减少偏差积累。",
           r"丢弃所有 TD 误差小于 1.0 的样本，只保留大误差样本。",
           r"直接将策略网络的学习率放大 100 倍。"],
       _answer=0, feedback_hint="重要性采样权重 w_i = (N * P(i))^{-beta} 用以抵消频繁采样大 TD 误差样本带来的梯度偏倚。"),

    _q("q34", "choice", "concept", "损失函数的工程选择：为什么 DQN 常用 Huber 损失（Smooth L1）替代 MSE？",
       r"在深度强化学习训练中，工程实现常将均方误差（MSE）损失替换为 Huber 损失（或 Smooth L1 Loss）："
       r"\n\[ \mathcal{L}(\delta) = \begin{cases} \frac{1}{2} \delta^2, & |\delta| \le 1 \\ |\delta| - \frac{1}{2}, & |\delta| > 1 \end{cases} \]"
       r"\n其核心工程收益是什么？",
       4, "掌握工业界针对强化学习大 TD 误差采取的稳健损失设计与梯度限制。", PER_HUBER_SOURCE,
       options=[
           r"当 TD 误差 \(\delta\) 很大时，MSE 损失的梯度随误差线性暴涨（\(\nabla \propto \delta\)），偶发异常大误差样本极易引发梯度爆炸击穿网络；Huber 损失在大误差区间将梯度截断为常数 \(\pm 1\)，极大增强了训练鲁棒性。",
           r"Huber 损失在所有点都是二阶连续可微的，因此收敛速度比 MSE 快 100 倍。",
           r"Huber 损失能够直接消除强化学习中的非稳态移动靶现象。",
           r"Huber 损失在反向传播时不需要计算偏导数。"],
       _answer=0, feedback_hint="Huber 损失小误差用 L2 平滑优化，大误差用 L1 将梯度幅值锁定在 1，防止异常 TD 误差导致梯度爆炸。"),

    _q("q35", "numeric", "calculation", "Double DQN vs 标准 DQN 目标值实战对比手算",
       r"在 CartPole 下一状态 \(s'\) 中，动作空间为 \(\{0:\text{推左}, 1:\text{推右}\}\)，即时奖励 \(r=1.0\)，\(\gamma = 0.9\)，`done = 0`。\n"
       r"各网络输出如下：\n"
       r"- 当前策略网络（Policy Net）：\(Q(s', \cdot; \theta) = [3.5, 4.2]\)（策略网络倾向推右）；\n"
       r"- 目标网络（Target Net）：\(Q(s', \cdot; \theta^-) = [4.8, 3.8]\)（推左被偶发高估为 4.8）。\n"
       r"请按顺序计算：[标准 DQN 计算出的目标值 \(y_{\text{DQN}}\), Double DQN 计算出的目标值 \(y_{\text{DDQN}}\)]（如 `5.32, 4.42`）：",
       5, "动手验证 Double DQN 如何避免盲目贪婪抓取高估动作，计算解耦目标。", DDQN_SOURCE,
       answer_type="list", _expected=[5.32, 4.42], tolerance=1e-3,
       feedback_hint="DQN: max=4.8, y=1+0.9*4.8=5.32; DDQN: argmax=1, target net evaluate action 1=3.8, y=1+0.9*3.8=4.42。"),

    _q("q36", "numeric", "calculation", "Dueling DQN 均值中心化前向张量数值还原手算",
       r"在小车接近竖直的状态 \(s\) 下，Dueling DQN 网络的双头分支输出如下：\n"
       r"- 标量状态价值流：\(V(s) = 15.0\)；\n"
       r"- 未中心化的优势向量：\(A(s, \cdot) = [2.0, 4.0]\)（对应推左、推右两个动作）。\n"
       r"按照均值中心化前向传播公式：\(Q(s, a) = V(s) + \big(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\big)\)。"
       r"请计算最终输出的两个动作的 Q 值 \([Q(s, a=0), Q(s, a=1)]\)（如 `14.0, 16.0`）：",
       5, "掌握 Dueling 架构均值中心化的计算逻辑，验证其相对优势保持不变。", DUELING_SOURCE,
       answer_type="list", _expected=[14.0, 16.0], tolerance=1e-3,
       feedback_hint="mean(A) = (2 + 4)/2 = 3.0; 中心化优势 = [-1.0, 1.0]; Q = 15.0 + [-1, 1] = [14.0, 16.0]。"),

    _q("q37", "open", "explanation", "【架构演进剖析】从 Vanilla DQN 到 Double / Dueling / PER 的工程精髓",
       r"在实际开发中，从标准 DQN 升级到 Double DQN、Dueling DQN 与 PER（优先经验回放）是强化学习工程的核心进化线。\n"
       r"1. 为什么说 Double DQN 是“极高性价比”的改造？（解决了什么痛点？改动了几行代码？）\n"
       r"2. Dueling DQN 将网络拆解为 \(V(s)\) 与 \(A(s, a)\)，为什么在类似 CartPole 这样的大多数状态下，能大幅加速学习？\n"
       r"3. PER 优先抽样大 TD 误差样本，直觉上非常高效，但为什么在反向传播前必须乘以重要性采样（Importance Sampling）权重？",
       6, "透彻理解三代经典 DQN 优化技术的工程痛点、改造逻辑与理论防线。", DDQN_SOURCE,
       rubric=[
           {"criterion": "说明 Double DQN 仅需两三行代码将 max 改为利用当前网络 argmax 寻优、目标网络打分，以极低改造成本根除了 Q 值虚高问题。", "points": 2},
           {"criterion": "说明 Dueling DQN 能够独立学好状态本身的固有价值 V(s)，无需对每个细微动作重复采样评估，极大提升了动作对状态影响微弱时的样本效率。", "points": 2},
           {"criterion": "说明非均匀优先采样改变了样本分布产生分布偏移（给大误差样本过大梯度），必须用重要性采样权重 w_i 抵消更新偏倚，确保收敛到真实期望。", "points": 2},
       ],
       model_answer=(
           "1. Double DQN 的极高性价比：\n"
           "- 痛点与代价：标准 DQN 因 max 算子的凸性导致严重的 Q 值虚高与盲目乐观。Double DQN 几乎零成本改造，仅需修改 2 行代码（用 policy_net.argmax(dim=1) 找动作索引，再用 target_net.gather 提取估值），就彻底切断了正向高估的恶性循环。\n\n"
           "2. Dueling DQN 加速学习的原理：\n"
           "- 在很多状态下（例如小车非常端正地处于轨道中央），无论选‘推左’还是‘推右’，小车的存活价值其实相差无几，决定当前回报的主要因素是状态本身的安全程度 V(s)。\n"
           "- 传统 DQN 必须对推左和推右分别更新才能学好各自的 Q 值；而 Dueling 架构将 V(s) 独立提取出来，每经历一个样本，V(s) 就会得到更新，使得所有动作的基准价值同时受益，极大节省了在无关紧要状态下探索不同动作的样本量。\n\n"
           "3. PER 必须结合重要性采样的原因：\n"
           "- 优先经验回放按 TD 误差大小改变了采样分布，使得高误差样本被抽中的概率远大于其在真实环境中的出现频率。\n"
           "- 如果直接用该非均匀样本做 SGD，更新期望将严重偏向高误差样本，导致梯度估计出现系统性偏差（相当于改变了期望中的平稳分布）。通过引入重要性采样权重 w_i = (N * P(i))^{-beta}，在反向传播时将高频采样的样本梯度相应压低，确保了参数更新方向依然是对真实期望梯度的无偏或渐进无偏估计。"
       )),

    _q("q38", "code", "implementation", "实现 Double DQN 交叉解耦目标计算函数",
       r"实现函数 `compute_ddqn_targets(rewards, q_eval_next, q_target_next, dones, gamma)`。\n"
       r"参数说明：\n"
       r"- `rewards`: 批次即时奖励，shape `[B]` 或 `[B, 1]`；\n"
       r"- `q_eval_next`: 当前策略网络在下一状态输出的各动作估值，shape `[B, num_actions]`；\n"
       r"- `q_target_next`: 目标网络在下一状态输出的各动作估值，shape `[B, num_actions]`；\n"
       r"- `dones`: 终止标志批次，shape `[B]` 或 `[B, 1]`；\n"
       r"- `gamma`: 折扣因子标量。\n"
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
           {"name": "check_hand_calc_q35", "description": "检验第q35题手算算例，验证 DDQN 消除高估。",
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

    _q("q39", "code", "implementation", "实现 Dueling DQN 均值中心化前向传播",
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
           {"name": "check_hand_calc_q36", "description": "检验第q36题手算数值，验证中心化优势与价值叠加。",
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

    _q("q40", "code", "implementation", "实现目标网络 Polyak 软更新函数",
       r"实现函数 `soft_update(target_weights, policy_weights, tau)`。\n"
       r"参数说明：\n"
       r"- `target_weights`: 目标网络参数数组（NumPy array 或列表）；\n"
       r"- `policy_weights`: 当前策略网络参数数组；\n"
       r"- `tau`: 平滑更新系数（浮点数，如 0.05）；\n"
       r"更新公式：\n"
       r"\[ \theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^- \]\n"
       r"返回更新后的目标参数 NumPy 数组副本。",
       6, "实现连续平滑跟踪目标网络更新算子。", DQN_CORE_SOURCE,
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
