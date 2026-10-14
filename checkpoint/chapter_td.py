"""Temporal-difference methods: TD(0), Sarsa, Q-learning, n-step, unified view."""

import math

import bank

CHAPTER_ID = "temporal-difference"
REFERENCE = (
    "Zhao《Mathematical Foundations of Reinforcement Learning》第7章"
    "（印刷页125–150 / PDF页138–163）；"
    "EasyRL v1.0.6 §3.3.2–3.3.3、§3.4.1–3.4.3"
    "（印刷页54–65 / PDF页62–73）"
)
CONVENTIONS = (
    "时间步 t 从 0 开始；v_0 / q_0 是初值，完成一次更新得到 v_1 / q_1。"
    "γ 为折扣因子，α 为学习率。"
    "代码题保证输入合法、维度匹配；不考类型校验或异常处理。"
    "短轨迹和 Q 表格数据是教学构造，不是真实训练记录。"
    "Zhao 的 TD error 定义为 δ_t = v_t(s_t) − (r_{t+1}+γv_t(s_{t+1}))，"
    "EasyRL 取相反符号 r+γV(s')−V(s)；更新方向一致，只是符号约定不同。"
)
PRESENTATION = {
    "eyebrow": "TEMPORAL-DIFFERENCE / 双教材",
    "intro": (
        "从 TD(0) 状态值估计到 Sarsa、Q-learning 和 n-step 统一视角，"
        "理解自举如何将 Bellman 方程变成增量随机近似。"
    ),
    "field_notes": [
        {"label": "01 / TD(0)",
         "formula": "v(s) ← v(s)+α[r+γv(s')−v(s)]",
         "tex": r"v_{t+1}(s_t)=v_t(s_t)+\alpha_t\big[r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)\big]",
         "description": "TD target = r+γv(s')，只更新当前访问状态的估计值。"},
        {"label": "02 / SARSA vs Q",
         "formula": "Sarsa: r+γq(s',a')  Q: r+γmax_a q(s',a)",
         "tex": r"\bar q_t=\begin{cases}r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}) & \text{Sarsa}\\r_{t+1}+\gamma\max_a q_t(s_{t+1},a) & \text{Q-learning}\end{cases}",
         "description": "Sarsa 的目标含实际下一动作，on-policy；Q-learning 取 max，off-policy。"},
        {"label": "03 / UNIFIED",
         "formula": "q ← q − α(q − q̄)",
         "tex": r"q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t\big[q_t(s_t,a_t)-\bar q_t\big]",
         "description": "n=1 退化 Sarsa，n→∞ 退化 MC，Q-learning 换 target 为 max。"},
    ],
    "note_footer": (
        "Zhao 以随机近似为骨架推导所有 TD 算法；"
        "EasyRL 用巴甫洛夫条件反射类比和 Q 表格直觉补充理解。"
    ),
}


def _source(zhao, printed, pdf, easy, easy_printed, easy_pdf, note=""):
    return {"text": (
        f"Zhao §{zhao}，印刷页 {printed} / PDF 页 {pdf}；"
        f"EasyRL v1.0.6 §{easy}，印刷页 {easy_printed} / PDF 页 {easy_pdf}。"
        + note
    )}


TD_SOURCE = _source("7.1", "126–132", "139–145",
                    "3.3.2", "54–56", "62–64",
                    "Zhao 给出 RM 推导和收敛定理；EasyRL 用条件反射类比提供直觉。")
SARSA_SOURCE = _source("7.2", "133–137", "146–150",
                       "3.4.1", "61", "69",
                       "两书 Sarsa 更新公式一致；Zhao 额外给出收敛定理 7.2。")
Q_SOURCE = _source("7.4", "140–144", "153–157",
                   "3.4.2", "62–64", "70–72",
                   "Q-learning 的 TD target 用 max；Zhao 证明它求解 Bellman 最优方程。")
NSTEP_SOURCE = _source("7.3、7.5", "138–139、145–147", "151–152、158–160",
                       "3.3.3", "57–58", "65–66",
                       "n-step Sarsa 统一 1-step Sarsa 和 MC；Zhao §7.5 给出统一公式 (7.20)。")
ONOFF_SOURCE = _source("7.4.2", "141–143", "154–156",
                       "3.4.3", "65", "73",
                       "Q-learning off-policy 因目标策略是 greedy，不依赖行为策略选取的 a_{t+1}。")
COMP_SOURCE = _source("7.1.2、7.5", "128–130、145–148", "141–143、158–161",
                      "3.3.2–3.3.3", "54–58", "62–66",
                      "DP bootstraps 不采样，MC 不 bootstraps 采样，TD 既 bootstraps 又采样。")


def _q(qid, qtype, category, title, prompt, score, goal, source, **extra):
    if qtype == "numeric" and extra.get("answer_type") == "list":
        extra["answer_count"] = len(extra["_expected"])
    return {
        "id": qid, "type": qtype, "category": category,
        "title": title, "prompt": prompt, "max_score": score,
        "grading_mode": {"choice": "auto", "numeric": "auto",
                         "open": "self", "code": "browser"}[qtype],
        "learning_goal": goal, "source_note": source,
        "review_hint": source["text"], **extra,
    }


_TD_SOLUTION = """\
def td_update(v, transition, alpha, gamma):
    s, r, s_next = transition
    new_v = list(v)
    target = r + gamma * v[s_next]
    new_v[s] = v[s] + alpha * (target - v[s])
    return new_v
"""

_SARSA_SOLUTION = """\
def sarsa_update(q, transition, alpha, gamma):
    s, a, r, s_next, a_next = transition
    new_q = [row[:] for row in q]
    target = r + gamma * q[s_next][a_next]
    new_q[s][a] = q[s][a] + alpha * (target - q[s][a])
    return new_q
"""

_QLEARNING_SOLUTION = """\
def q_learning_update(q, transition, alpha, gamma):
    s, a, r, s_next = transition
    new_q = [row[:] for row in q]
    target = r + gamma * max(q[s_next])
    new_q[s][a] = q[s][a] + alpha * (target - q[s][a])
    return new_q
"""

_EPS_GREEDY_SOLUTION = """\
def epsilon_greedy_probs(q_values, epsilon):
    n = len(q_values)
    greedy = max(range(n), key=lambda i: q_values[i])
    probs = [epsilon / n] * n
    probs[greedy] += 1 - epsilon
    return probs
"""


QUESTIONS = [
    _q("q01", "choice", "concept",
       "TD 与 MC 最本质的区别是什么？",
       r"TD(0) 更新为 \(v_{t+1}(s_t)=v_t(s_t)+\alpha_t[r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)]\)，"
       r"MC 更新为 \(v_{t+1}(s_t)=v_t(s_t)+\alpha_t[G_t-v_t(s_t)]\)。"
       r"两者都是无模型方法，但更新方式不同。下列哪项准确描述了本质区别？",
       4, "区分自举 (bootstrapping) 与完整回报采样。", TD_SOURCE,
       options=[
           r"TD 用当前估计值 \(v_t(s_{t+1})\) 自举，只需一步转移；MC 等待整条轨迹结束后用完整回报 \(G_t\) 更新。",
           "TD 需要系统模型，MC 不需要。",
           "TD 只能估计状态值，MC 只能估计动作值。",
           r"TD 更新所有状态，MC 只更新访问过的状态。",
       ],
       _answer=0,
       feedback_hint="关键在 bootstrapping：TD 用估计值估计，MC 用完整回报。"),
    _q("q02", "choice", "concept",
       "TD(0) 是哪个方程的随机近似？",
       r"Zhao 指出 TD(0) 可由 Robbins-Monro 算法求解 Bellman 期望方程"
       r" \(v^\pi(s)=\mathbb E[R_{t+1}+\gamma v^\pi(S_{t+1})\mid S_t=s]\) 得到。"
       r"下面哪个说法正确？",
       4, "将 TD(0) 与 Bellman 期望方程的随机求根联系起来。", TD_SOURCE,
       options=[
           "TD(0) 求解 Bellman 最优方程。",
               r"TD(0) 求解 Bellman 期望方程（策略 \(\pi\) 的状态值方程）。",
           "TD(0) 求解样本均值方程。",
           "TD(0) 求解梯度下降方程。",
       ],
       _answer=1,
       feedback_hint="TD(0) 估计给定策略的 v^π，对应 Bellman 期望方程；Q-learning 才求解最优方程。"),
    _q("q03", "choice", "concept",
       "DP、MC、TD 在自举和采样上的区别",
       r"EasyRL §3.3.3 用一张表对比 DP、MC、TD。"
       r"自举 (bootstrapping) 指用当前估计值更新，采样 (sampling) 指用经验样本而非全宽度回撤。"
       r"下面哪项正确？",
       4, "理解 DP/MC/TD 在自举和采样维度的分类。", COMP_SOURCE,
       options=[
           "DP 自举不采样，MC 不自举采样，TD 既自举又采样。",
           "DP 不自举不采样，MC 自举采样，TD 自举采样。",
           "三者都同时自举和采样。",
           "只有 MC 自举，只有 DP 采样。",
       ],
       _answer=0,
       feedback_hint="DP 用模型做全宽度回撤（自举、不采样）；MC 用完整回报（不自举、采样）；TD 两者都做。"),
    _q("q04", "choice", "concept",
       "MC 与 TD 的偏差和方差",
       r"关于 MC 和 TD(0) 的偏差 (bias) 和方差 (variance)："
       r"MC 用完整回报 \(G_t\)，TD(0) 用 \(r_{t+1}+\gamma v_t(s_{t+1})\)。"
       r"下面哪项正确？",
       4, "理解 bootstrapping 引入偏差但降低方差。", TD_SOURCE,
       options=[
           "TD(0) 无偏、方差大；MC 有偏、方差小。",
           r"MC 无偏（对终止回合）但方差大；TD(0) 有偏（依赖初始估计自举）但方差小。",
           "两者都无偏。",
           "两者都有偏且方差相同。",
       ],
       _answer=1,
       feedback_hint="MC 用真实回报无偏但目标随机性强方差大；TD 的 target 含估计值故有偏，但只依赖一步随机性方差小。"),
    _q("q05", "choice", "concept",
       "Sarsa 每次更新需要什么数据？",
       r"Sarsa 更新为 \(q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t[r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})-q_t(s_t,a_t)]\)。"
       r"每次迭代需要哪些经验数据？",
       4, "区分 on-policy 所需数据与 off-policy。", SARSA_SOURCE,
       options=[
           r"只需要 \((r_{t+1}, s_{t+1})\)。",
           r"需要 \((r_{t+1}, s_{t+1}, a_{t+1})\)——必须知道下一步实际采取的动作。",
           r"只需要 \((s_t, a_t)\)。",
           r"需要完整轨迹的回报 \(G_t\)。",
       ],
       _answer=1,
       feedback_hint="Sarsa 的 TD target 含 q(s',a')，a' 是行为策略实际选取的下一动作，所以 on-policy。"),
    _q("q06", "choice", "concept",
       "Q-learning 的 TD target 和求解的方程",
       r"Q-learning 更新为 \(q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t[r_{t+1}+\gamma\max_a q_t(s_{t+1},a)-q_t(s_t,a_t)]\)。"
       r"它的 TD target 是什么，求解什么方程？",
       4, "区分 Bellman 期望方程与最优方程。", Q_SOURCE,
       options=[
           r"target \(=r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})\)，求解 Bellman 期望方程。",
           r"target \(=r_{t+1}+\gamma\max_a q_t(s_{t+1},a)\)，求解 Bellman 最优方程。",
           r"target \(=r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})\)，求解 Bellman 最优方程。",
           r"target \(=r_{t+1}+\gamma\max_a q_t(s_{t+1},a)\)，求解 Bellman 期望方程。",
       ],
       _answer=1,
       feedback_hint="max 对应 Bellman 最优方程 q*(s,a)=E[R+γmax_a' q*(S',a')]；Sarsa 的 target 不含 max。"),
    _q("q07", "choice", "concept",
       "n-step Sarsa 如何统一 Sarsa 和 MC？",
       r"Zhao §7.3 的 n-step Sarsa 用 n 步回报"
       r" \(\bar q_t=\sum_{i=1}^{n}\gamma^{i-1}r_{t+i}+\gamma^n q_t(s_{t+n},a_{t+n})\)。"
       r"当 \(n=1\) 和 \(n\to\infty\) 时分别退化为什么？",
       4, "理解 n-step 的两端特例。", NSTEP_SOURCE,
       options=[
           r"\(n=1\) 是 Q-learning，\(n\to\infty\) 是 MC。",
           r"\(n=1\) 是 Sarsa（1-step），\(n\to\infty\) 是 MC。",
           r"\(n=1\) 是 TD(0) 状态值估计，\(n\to\infty\) 是 DP。",
           r"\(n=0\) 是 Sarsa，\(n=1\) 是 MC。",
       ],
       _answer=1,
       feedback_hint="n=1 时 target=r+γq(s',a') 即 Sarsa；n→∞ 时尾部 q 项消失，target=G_t 即 MC。"),
    _q("q08", "choice", "concept",
       "为什么 Q-learning 是 off-policy？",
       r"Sarsa 的 TD target 用 \(q_t(s_{t+1},a_{t+1})\)，其中 \(a_{t+1}\) 是行为策略实际采取的动作。"
       r"Q-learning 的 TD target 用 \(\max_a q_t(s_{t+1},a)\)。"
       r"为什么 Q-learning 是 off-policy 而 Sarsa 是 on-policy？",
       4, "从 TD target 的结构理解 on/off-policy 的根源。", ONOFF_SOURCE,
       options=[
           r"Q-learning 的目标策略是 greedy（取 max），不依赖行为策略实际选取的 \(a_{t+1}\)；"
           r"Sarsa 的 target 含行为策略实际选取的 \(a_{t+1}\)，目标策略=行为策略。",
           "Q-learning 用了 max 操作所以是 off-policy，与目标策略无关。",
           "Sarsa 用了 ε-greedy 所以是 off-policy。",
           "两者都是 off-policy，因为都用经验样本。",
       ],
       _answer=0,
       feedback_hint="关键不在用了 max，而在 target 是否依赖行为策略实际选取的下一动作。"),
    _q("q09", "numeric", "calculation",
       "手动做一次 TD(0) 状态值更新",
       r"给定 \(v_0(s)=2.0\)，\(v_0(s')=5.0\)，一次转移 \((s,\, r=3.0,\, s')\)，"
       r"\(\gamma=0.9\)，\(\alpha=0.1\)。"
       r"求更新后的 \(v_1(s)\)。结果保留至少 4 位小数。",
       5, "用 TD target 和增量更新公式手算。", TD_SOURCE,
       answer_type="number", tolerance=1e-6, _expected=2.55,
       feedback_hint="TD target = r+γv(s')=3+0.9×5=7.5；v_1=v_0+α(target−v_0)=2+0.1×5.5=2.55。"),
    _q("q10", "numeric", "calculation",
       "手动做一次 Sarsa 更新",
       r"给定 \(q_0(s,a)=1.0\)，\(q_0(s',a')=4.0\)，"
       r"转移 \((s,a,\, r=2.0,\, s',a')\)，\(\gamma=0.8\)，\(\alpha=0.5\)。"
       r"求更新后的 \(q_1(s,a)\)。",
       5, "用 Sarsa 的 TD target 手算增量更新。", SARSA_SOURCE,
       answer_type="number", tolerance=1e-6, _expected=3.1,
       feedback_hint="target = r+γq(s',a')=2+0.8×4=5.2；q_1=1+0.5×(5.2−1)=3.1。"),
    _q("q11", "numeric", "calculation",
       "手动做一次 Q-learning 更新",
       r"给定 \(q_0(s,a)=3.0\)，\(s'\) 有两个动作："
       r"\(q_0(s',a_1)=6.0\)，\(q_0(s',a_2)=4.0\)。"
       r"转移 \((s,a,\, r=1.0,\, s')\)，\(\gamma=0.5\)，\(\alpha=0.2\)。"
       r"求更新后的 \(q_1(s,a)\)。",
       5, "用 max 选 target 并手算增量更新。", Q_SOURCE,
       answer_type="number", tolerance=1e-6, _expected=3.2,
       feedback_hint="target = r+γmax q(s',·)=1+0.5×6=4；q_1=3+0.2×(4−3)=3.2。"),
    _q("q12", "numeric", "calculation",
       "计算 n-step TD 回报",
       r"已知 \(r_1=1.0\)，\(r_2=2.0\)，\(v(s_2)=5.0\)，\(\gamma=0.9\)，\(n=2\)。"
       r"2-step TD target 为 \(r_1+\gamma r_2+\gamma^2 v(s_2)\)。"
       r"求该 target 的值。",
       5, "理解 n-step 回报的折扣累加形式。", NSTEP_SOURCE,
       answer_type="number", tolerance=1e-6, _expected=6.85,
       feedback_hint="1+0.9×2+0.81×5=1+1.8+4.05=6.85。"),
    _q("q13", "numeric", "calculation",
       "ε-greedy 下的动作概率",
       r"3 个动作，\(Q(s)=[5,\,3,\,2]\)，\(\epsilon=0.3\)。"
       r"最优动作为 \(a_1\)（\(Q=5\)）。"
       r"ε-greedy 策略下选择 \(a_1\) 和 \(a_2\)（\(Q=3\)）的概率各是多少？"
       r"输入两个数，用逗号分隔，依次为 \(P(a_1)\) 和 \(P(a_2)\)。",
       5, "计算 ε-greedy 的完整概率分配。", SARSA_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[0.8, 0.1],
       feedback_hint="greedy 动作概率=(1−ε)+ε/|A|=0.7+0.1=0.8；非贪心=ε/|A|=0.1。"),
    _q("q14", "open", "explanation",
       "为什么 TD error 在 v = v^π 时期望为零？",
       r"Zhao 定义 TD error 为 \(\delta_t=v_t(s_t)-(r_{t+1}+\gamma v_t(s_{t+1}))\)。"
       r"当 \(v_t=v^\pi\) 时，证明 \(\mathbb E[\delta_t\mid S_t=s]=0\)，"
       r"并解释这个零期望的物理含义——它反映的是"
       r"\(t\) 与 \(t+1\) 两步之间的差异，还是估计值与真值之间的差异？",
       5, "用 Bellman 方程推导 TD error 的零期望性质。", TD_SOURCE,
       rubric=[
           {"criterion": r"写出 δ=v^π(s)−(R_{t+1}+γv^π(S_{t+1}))。", "points": 2},
           {"criterion": r"用 Bellman 期望方程 E[R_{t+1}+γv^π(S_{t+1})|S_t=s]=v^π(s) 得 E[δ|S_t=s]=0。", "points": 2},
           {"criterion": "说明零期望反映的是估计值与真值之差，而不仅是两步时间差。", "points": 1},
       ],
       model_answer=(
           r"当 v_t=v^π 时，δ_t=v^π(s_t)−(r_{t+1}+γv^π(s_{t+1}))。"
           r"取条件期望：E[δ_t|S_t=s]=v^π(s)−E[R_{t+1}+γv^π(S_{t+1})|S_t=s]。"
           r"由 Bellman 期望方程 (7.4)，E[R_{t+1}+γv^π(S_{t+1})|S_t=s]=v^π(s)，"
           r"因此 E[δ_t|S_t=s]=0。"
           "这说明 TD error 不只是两个时间步之间的差异，"
           "更本质地反映了当前估计值与真实状态值之间的偏差。"
           "估计准确时期望为零，估计偏高时期望为正、偏低时为负，"
           "算法据此方向修正。"),
       ),
    _q("q15", "open", "explanation",
       "对比 Sarsa 和 Q-learning 的更新公式、target 和 on/off-policy",
       r"请分别写出 Sarsa 和 Q-learning 的更新公式，指出它们的 TD target 各是什么、"
       r"每次更新需要哪些数据、以及为什么一个是 on-policy 而另一个是 off-policy。"
       r"结合 Zhao §7.4.2 或 EasyRL §3.4.3 的论述说明。",
       5, "系统对比两个核心 TD 算法的结构和策略属性。", ONOFF_SOURCE,
       rubric=[
           {"criterion": "写出 Sarsa 和 Q-learning 的更新公式。", "points": 2},
           {"criterion": r"target：Sarsa 用 r+γq(s',a')，Q-learning 用 r+γmax_a q(s',a)。", "points": 1},
           {"criterion": r"数据：Sarsa 需 (r,s',a')，Q-learning 只需 (r,s')。", "points": 1},
           {"criterion": "on/off：Sarsa target 含行为策略实际选取的 a'，故 on-policy；Q-learning target 用 max（greedy 目标策略），不依赖行为策略的 a'，故 off-policy。", "points": 1},
       ],
       model_answer=(
           r"Sarsa：q(s,a)←q(s,a)+α[r+γq(s',a')−q(s,a)]。"
           r"target = r+γq(s',a')，需要 (r,s',a')。"
           r"因为 a' 是行为策略实际选取的下一动作，"
           r"target 依赖于行为策略，目标策略=行为策略，故 on-policy。"
           "\n"
           r"Q-learning：q(s,a)←q(s,a)+α[r+γmax_a q(s',a)−q(s,a)]。"
           r"target = r+γmax_a q(s',a)，只需 (r,s')。"
           r"target 用 max（greedy），不依赖行为策略实际选取的 a'，"
           r"目标策略（greedy）≠行为策略（如 ε-greedy），故 off-policy。"),
       ),
    _q("q16", "open", "explanation",
       "用统一视角解释 n-step Sarsa 如何统一 Sarsa 和 MC",
       r"Zhao §7.5 给出统一公式 \(q_{t+1}=q_t-\alpha_t(q_t-\bar q_t)\)。"
       r"请写出 n-step Sarsa 的 \(\bar q_t\) 表达式，"
       r"说明 \(n=1\) 时退化为 Sarsa、\(n\to\infty\) 时退化为 MC，"
       r"并指出 Q-learning 如何在同一框架下取不同的 \(\bar q_t\)。",
       5, "理解统一框架的 target 如何涵盖 Sarsa、MC 和 Q-learning。", NSTEP_SOURCE,
       rubric=[
           {"criterion": r"写出 \bar q_t=\sum_{i=1}^{n}\gamma^{i-1}r_{t+i}+\gamma^n q_t(s_{t+n},a_{t+n})。", "points": 2},
           {"criterion": r"n=1：\bar q_t=r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}) 即 Sarsa。", "points": 1},
           {"criterion": r"n→∞：尾部 q 项消失，\bar q_t=G_t 即 MC。", "points": 1},
           {"criterion": r"Q-learning：\bar q_t=r_{t+1}+\gamma\max_a q_t(s_{t+1},a)，换 max 但公式形式相同。", "points": 1},
       ],
       model_answer=(
           r"统一公式为 q_{t+1}=q_t−α(q_t−\bar q_t)，其中"
           r"\bar q_t=\sum_{i=1}^{n}\gamma^{i-1}r_{t+i}+\gamma^n q_t(s_{t+n},a_{t+n})。"
           "\n"
           r"n=1 时 \bar q_t=r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})，即 Sarsa。"
           r"n→∞ 时 γ^n 项趋于 0，\bar q_t=\sum_{i=1}^{\infty}\gamma^{i-1}r_{t+i}=G_t，即 MC。"
           "\n"
           r"Q-learning 在同一框架中取 \bar q_t=r_{t+1}+\gamma\max_a q_t(s_{t+1},a)，"
           r"只是把 q(s_{t+1},a_{t+1}) 换成 max_a q(s_{t+1},a)，"
           r"公式形式不变，但求解的方程从 Bellman 期望方程变为最优方程。"),
       ),
    _q("q17", "code", "coding",
       "实现 TD(0) 状态值一步更新",
       "实现 td_update(v, transition, alpha, gamma)。"
       "transition 是元组 (s, r, s_next)；v 是状态值列表，索引为状态编号。"
       "返回一个新列表，其中只有 v[s] 按 TD(0) 更新，其余不变。"
       "不得原地修改传入的 v。只用标准库；公开测试在浏览器 Pyodide 中运行。",
       8, "实现 TD(0) 的增量更新并保证不修改输入。", TD_SOURCE,
       language="python", entry_point="td_update",
       starter_code=(
           "def td_update(v, transition, alpha, gamma):\n"
           "    s, r, s_next = transition\n"
           "    # 返回新列表，只更新 v[s]\n"
           "    pass\n"
       ),
       tests=[
           {"name": "basic_update", "description": "v=[2,5], 转移(0,3,1), α=0.1, γ=0.9 → v[0]=2.55。",
            "code": ("r = td_update([2.0, 5.0], (0, 3.0, 1), 0.1, 0.9)\n"
                     "assert abs(r[0] - 2.55) < 1e-9 and abs(r[1] - 5.0) < 1e-9")},
           {"name": "only_visited_changes", "description": "只有被访问的状态更新，其余不变。",
            "code": ("r = td_update([1.0, 2.0, 3.0], (1, 0.0, 2), 0.5, 0.9)\n"
                     "assert abs(r[0] - 1.0) < 1e-9 and abs(r[2] - 3.0) < 1e-9\n"
                     "assert abs(r[1] - 2.35) < 1e-9")},
           {"name": "no_mutation", "description": "不得原地修改传入的 v。",
            "code": ("v = [2.0, 5.0]\n"
                     "out = td_update(v, (0, 3.0, 1), 0.1, 0.9)\n"
                     "assert v == [2.0, 5.0] and out is not v")},
           {"name": "self_transition", "description": "s_next=s 时也正确计算。",
            "code": ("r = td_update([10.0], (0, 1.0, 0), 0.5, 1.0)\n"
                     "assert abs(r[0] - 10.5) < 1e-9")},
       ],
       rubric=[
           {"criterion": "正确计算 TD target = r + γ·v[s_next] 并做增量更新。", "points": 4},
           {"criterion": "只更新被访问状态，其余不变。", "points": 2},
           {"criterion": "不修改输入列表，返回新列表；只用标准库。", "points": 2},
       ],
       _solution=_TD_SOLUTION),
    _q("q18", "code", "coding",
       "实现 Sarsa 一步 Q 值更新",
       "实现 sarsa_update(q, transition, alpha, gamma)。"
       "transition 是 (s, a, r, s_next, a_next)；q 是二维列表 q[s][a]。"
       "返回深拷贝新列表，只更新 q[s][a]，其余不变。不得原地修改输入。"
       "只用标准库；公开测试在浏览器 Pyodide 中运行。",
       8, "实现 on-policy Sarsa 增量更新并保证不修改输入。", SARSA_SOURCE,
       language="python", entry_point="sarsa_update",
       starter_code=(
           "def sarsa_update(q, transition, alpha, gamma):\n"
           "    s, a, r, s_next, a_next = transition\n"
           "    # 返回新 q，只更新 q[s][a]\n"
           "    pass\n"
       ),
       tests=[
           {"name": "basic_update", "description": "q=[[1,0],[4,0]], 转移(0,0,2,1,0), α=0.5, γ=0.8 → q[0][0]=3.1。",
            "code": ("r = sarsa_update([[1.0, 0.0], [4.0, 0.0]], (0, 0, 2.0, 1, 0), 0.5, 0.8)\n"
                     "assert abs(r[0][0] - 3.1) < 1e-9 and abs(r[1][0] - 4.0) < 1e-9")},
           {"name": "only_one_entry_changes", "description": "只有 q[s][a] 更新。",
            "code": ("r = sarsa_update([[1.0, 0.0], [4.0, 0.0]], (0, 0, 2.0, 1, 0), 0.5, 0.8)\n"
                     "assert abs(r[0][1] - 0.0) < 1e-9 and abs(r[1][1] - 0.0) < 1e-9")},
           {"name": "no_mutation", "description": "不得原地修改输入。",
            "code": ("q = [[1.0, 0.0], [4.0, 0.0]]\n"
                     "out = sarsa_update(q, (0, 0, 2.0, 1, 0), 0.5, 0.8)\n"
                     "assert q == [[1.0, 0.0], [4.0, 0.0]] and out is not q and out[0] is not q[0]")},
           {"name": "three_states", "description": "3 状态 2 动作的正确性。",
            "code": ("q = [[0.0, 0.0], [0.0, 0.0], [5.0, 1.0]]\n"
                     "r = sarsa_update(q, (0, 1, 3.0, 2, 0), 0.1, 0.9)\n"
                     "assert abs(r[0][1] - 0.45) < 1e-9")},
       ],
       rubric=[
           {"criterion": "正确计算 Sarsa target = r + γ·q[s_next][a_next] 并增量更新。", "points": 4},
           {"criterion": "深拷贝二维列表，只更新 q[s][a]，不修改输入。", "points": 3},
           {"criterion": "只用标准库。", "points": 1},
       ],
       _solution=_SARSA_SOLUTION),
    _q("q19", "code", "coding",
       "实现 Q-learning 一步更新",
       "实现 q_learning_update(q, transition, alpha, gamma)。"
       "transition 是 (s, a, r, s_next)；q 是二维列表 q[s][a]。"
       "返回深拷贝新列表，只更新 q[s][a]，用 max(q[s_next]) 作为 target。"
       "不得原地修改输入。只用标准库。",
       6, "实现 off-policy Q-learning 增量更新。", Q_SOURCE,
       language="python", entry_point="q_learning_update",
       starter_code=(
           "def q_learning_update(q, transition, alpha, gamma):\n"
           "    s, a, r, s_next = transition\n"
           "    # 返回新 q，用 max(q[s_next]) 做 target\n"
           "    pass\n"
       ),
       tests=[
           {"name": "basic_update", "description": "q=[[3,0],[6,4]], 转移(0,0,1,1), α=0.2, γ=0.5 → q[0][0]=3.2。",
            "code": ("r = q_learning_update([[3.0, 0.0], [6.0, 4.0]], (0, 0, 1.0, 1), 0.2, 0.5)\n"
                     "assert abs(r[0][0] - 3.2) < 1e-9 and abs(r[1][0] - 6.0) < 1e-9")},
           {"name": "uses_max_not_first", "description": "正确选 max，不是第一个动作。",
            "code": ("r = q_learning_update([[0.0, 0.0], [1.0, 9.0]], (0, 0, 0.0, 1), 1.0, 1.0)\n"
                     "assert abs(r[0][0] - 9.0) < 1e-9")},
           {"name": "no_mutation", "description": "不得原地修改输入。",
            "code": ("q = [[3.0, 0.0], [6.0, 4.0]]\n"
                     "out = q_learning_update(q, (0, 0, 1.0, 1), 0.2, 0.5)\n"
                     "assert q == [[3.0, 0.0], [6.0, 4.0]] and out is not q and out[0] is not q[0]")},
       ],
       rubric=[
           {"criterion": "正确计算 target = r + γ·max(q[s_next]) 并增量更新。", "points": 3},
           {"criterion": "深拷贝二维列表，不修改输入。", "points": 2},
           {"criterion": "只用标准库。", "points": 1},
       ],
       _solution=_QLEARNING_SOLUTION),
    _q("q20", "code", "coding",
       "实现 ε-greedy 动作概率",
       "实现 epsilon_greedy_probs(q_values, epsilon)，返回每个动作在 ε-greedy 策略下的概率列表。"
       "greedy 动作取 Q 值最大的第一个（ties 取最先出现的）；"
       "greedy 动作概率为 (1−ε)+ε/|A|，其余为 ε/|A|。"
       "只用标准库。",
       6, "实现 ε-greedy 的完整概率分配与并列处理。", SARSA_SOURCE,
       language="python", entry_point="epsilon_greedy_probs",
       starter_code=(
           "def epsilon_greedy_probs(q_values, epsilon):\n"
           "    # 返回每个动作的概率列表\n"
           "    pass\n"
       ),
       tests=[
           {"name": "basic", "description": "Q=[5,3,2], ε=0.3 → [0.8, 0.1, 0.1]。",
            "code": ("p = epsilon_greedy_probs([5, 3, 2], 0.3)\n"
                     "assert all(abs(a - b) < 1e-9 for a, b in zip(p, [0.8, 0.1, 0.1]))")},
           {"name": "eps_zero", "description": "ε=0 时 greedy 动作概率为 1。",
            "code": ("p = epsilon_greedy_probs([5, 3, 2], 0.0)\n"
                     "assert abs(p[0] - 1.0) < 1e-9 and abs(p[1]) < 1e-9 and abs(p[2]) < 1e-9")},
           {"name": "eps_one", "description": "ε=1 时均匀分布。",
            "code": ("p = epsilon_greedy_probs([5, 3, 2], 1.0)\n"
                     "assert all(abs(a - 1/3) < 1e-9 for a in p)")},
           {"name": "sums_to_one", "description": "概率之和为 1。",
            "code": ("p = epsilon_greedy_probs([1, 5, 3, 2], 0.2)\n"
                     "assert abs(sum(p) - 1.0) < 1e-9")},
           {"name": "tie_breaking", "description": "并列取第一个 max。",
            "code": ("p = epsilon_greedy_probs([3, 3, 2], 0.3)\n"
                     "assert abs(p[0] - 0.8) < 1e-9 and abs(p[1] - 0.1) < 1e-9")},
       ],
       rubric=[
           {"criterion": "正确计算 greedy 动作和非 greedy 动作的概率。", "points": 3},
           {"criterion": "并列时取第一个 max。", "points": 2},
           {"criterion": "概率之和为 1；只用标准库。", "points": 1},
       ],
       _solution=_EPS_GREEDY_SOLUTION),
]

TOTAL_SCORE = sum(q["max_score"] for q in QUESTIONS)
assert TOTAL_SCORE == 100, "question bank must total 100 points"

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
            raise bank.InvalidAnswer("答案个数不符，期望 %d 个数" % len(wanted))
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
