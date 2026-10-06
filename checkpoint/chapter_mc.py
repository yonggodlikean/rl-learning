"""Source-grounded Monte Carlo checkpoint for Zhao chapter 5 and EasyRL chapter 3."""

import math

import bank


CHAPTER_ID = "monte-carlo"
REFERENCE = (
    "Zhao《Mathematical Foundations of Reinforcement Learning》第5章"
    "（印刷页77–100 / PDF页90–113）；"
    "EasyRL v1.0.6 §3.3.1、§3.3.3、§3.4（印刷页52–53、57–60 / PDF页60–61、65–68）"
)
CONVENTIONS = (
    "回报使用 G_t = r_{t+1} + γr_{t+2} + …。"
    "有限自然终止轨迹取 G_T=0；持续任务的截断不是自然终止。"
    "本章的 Q 默认带有指定策略，不能自动替换成 Q*。"
    "除注明原书图5.3外，小样本与轨迹均为教学构造，不是训练日志。"
)
PRESENTATION = {
    "eyebrow": "MONTE CARLO / 双教材",
    "intro": (
        "从回报样本估计价值，再比较动作、改进策略。"
        "本章连接 Zhao 第5章与 EasyRL 的 MC 评估和免模型控制。"
    ),
    "field_notes": [
        {"label": "01 / RETURN", "formula": "G_t = r_(t+1) + γ G_(t+1)",
         "tex": r"G_t=r_{t+1}+\gamma G_{t+1}",
         "description": "先确定轨迹结束的含义，再从后往前计算。"},
        {"label": "02 / ESTIMATE", "formula": "Q = ΣG / N",
         "tex": r"\widehat Q(s,a)=\frac{\sum_i G_i(s,a)}{N(s,a)}",
         "description": "访问计数和回报累计必须采用同一种访问规则。"},
        {"label": "03 / EXPLORE", "formula": "π(greedy|s) = 1−ε+ε/m",
         "tex": r"\pi(a_g\mid s)=1-\epsilon+\frac{\epsilon}{m}",
         "description": "随机探索仍可能选中贪心动作；固定 ε 限制了策略集合。"},
    ],
    "note_footer": "书中两个 MC ES 伪代码的访问规则不同：Zhao 每次访问，EasyRL 首次访问。",
}


def _source(zhao, zhao_printed, zhao_pdf, easy, easy_printed, easy_pdf):
    return {"text": (
        f"Zhao §{zhao}，印刷页 {zhao_printed} / PDF 页 {zhao_pdf}；"
        f"EasyRL v1.0.6 §{easy}，印刷页 {easy_printed} / PDF 页 {easy_pdf}。"
    )}


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


_NUMBER_CODE = '''import math

def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite int or float")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("number outside float range") from exc
    if not math.isfinite(result):
        raise ValueError("expected a finite number")
    return result
'''

_RETURNS_SOLUTION = _NUMBER_CODE + '''
def discounted_returns(rewards, gamma):
    if not isinstance(rewards, (list, tuple)):
        raise ValueError("rewards must be a list or tuple")
    gamma = _number(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma outside [0, 1]")
    values = [_number(reward) for reward in rewards]
    output = [0.0] * len(values)
    total = 0.0
    for t in range(len(values) - 1, -1, -1):
        total = values[t] + gamma * total
        if not math.isfinite(total):
            raise ValueError("return outside float range")
        output[t] = total
    return output
'''

_MC_SOLUTION = _NUMBER_CODE + '''
def mc_action_values(episodes, gamma, visit="every"):
    gamma = _number(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma outside [0, 1]")
    if visit not in ("first", "every"):
        raise ValueError("visit must be first or every")
    if not isinstance(episodes, (list, tuple)):
        raise ValueError("episodes must be a list or tuple")
    sums, counts = {}, {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("episode must be a dict")
        pairs, rewards = episode.get("pairs"), episode.get("rewards")
        if not isinstance(pairs, (list, tuple)) or not isinstance(rewards, (list, tuple)):
            raise ValueError("pairs and rewards must be lists or tuples")
        if len(pairs) != len(rewards):
            raise ValueError("one reward is required for each pair")
        keys = []
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(not isinstance(x, str) or not x for x in pair)):
                raise ValueError("each pair must contain two nonempty strings")
            keys.append(tuple(pair))
        values = [_number(reward) for reward in rewards]
        returns = [0.0] * len(values)
        total = 0.0
        for t in range(len(values) - 1, -1, -1):
            total = values[t] + gamma * total
            if not math.isfinite(total):
                raise ValueError("return outside float range")
            returns[t] = total
        seen = set()
        for key, total in zip(keys, returns):
            if visit == "first" and key in seen:
                continue
            seen.add(key)
            sums[key] = sums.get(key, 0.0) + total
            if not math.isfinite(sums[key]):
                raise ValueError("sum outside float range")
            counts[key] = counts.get(key, 0) + 1
    return ({key: sums[key] / counts[key] for key in sums}, counts)
'''

_EPSILON_SOLUTION = _NUMBER_CODE + '''
def epsilon_greedy_probs(q_values, epsilon):
    if not isinstance(q_values, (list, tuple)) or not q_values:
        raise ValueError("q_values must be a nonempty list or tuple")
    epsilon = _number(epsilon)
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon outside [0, 1]")
    values = [_number(value) for value in q_values]
    greedy = max(range(len(values)), key=lambda index: values[index])
    result = [epsilon / len(values)] * len(values)
    result[greedy] += 1 - epsilon
    return result
'''

_INVALID_TEST_HELPER = '''
def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")
'''

QUESTIONS = [
    _q("q01", "choice", "concept", "MC 的免模型到底省去了什么？",
       "固定策略 π。你能与环境交互并记录完整轨迹，但不知道奖励分布与状态转移概率。"
       "下列哪项正确描述 MC 价值估计？", 4,
       "区分不需要已知模型与不需要数据。",
       _source("5.1、5.2.1", "78–81", "91–94", "3.3.1", "52", "60"),
       options=[
           "直接对轨迹的折扣回报求样本均值，不必事先知道 P 和 R 的分布。",
           "不需要环境模型，也不需要任何轨迹，只对状态编号求均值。",
           "先精确求出 P，再使用贝尔曼方程，才叫 MC。",
           "只统计即时奖励，不考虑后续奖励。"],
       _answer=0, feedback_hint="免模型不等于无数据；价值是回报的期望。"),
    _q("q02", "choice", "concept", "免模型控制为什么直接估计 Q？",
       r"不知道环境模型，只估计出固定策略的 \(V^\pi(s)\)。"
       r"为了逐状态比较动作、进行贪心改进，哪种做法符合两本书的 MC 控制思路？", 4,
       "说明动作条件价值在免模型策略改进中的作用。",
       _source("5.2.1–5.2.2", "80–82", "93–95", "3.4", "58–59", "66–67"),
       options=[
           r"对 \(V^\pi(s)\) 的状态下标取 argmax，就得到动作。",
           r"直接由从 \((s,a)\) 出发、之后跟随 π 的回报估计 \(Q^\pi(s,a)\)，再对动作取 argmax。",
           r"固定策略的 \(Q^\pi\) 定义为今后每步都选择最优动作，因此已经等于 \(Q_*\)。",
           "只需选训练中出现次数最多的动作。"],
       _answer=1, feedback_hint="Q 固定首步动作，后续跟随指定策略；估计 V 后转为 Q 通常仍需模型。"),
    _q("q03", "choice", "concept", "initial、first、every visit 如何区分？",
       "教学轨迹依时间顺序访问四个状态动作对："
       "(s1,a2)、(s2,a4)、(s1,a2)、(s2,a3)。"
       "只讨论这一回合贡献的回报样本数。"
       "Zhao 算法5.2采用 every-visit；EasyRL 图3.25采用 first-visit。哪项正确？", 4,
       "区分两书伪代码的访问规则与 MC Basic 的 initial-visit。",
       _source("5.3.1–5.3.3", "86–88", "99–101", "3.4，图3.25", "60", "68"),
       options=[
           "initial、first、every 都只能更新最初的 (s1,a2)。",
           "first 从回合末端看，选最后一次出现；every 每个状态只更新一次。",
           "initial 共1个样本；first 共3个；every 共4个，后两者覆盖3种状态动作对。",
           "first 共4个样本；every 共3个，因为 every 要去重。"],
       _answer=2, feedback_hint="首次访问指时间正序中的首次，不是逆序扫描时首次遇到。"),
    _q("q04", "choice", "concept", "MC 是否使用自举？",
       r"一个有限回合自然终止，终止后的回报为0。"
       r"标准 MC 用实际观察到的 \(G_t\) 更新价值。以下说法正确的是？", 4,
       "区分采样回报与估计值自举。",
       _source("5.2.1、5.3.3", "81、88", "94、101", "3.3.1、3.3.3", "52–53、57–58", "60–61、65–66"),
       options=[
           "MC 不采样，只对所有可能下一状态求精确期望。",
           "只要采用递推 G_t=r_(t+1)+γG_(t+1)，就一定在自举。",
           "MC 的目标必须包含当前估计的 V(s_(t+1))。",
           "MC 使用采样轨迹的实际回报，不用另一个估计价值补齐尾部，因此不自举。"],
       _answer=3, feedback_hint="递推计算已观察奖励的回报，不等于用估计的 V 自举。"),
    _q("q05", "choice", "concept", "Exploring Starts 覆盖什么？",
       "环境允许你重置初始状态，并指定第一个动作。"
       "要满足 MC Exploring Starts 的探索性开始条件，哪项要求最关键？", 4,
       "识别探索条件作用于状态动作对而非仅状态。",
       _source("5.3.3", "88–89", "101–102", "3.4，图3.25", "59–60", "67–68"),
       options=[
           "每个合法 (s,a) 都有正概率作为回合起点，并在持续采样时得到充分探索。",
           "每个状态都能作为起点，但首步永远固定为同一个动作即可。",
           "只要某个回合很长，即使有不可达状态，也必然覆盖所有状态动作对。",
           "对每个状态建立 Q 的字典条目，就已经完成探索。"],
       _answer=0, feedback_hint="有表项不等于有经验；只探索状态也可能漏掉好动作。"),
    _q("q06", "choice", "concept", "固定 ε 的“最优”属于哪个集合？",
       r"MC ε-Greedy 始终固定 \(\epsilon=0.2\)，不衰减；"
       "假定采样和可达性条件充分，价值估计已足够准确。"
       "按 Zhao §5.4.2，关于所得策略的最优性，哪项最准确？", 4,
       "区分 ε 受限策略集合中的最优与所有策略中的最优。",
       _source("5.4.2、5.5", "90–91、92–97", "103–104、105–110", "3.4", "59–60", "67–68"),
       options=[
           "固定 ε>0 与 ε=0 得到的策略在任何环境里都相同。",
           "只能说它在给定 ε 的策略集合内最优，不一定是所有策略中的全局最优。",
           "ε 越大，长期价值在所有环境里越高。",
           "任意把 ε 快速降到0，都自动保证充分探索和全局最优。"],
       _answer=1, feedback_hint="固定探索概率可能牺牲收益；衰减本身也不证明覆盖充分。"),

    _q("q07", "numeric", "hand-calculation", "从后往前算三个回报",
       r"教学回合有3步，奖励依次为 \([r_1,r_2,r_3]=[0,-1,2]\)，"
       r"\(\gamma=0.9\)。第3步后自然终止，\(G_3=0\)。"
       r"按时间正序输入 \([G_0,G_1,G_2]\)，不是逆序输出。", 5,
       "正确处理奖励下标、折扣次数和终止尾部。",
       _source("5.3.3，算法5.2", "88", "101", "3.3.1，式3.2", "52", "60"),
       answer_type="list", tolerance=1e-6, _expected=[.72, .8, 2],
       feedback_hint="每个 G_t 从下一次奖励开始，计算可逆序，输出要正序。"),
    _q("q08", "numeric", "hand-calculation", "样本均值与固定学习率不是同一更新",
       r"同一个 \((s,a)\) 已有3个回报样本，样本均值为2；现在收到新回报6。"
       "分别从旧值2出发计算：①更新计数后用样本均值的增量公式；"
       r"②改用固定学习率 \(\alpha=0.2\) 做一次 \(Q\leftarrow Q+\alpha(G-Q)\)。"
       "输入 [①的新值, ②的新值]；两种更新不是先后执行。", 5,
       "区别 1/N 样本平均与固定步长更新。",
       _source("5.1、5.3.3", "78–80、88", "91–93、101", "3.3.1，式3.5–3.7", "52–53", "60–61"),
       answer_type="list", tolerance=1e-6, _expected=[3, 2.8],
       feedback_hint="新的 N 是4；固定 α 的第二个计算仍从旧值2出发。"),
    _q("q09", "numeric", "hand-calculation", "重复访问怎样进入 Q 的均值？",
       "教学回合按正序给出：\n"
       "t=0: (s1,a2)，下一次奖励 r1=0\n"
       "t=1: (s2,a4)，下一次奖励 r2=0\n"
       "t=2: (s1,a2)，下一次奖励 r3=0\n"
       "t=3: (s2,a3)，下一次奖励 r4=1\n"
       r"之后自然终止，\(G_4=0\)，\(\gamma=0.5\)。没有历史样本。"
       "仅对 (s1,a2) 计算 [first-visit 的 Q, every-visit 的 Q]。", 5,
       "使用各次访问的后缀回报，而不是重复使用整回合回报。",
       _source("5.3.1", "87", "100", "3.4，图3.25", "60", "68"),
       answer_type="list", tolerance=1e-6, _expected=[.125, .3125],
       feedback_hint="同一对在 t=0 与 t=2 的回报不同；first 只取时间最早的访问。"),
    _q("q10", "numeric", "hand-calculation", "探索分支还能选到贪心动作",
       r"动作顺序固定为 \([a_1,a_2,a_3,a_4,a_5]\)，"
       r"教学估计 \(Q=[1,4,2,0,-1]\)，唯一贪心动作是 \(a_2\)。"
       r"\(\epsilon=0.2\)：以 \(1-\epsilon\) 选贪心动作，"
       "以 ε 在全部5个动作中均匀抽取（包括贪心动作）。"
       "按上述动作顺序输入5个概率。", 5,
       "把探索分支对贪心动作的概率也计入。",
       _source("5.4.1", "89–90", "102–103", "3.4", "59–60", "67–68"),
       answer_type="list", tolerance=1e-6, _expected=[.04, .84, .04, .04, .04],
       feedback_hint="贪心动作概率不是简单的 1−ε；随机分支也可能抽到它。"),
    _q("q11", "numeric", "hand-calculation", "s3：完整 Q 表与贪心改进",
       r"原书图5.3是持续任务。格子排列如下："
       r"\[\begin{matrix}s_1&s_2&s_3\\s_4&s_5&s_6\\s_7&s_8&s_9\end{matrix}\]"
       "\n动作编号：a1上、a2右、a3下、a4左、a5停留。"
       "\n规则：移动一格；撞边界留在原地，奖励−1；"
       "进入或停在 s6、s7 奖励−1（禁区可进入）；进入或停在 s9 奖励+1；其余奖励0。"
       "到 s9 不终止，停留可持续获得奖励。"
       "\nγ=0.9；初始策略 π0：s1上、s2下、s3右、s4右、s5下、s6下、s7右、s8右、s9停。"
       "\n从 s3 分别执行一次首步动作，然后一直遵循 π0。"
       r"输入 \([q_{\pi_0}(s_3,a_1),\ldots,q_{\pi_0}(s_3,a_5),k]\)，"
       "其中 k 是这五个值中最大者的动作编号（1至5）。本题不是直接求 Q*。", 5,
       "在完整任务定义下评估初始策略，并完成一次贪心改进。",
       _source("5.2.3，图5.3；1.2–1.3", "83–84、2–3", "96–97、15–16", "3.4", "58–59", "66–67"),
       answer_type="list", tolerance=1e-6, _expected=[-10, -10, 8, 7.29, -9, 3],
       feedback_hint="右转后 π0 会一直撞墙；向下先罚一次，但更早进入持续给奖励的目标。"),
    _q("q12", "numeric", "hand-calculation", "截断遗漏了多少尾部？",
       r"教学持续任务：已经在目标状态，每步停留的奖励均为+1，\(\gamma=0.9\)，永不自然终止。"
       "程序只记录接下来的3个奖励，然后把截断点以后的回报当成0。"
       r"输入 [三步截断回报, 被遗漏尾部在当前时刻的折扣价值]。"
       "尾部从第4个奖励开始；不要将截断当作任务真实终止。", 5,
       "计算截断误差并识别其不由重复采样自动消除。",
       _source("5.2.3", "83–86", "96–99", "3.3.1", "52–53", "60–61"),
       answer_type="list", tolerance=1e-6, _expected=[2.71, 7.29],
       feedback_hint="前三项是 1+γ+γ²，遗漏的第一项是 γ³。"),

    _q("q13", "open", "application", "把目标改成终止状态，会改变路线偏好吗？",
       "两条指定路线（不是要求搜索所有路线）：A 为 s3→s6→s9，"
       "前两次奖励为−1、+1；B 为 s3→s2→s5→s8→s9，前四次奖励为0、0、0、+1。"
       "\n原书持续任务：到 s9 后一直停留，每步继续奖励+1；γ=0.9。"
       "\n教学改动：到 s9 的那一步仍奖励+1，但随后立即自然终止，之后无奖励。"
       "\n分别计算两种任务下 A、B 的回报并比较路线偏好。"
       "说明差异是 MC 算错了，还是任务定义变了。所有比较都以给定路线为准。", 6,
       "识别持续任务与回合制任务的奖励及终止语义差异。",
       _source("5.2.3，图5.3", "83–84", "96–97", "3.3.1", "52", "60"),
       rubric=[
           {"criterion": "持续任务：A=8、B=7.29，A更好。", "points": 2},
           {"criterion": "改为终止任务：A=−0.1、B=0.729，B更好。", "points": 2},
           {"criterion": "解释目标尾部奖励被删除，任务定义和回报变了，并非 MC 统计规则错误。", "points": 2},
       ],
       model_answer=(
           "持续任务：A=−1+0.9/(1−0.9)=8；B=0.9³/(1−0.9)=7.29，选A。"
           "\n终止任务：A=−1+0.9=−0.1；B=0.9³=0.729，选B。"
           "\n前者尽早到目标会得到更长的正奖励尾部；后者只有一次目标奖励。"
           "改变的是任务的终止与奖励定义，不能把原书的持续任务公式直接套在修改后的环境上。")),
    _q("q14", "open", "application", "100个回报不一定等于100个独立样本",
       "随机回报 X 的均值为0、方差为1。对两种教学采样方案各取100个值："
       "\nA：100次相互独立、同分布的回报 X1,…,X100。"
       "\nB：只抽一次 X，然后复制100次，所有值都等于这一次 X。"
       "\n分别给出两种样本均值的期望与方差，判断是否会随数量增加而可靠地集中到0。"
       "再解释为什么同一回合的 every-visit 后缀回报，不能不加条件就套用 iid 的方差除以 n 公式。", 6,
       "核对均值估计的独立性假设，区分样本条数与有效信息。",
       _source("5.1，Box5.1；5.3.1", "79–80、87", "92–93、100", "3.3.1", "52–53", "60–61"),
       rubric=[
           {"criterion": "两种均值的期望都为0，B仍可能无偏，但不等于一致或精确。", "points": 2},
           {"criterion": "A的均值方差为0.01，B为1；只有A的此项方差随样本数下降。", "points": 2},
           {"criterion": "every-visit 后缀共享奖励，存在相关性；不能自动视为 iid，不能据此宣称所有相关样本均无效。", "points": 2},
       ],
       model_answer=(
           "A：E[均值]=0，Var(均值)=1/100=0.01，独立样本增多会集中到0。"
           "\nB：均值始终等于唯一抽到的X，所以期望仍为0，方差始终为1；复制不会创造信息。"
           "\nevery-visit 的后缀回报会重叠，Box5.1里的独立性步骤不能直接使用。"
           "这不等于每次访问法不能学习，只是必须另行考虑相关性、覆盖与任务条件。")),
    _q("q15", "open", "application", "串起 MC Basic、MC ES 和 MC ε-Greedy",
       "考虑有限状态动作空间中的持续学习。你能采集回合，但真实设备不允许任意指定初始 (s,a)。"
       "请比较两本书介绍的三条思路："
       "\n①MC Basic 如何从策略迭代变成免模型？②MC Exploring Starts 在样本利用和策略更新上改了什么？"
       "③MC ε-Greedy 如何替代任意重置起点的探索机制？"
       "\n最后说明：若某状态从允许的起点完全不可达，soft policy 能否保证访问它？"
       "若固定 ε>0，所得“最优”属于什么集合？不要声称跑一次有限回合必然覆盖全部动作。", 6,
       "解释算法改进的目的及其探索和最优性边界。",
       _source("5.2–5.5", "80–97", "93–110", "3.4", "58–60", "66–68"),
       rubric=[
           {"criterion": "Basic 用 Q 的回报样本均值替代依赖 P、R 的评估，再贪心改进；initial-visit 利用较少。", "points": 2},
           {"criterion": "ES复用首访或每次访问的后缀回报，逐回合进行广义策略迭代，仍要求状态动作起点覆盖。", "points": 2},
           {"criterion": "ε-soft 保持动作探索，但不创造不可达状态；固定ε的最优仅相对于该ε策略集合。", "points": 2},
       ],
       model_answer=(
           "Basic：对各(s,a)采集回报、取平均估计Q，然后按Q贪心改进，替换的是模型依赖的评估。"
           "\nES：一条回合的多个后缀可用于不同访问的估计，不必只用起始访问；"
           "策略可逐回合改进。Zhao算法5.2是every-visit，EasyRL图3.25是first-visit。"
           "\nε-Greedy：把贪心改进改为ε-greedy，使可到达状态下的每个动作保持探索机会。"
           "这仍依赖足够采样与可达性，不能访问物理上完全不可达的状态。"
           "固定ε的最佳策略仅是对应受限策略集合中的最佳，不一定等于全局贪心最优策略。")),

    _q("q16", "code", "coding", "实现有限轨迹的全部折扣回报",
       "实现 discounted_returns(rewards, gamma)，仅用 Python 标准库。"
       "\nrewards 为有限 int/float 的 list 或 tuple，rewards[t] 对应 r_(t+1)。"
       "自然终止后的 G_T=0；gamma 是 [0,1] 内的有限 int/float。"
       "返回按时间正序排列的 [G0,…,G_(T−1)]，空序列返回 []，不得修改输入。"
       "\n教学例：[1,0,2]，gamma=0.5。此函数不负责估计持续任务被截去的尾部。"
       "\n布尔值、非数值、非有限数、非法容器或范围，以及计算结果溢出，均抛 ValueError。", 8,
       "将完整有限回报的逆序递推实现为无副作用函数。",
       _source("5.3.3，算法5.2", "88", "101", "3.3.1，式3.2", "52", "60"),
       language="python", entry_point="discounted_returns",
       starter_code="def discounted_returns(rewards, gamma):\n    # G_T = 0; return G_0 through G_(T-1).\n    return [0.0] * len(rewards)\n",
       tests=[
           {"name": "discount_and_order", "description": "教材递推式应用于教学轨迹，输出保持时间正序。",
            "code": "import math\nout=discounted_returns([1,0,2],.5)\nassert len(out)==3 and all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(out,[1.5,1,2]))"},
           {"name": "gamma_edges", "description": "有限回合支持 gamma=0 和 gamma=1。",
            "code": "assert discounted_returns([1,2,3],0)==[1,2,3]\nassert discounted_returns((1,2,3),1)==[6,5,3]"},
           {"name": "empty_and_mutation", "description": "空输入与输入不被修改。",
            "code": "assert discounted_returns([],1)==[]\nr=[1,2];before=r[:];out=discounted_returns(r,.5)\nassert r==before and out is not r"},
           {"name": "invalid_inputs", "description": "拒绝布尔值、NaN/Inf、非法类型和参数。",
            "code": _INVALID_TEST_HELPER + "\nfor r,g in [([True],.9),([float('nan')],.9),([float('inf')],.9),([1],True),([1],-.1),([1],1.1),([1],float('nan')),([1],'0.9'),('12',.9)]:\n    rejects(lambda r=r,g=g: discounted_returns(r,g))"},
           {"name": "overflow", "description": "可表示的输入若累计溢出，也应明确拒绝。",
            "code": _INVALID_TEST_HELPER + "\nrejects(lambda: discounted_returns([1e308,1e308],1))\nrejects(lambda: discounted_returns([10**1000],.9))"},
       ],
       rubric=[{"criterion": "奖励下标、折扣和逆序递推正确，输出保持正序。", "points": 4},
               {"criterion": "终止尾部、gamma端点、空序列且不修改输入。", "points": 2},
               {"criterion": "非法与非有限输入、溢出按约定抛ValueError。", "points": 2}],
       _solution=_RETURNS_SOLUTION),
    _q("q17", "code", "coding", "按 first/every visit 累计动作价值",
       '实现 mc_action_values(episodes, gamma, visit="every")，仅用标准库，返回 (Q, N) 两个字典。'
       "\nepisodes 为 list/tuple，每个元素是 {'pairs': [...], 'rewards': [...]}。"
       "pairs[t] 是两个非空字符串构成的 list/tuple，表示 (s_t,a_t)；rewards[t] 是下一次奖励。"
       "两序列长度相等；每个回合自然终止，尾部回报0；空回合允许。"
       "\nQ、N 的键为 (状态字符串, 动作字符串) 元组：Q 为被计入的后缀回报均值，N 为计入次数。"
       'visit 只允许 "first" 或 "every"；first 是每回合时间正序首次出现，跨回合重新判断。'
       "\ngamma 是 [0,1] 内有限 int/float；奖励为有限 int/float。空回合集返回 ({},{})。不得修改输入。"
       "\n教学例：pairs=[('A','x'),('B','y'),('A','x')]，rewards=[1,0,2]，gamma=0.5。"
       "\n布尔值、非数值/非有限值、长度不符、非法容器/键/模式、参数越界或累计溢出均抛 ValueError。"
       "这是固定数据的 MC 估计函数，不是整个 MC 控制训练器；不要依赖其他题已定义的函数。", 12,
       "实现两种访问策略，区分状态动作键并逐回合重置首访判定。",
       _source("5.3.1–5.3.3", "87–88", "100–101", "3.4，图3.25", "60", "68"),
       language="python", entry_point="mc_action_values",
       starter_code='def mc_action_values(episodes, gamma, visit="every"):\n    # Return (Q, N), keyed by (state, action).\n    return {}, {}\n',
       tests=[
           {"name": "every_suffix_returns", "description": "重复同一状态动作对要用不同后缀回报。",
            "code": "e=[{'pairs':[('A','x'),('B','y'),('A','x')],'rewards':[1,0,2]}]\nq,n=mc_action_values(e,.5)\nassert q=={('A','x'):1.75,('B','y'):1}\nassert n=={('A','x'):2,('B','y'):1}"},
           {"name": "first_is_earliest", "description": "首次是时间正序最早，不是从后扫描最先遇到。",
            "code": "e=[{'pairs':[('A','x'),('B','y'),('A','x')],'rewards':[1,0,2]}]\nq,n=mc_action_values(e,.5,'first')\nassert q[('A','x')]==1.5 and n[('A','x')]==1"},
           {"name": "reset_each_episode", "description": "首访集合逐回合重置，不能合并同状态的不同动作。",
            "code": "e=[{'pairs':[('s','a'),('s','b')],'rewards':[1,2]}, {'pairs':[('s','a')],'rewards':[4]}]\nq,n=mc_action_values(e,1,'first')\nassert q=={('s','a'):3.5,('s','b'):2} and n=={('s','a'):2,('s','b'):1}"},
           {"name": "empty_and_edges", "description": "空回合、空数据、gamma=0。",
            "code": "assert mc_action_values([],1)==({},{})\nassert mc_action_values([{'pairs':[],'rewards':[]}],1,'first')==({},{})\nq,n=mc_action_values([{'pairs':[['s','a'],['s','a']],'rewards':[2,4]}],0)\nassert q=={('s','a'):3} and n=={('s','a'):2}"},
           {"name": "no_mutation", "description": "嵌套输入不应被修改。",
            "code": "import copy\ne=[{'pairs':[['s','a']],'rewards':[2]}];before=copy.deepcopy(e)\nmc_action_values(e,.9);assert e==before"},
           {"name": "invalid_shapes_and_modes", "description": "拒绝长度不符、非法对和未知访问规则。",
            "code": _INVALID_TEST_HELPER + "\nfor e,g,v in [([{'pairs':[('s','a')],'rewards':[]}],.9,'every'),([{'pairs':[('s',)],'rewards':[1]}],.9,'first'),([{'pairs':[('', 'a')],'rewards':[1]}],.9,'first'),([{}],.9,'every'),([],1,'last'),([],1,True),(None,.9,'every')]:\n    rejects(lambda e=e,g=g,v=v: mc_action_values(e,g,v))"},
           {"name": "nonfinite_and_overflow", "description": "拒绝布尔/非有限输入、累计溢出。",
            "code": _INVALID_TEST_HELPER + "\nfor reward in [True,float('nan'),float('inf'),10**1000]:\n    rejects(lambda reward=reward: mc_action_values([{'pairs':[('s','a')],'rewards':[reward]}],.9))\nrejects(lambda: mc_action_values([],True))\nrejects(lambda: mc_action_values([],1.1))\nrejects(lambda: mc_action_values([{'pairs':[('s','a'),('s','a')],'rewards':[1e308,1e308]}],1))"},
       ],
       rubric=[{"criterion": "各访问后缀回报、累积和与计数一致。", "points": 4},
               {"criterion": "first取时间最早、每回合重置，并以完整(s,a)为键。", "points": 4},
               {"criterion": "空数据、边界、错误输入、溢出、无输入副作用。", "points": 4}],
       _solution=_MC_SOLUTION),
    _q("q18", "code", "coding", "实现 ε-greedy 的概率分布",
       "实现 epsilon_greedy_probs(q_values, epsilon)，仅用标准库，返回与动作顺序一致的概率列表。"
       "\nq_values 是非空 list/tuple，其中每个 Q 为有限 int/float；epsilon 是 [0,1] 内有限 int/float。"
       "探索分支在全部动作上均匀抽取，包含贪心动作。"
       "\n本题额外约定：Q 最大值并列时，唯一指定贪心动作为最小下标；不是在并列者间均分利用概率。"
       "支持 epsilon=0、epsilon=1 与单一动作；不修改输入。"
       "\n教学例：Q=[2,5,−1]，epsilon=0.3。"
       "布尔值、非数值、NaN/Inf、空/非法容器、参数越界均抛 ValueError。"
       "函数返回概率而不抽取动作，也不依赖其他题的函数。", 8,
       "实现 ε-greedy 公式并明确并列动作约定。",
       _source("5.4.1–5.4.2", "89–90", "102–103", "3.4", "59–60", "67–68"),
       language="python", entry_point="epsilon_greedy_probs",
       starter_code="def epsilon_greedy_probs(q_values, epsilon):\n    # Return one probability per action, in input order.\n    return [0.0] * len(q_values)\n",
       tests=[
           {"name": "includes_greedy_in_exploration", "description": "探索也能抽到贪心动作。",
            "code": "import math\np=epsilon_greedy_probs([2,5,-1],.3)\nassert len(p)==3 and all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(p,[.1,.8,.1]))\nassert math.isclose(sum(p),1,abs_tol=1e-9)"},
           {"name": "tie_first", "description": "并列时选最小下标为唯一指定贪心动作。",
            "code": "import math\np=epsilon_greedy_probs([2,2,0],.3)\nassert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(p,[.8,.1,.1]))"},
           {"name": "epsilon_edges", "description": "epsilon=0纯贪心，epsilon=1均匀。",
            "code": "assert epsilon_greedy_probs([1,3],0)==[0,1]\nassert epsilon_greedy_probs([1,3],1)==[.5,.5]\nassert epsilon_greedy_probs([9],.3)==[1]"},
           {"name": "no_mutation", "description": "输入 Q 不变，支持 tuple。",
            "code": "q=[2,1];before=q[:];epsilon_greedy_probs(q,.4);assert q==before\nassert epsilon_greedy_probs((2,1),0)==[1,0]"},
           {"name": "invalid_inputs", "description": "拒绝空输入、布尔值、NaN/Inf及参数越界。",
            "code": _INVALID_TEST_HELPER + "\nfor q,e in [([],.2),([True],.2),([float('nan')],.2),([float('inf')],.2),([1],True),([1],-.1),([1],1.1),([1],float('nan')),([1],'0.2'),('12',.2),([10**1000],.2)]:\n    rejects(lambda q=q,e=e: epsilon_greedy_probs(q,e))"},
       ],
       rubric=[{"criterion": "每动作探索底数与贪心动作额外概率正确。", "points": 4},
               {"criterion": "并列、epsilon端点、单动作与无输入副作用。", "points": 2},
               {"criterion": "类型、有限性、范围和空输入校验。", "points": 2}],
       _solution=_EPSILON_SOLUTION),
]

TOTAL_SCORE = sum(q["max_score"] for q in QUESTIONS)
assert TOTAL_SCORE == 100
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
