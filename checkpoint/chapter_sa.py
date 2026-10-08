"""Stochastic approximation: core recurrences, convergence and sample gradients."""

import math

import bank


CHAPTER_ID = "stochastic-approximation"
REFERENCE = (
    "Zhao《Mathematical Foundations of Reinforcement Learning》第6章"
    "（印刷页101–124 / PDF页114–137）；"
    "EasyRL v1.0.6 §3.3.1、§3.3.2、§6.5–6.6"
    "（印刷页52–53、55–56、102–104 / PDF页60–61、63–64、110–112）"
)
CONVENTIONS = (
    "更新编号 k 从1开始；w_1 是初值，完成 k 次更新后得到 w_(k+1)。"
    "路径返回 [初值, 第一次更新结果, …]。"
    "代码题保证输入合法、序列长度匹配、数值规模适当；不考类型校验或异常处理。"
    "短样本、固定噪声和回归数据是教学构造，不是真实训练记录。"
    "有限轨迹接近真解不等于证明几乎必然收敛。"
)
PRESENTATION = {
    "eyebrow": "STOCHASTIC APPROXIMATION / 双教材",
    "intro": "从增量均值到随机求根，再把样本梯度与期望梯度联系起来。",
    "field_notes": [
        {"label": "01 / INCREMENT", "formula": "w_(k+1) = w_k + α_k(x_k − w_k)",
         "tex": r"w_{k+1}=w_k+\alpha_k(x_k-w_k)",
         "description": "取 α_k=1/k 时，更新结果等于已收到样本的算术平均。"},
        {"label": "02 / ROOT", "formula": "w_(k+1) = w_k − α_k g̃_k",
         "tex": r"w_{k+1}=w_k-\alpha_k\widetilde g_k",
         "description": "RM 求 g(w)=0；g 若是损失的梯度，便连接到 SGD。"},
        {"label": "03 / STEP SIZE", "formula": "Σα_k=∞；Σα_k²<∞",
         "tex": r"\sum_k\alpha_k=\infty,\qquad\sum_k\alpha_k^2<\infty",
         "description": "还需函数与噪声条件，不能只凭步长宣称收敛。"},
    ],
    "note_footer": "EasyRL 提供增量更新和批量回归应用；RM 与 Dvoretzky 定理以 Zhao 第6章为主。",
}


def _source(zhao, printed, pdf, easy, easy_printed, easy_pdf, note=""):
    return {"text": (
        f"Zhao §{zhao}，印刷页 {printed} / PDF 页 {pdf}；"
        f"EasyRL v1.0.6 §{easy}，印刷页 {easy_printed} / PDF 页 {easy_pdf}。"
        + note
    )}


MEAN_SOURCE = _source("6.1、6.2.2", "102–103、108–109", "115–116、121–122",
                      "3.3.1，式3.4–3.7", "52–53", "60–61")
RM_SOURCE = _source("6.2–6.2.1", "103–108", "116–121",
                    "3.3.1–3.3.2", "52–53、55–56", "60–61、63–64",
                    "EasyRL 仅为增量更新的应用对照，不含独立 RM 收敛定理。")
PROOF_SOURCE = _source("6.3.1–6.3.3", "110–113", "123–126",
                       "3.3.1–3.3.2", "52–53、55–56", "60–61、63–64",
                       "Dvoretzky 定理及证明来自 Zhao；EasyRL 只对应更新形式。")
SGD_SOURCE = _source("6.4–6.4.3、6.4.5", "114–119、121–123", "127–132、134–136",
                     "6.5–6.6", "102–104", "110–112",
                     "EasyRL 对应从样本批次做回归的应用，不考 DQN 控制器。")
BATCH_SOURCE = _source("6.4.3–6.4.4", "118–121", "131–134",
                       "6.5–6.6", "102–104", "110–112",
                       "本题的平方损失和小样本是教学构造。")


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


_MEAN_SOLUTION = """def mean_path(samples, steps, initial=0.0):
    w = initial
    path = [w]
    for x, alpha in zip(samples, steps):
        w = w + alpha * (x - w)
        path.append(w)
    return path
"""

_RM_SOLUTION = """def rm_path(oracle, steps, initial):
    w = initial
    path = [w]
    for alpha in steps:
        observed = oracle(w)
        w = w - alpha * observed
        path.append(w)
    return path
"""

_REGRESSION_SOLUTION = """def regression_batch_step(w, points, alpha):
    residuals = [(w * x - y, x) for x, y in points]
    loss = sum(0.5 * residual ** 2 for residual, x in residuals) / len(points)
    gradient = sum(residual * x for residual, x in residuals) / len(points)
    return w - alpha * gradient, loss, gradient
"""

_BATCH_SOLUTION = """def batch_gradients(w, values, batch_size):
    from itertools import product
    return [
        sum(w - x for x in batch) / batch_size
        for batch in product(values, repeat=batch_size)
    ]
"""


QUESTIONS = [
    _q("q01", "choice", "concept", "RM 是求根，还是把 g 当损失最小化？",
       r"给定递增函数 \(g\)，你只能在输入 \(w_k\) 时观测"
       r" \(\widetilde g_k=g(w_k)+\eta_k\)，不能查询其解析表达式或导数。"
       r"RM 更新为 \(w_{k+1}=w_k-\alpha_k\widetilde g_k\)。它直接试图解决什么问题？",
       4, "区分随机求根、损失最小化与 Newton 法。", RM_SOURCE,
       options=[
           r"最小化 \(g(w)\) 本身，使其越负越好。",
           r"寻找 \(g(w)=0\) 的根；当 \(g=J'\) 时，这也对应损失 \(J\) 的驻点问题。",
           r"先获取 \(g'(w_k)\)，再计算 Newton 更新 \(-g/g'\)。",
           "解析计算噪声序列，并要求每个噪声样本都等于0。"],
       _answer=1, feedback_hint="更新用的是带噪声的函数输出，不是 g 的导数。"),
    _q("q02", "choice", "concept", "什么时候增量更新恰好是样本均值？",
       r"样本依次为 \(x_1,x_2,\ldots\)，初值 \(w_1\) 是任意有限数。"
       r"每收到一个样本就做 \(w_{k+1}=w_k+\alpha_k(x_k-w_k)\)。"
       r"哪项精确描述这个递推，而不只是其期望？", 4,
       "识别样本均值的1/k权重和固定增益的区别。", MEAN_SOURCE,
       options=[
           r"任意正的 \(\alpha_k\) 都使 \(w_{k+1}\) 等于前k个样本的算术平均。",
           r"固定 \(\alpha=0.1\) 时，每个历史样本始终有相同权重。",
           r"\(\alpha_k=1/k\) 时，\(w_{k+1}\) 等于前k个样本的算术平均，首步消除初值的影响。",
           "增量更新必须存储全部历史样本，才能计算本次更新。"],
       _answer=2, feedback_hint="首步 α_1=1；固定增益一般产生指数加权，而不是等权平均。"),
    _q("q03", "choice", "concept", "两条步长求和条件同时限制了什么？",
       r"本题讨论经典充分条件：\(\alpha_k>0\)、"
       r"\(\sum_k\alpha_k=\infty\)、\(\sum_k\alpha_k^2<\infty\)。"
       r"令 \(\alpha_k=c/k^\beta\)，其中 \(c>0\) 为常数，k从1开始。"
       "下面哪个指数满足两条求和条件？", 4,
       "用p级数判断步长，而非只检查步长趋零。", RM_SOURCE,
       options=[r"\(\beta=1/2\)", r"\(\beta=3/2\)",
                r"\(\beta=3/4\)", "所有 β>0 都满足，因为步长都趋于0。"],
       _answer=2, feedback_hint="第一条要求 β≤1，第二条要求 2β>1。"),
    _q("q04", "choice", "concept", "噪声需要高斯吗？",
       r"\(\mathcal F_k\) 为当前更新前的历史，包含 \(w_k\)。"
       "本题采用条件二阶矩统一有界的常用严格版本，函数与步长条件已满足。"
       "哪项足以描述所需的噪声条件？", 4,
       "区分条件无偏、普通均值为零和噪声分布名称。", RM_SOURCE,
       options=[
           r"\(\mathbb E[\eta_k\mid\mathcal F_k]=0\)，且"
           r" \(\mathbb E[\eta_k^2\mid\mathcal F_k]\le C<\infty\)，C不随k变化。",
           "必须是独立标准正态；其他零均值分布都不能使用。",
           r"只需 \(\mathbb E[\eta_k]=0\)，不管它与当前历史的关系。",
           "噪声绝对值必须每一步严格减小，才能使用 RM。"],
       _answer=0, feedback_hint="条件无偏是关键；高斯与独立同分布是常见特例，不是必需的分布形式。"),
    _q("q05", "choice", "concept", "立方例子能直接套用全局定理吗？",
       r"本题给定一组充分条件：存在常数 \(0<m\le g'(w)\le M<\infty\)"
       r" 对所有实数w成立。现在 \(g(w)=w^3-5\)。"
       "即使某条有限实验轨迹接近根，下列判断哪项正确？", 4,
       "将具体函数与全局导数条件逐项对照。", RM_SOURCE,
       options=[
           "g有唯一实根，所以自动满足所有收敛条件。",
           "只要用1/k步长，就可以忽略函数和噪声条件。",
           "它不满足这组充分条件，所以任何步长、任何初值都不可能收敛。",
           r"\(g'(w)=3w^2\) 在0处为0且全局无上界；不能直接套用该定理，"
           "但这也不等于所有轨迹必定发散。"],
       _answer=3, feedback_hint="充分条件不满足，只说明不能用这条定理保证，不是证明永远不收敛。"),
    _q("q06", "choice", "concept", "mini-batch 的大小等于数据集就一定是 BGD？",
       r"固定数据集只有 \(x_1=0,x_2=4\)，损失为"
       r" \(f(w,x)=\tfrac12(w-x)^2\)。BGD每次使用两条原始记录各一次。"
       "另一算法每步独立、均匀、有放回抽两次，取两个样本梯度的平均。"
       "哪项正确？", 4,
       "区分批量大小、样本覆盖与有放回采样。", BATCH_SOURCE,
       options=[
           "两者每步都使用2个数，所以每一步的梯度必定相等。",
           "有放回批次可能是(0,0)或(4,4)；其梯度均值无偏，但单次不必等于BGD。",
           "有放回抽样一定有偏，所以不能称为SGD或mini-batch。",
           "mini-batch只要出现重复记录，就应该抛出异常拒绝该批次。"],
       _answer=1, feedback_hint="m=n并不保证覆盖所有记录；随机抽样的重复记录是算法现象，不是非法输入。"),

    _q("q07", "numeric", "calculation", "从任意初值算三次样本均值更新",
       r"教学样本按顺序为 \(x_1=4,x_2=-2,x_3=6\)，初值 \(w_1=99\)。"
       r"使用 \(w_{k+1}=w_k+\frac1k(x_k-w_k)\)，k从1开始。"
       r"依次提交 \([w_2,w_3,w_4]\)，不要包含初值。", 4,
       "实际执行1/k均值递推，核对时间下标。", MEAN_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[4, 1, 8 / 3],
       feedback_hint="每次用已经更新过的w；k=1时初值的权重变为0。"),
    _q("q08", "numeric", "calculation", "固定增益为什么不是算术平均？",
       r"教学样本按顺序为 \(2,4,-2\)，初值 \(w_1=0\)。"
       r"固定 \(\alpha=1/4\)，更新 \(w_{k+1}=w_k+\alpha(x_k-w_k)\)。"
       r"依次提交 \([w_2,w_3,w_4]\)。本题不是1/k样本均值算法。", 4,
       "手算指数加权递推，而非误用sum/n。", MEAN_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[0.5, 1.375, 0.53125],
       feedback_hint="每一步都保留旧估计的3/4，再加入新样本的1/4。"),
    _q("q09", "numeric", "calculation", "审计立方求根例子的前两步",
       r"给定 \(g(w)=w^3-5\)，求唯一实根 \(w^*\)。"
       r"然后做一个明确改为无噪声的参数核验：\(w_1=0,\alpha_k=1/k,\eta_k=0\)，"
       r"\(w_{k+1}=w_k-\alpha_k g(w_k)\)。"
       r"依次提交 \([w^*,w_2,w_3]\)，根至少保留6位小数。"
       "这是对书中参数的无噪声教学核验，不是声称复现了原图的随机噪声。",
       4, "区分解析根与数值迭代，识别过大的初始更新。", RM_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[5 ** (1 / 3), 5, -55],
       feedback_hint="解析根来自解w³=5；第二次更新必须在新位置w_2重新计算立方。"),
    _q("q10", "numeric", "calculation", "带噪声的线性 RM 实际更新",
       r"教学函数 \(g(w)=2w-6\)，初值 \(w_1=0\)。"
       r"两次步长均为 \(1/4\)，固定噪声按顺序为 \(\eta_1=1,\eta_2=-1\)。"
       r"每次观测 \(\widetilde g_k=g(w_k)+\eta_k\)，"
       r"更新 \(w_{k+1}=w_k-\alpha_k\widetilde g_k\)。"
       r"提交 \([w_2,w_3]\)。这些噪声是给定的演算数据，不要求随机抽样。", 4,
       "将观测噪声加到函数输出，再执行负反馈更新。", RM_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[1.25, 2.375],
       feedback_hint="先算g加噪声，再减去步长乘观测；不是把噪声加到更新后的w。"),
    _q("q11", "numeric", "calculation", "同一位置比较 BGD、SGD、MBGD",
       r"教学数据集 \([-1,3,5]\)，损失 \(f(w,x)=\tfrac12(w-x)^2\)，"
       r"样本梯度为 \(w-x\)。三种算法都从同一个旧值 \(w=0\) 出发，步长 \(\alpha=0.2\)。"
       "BGD使用整个数据集；SGD此次取样x=5；MBGD此次批次为[-1,5]，取梯度平均。"
       r"三次更新相互独立，提交 \([w_{\rm BGD},w_{\rm SGD},w_{\rm MBGD}]\)。",
       4, "比较同一点的梯度估计，不把三个算法接成顺序更新。", BATCH_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[7 / 15, 1, 0.4],
       feedback_hint="每种方法都从w=0重算；批量梯度必须除以相应样本数。"),
    _q("q12", "numeric", "calculation", "mini-batch 梯度究竟降低了什么？",
       r"固定参数 \(w=7\)，\(f(w,X)=\tfrac12(w-X)^2\)。"
       r"已知 \(\mathbb E[X]=2,\operatorname{Var}(X)=9\)。"
       "一个mini-batch含9个相互独立、同分布样本，所有样本梯度都在同一个w计算并取平均。"
       r"提交 \([\mathbb E[w-X],\operatorname{Var}(w-X),"
       r"\operatorname{Var}(\text{批次平均梯度})]\)。", 4,
       "区分梯度均值与梯度方差，使用独立性而非口头宣称更稳定。", BATCH_SOURCE,
       answer_type="list", tolerance=1e-6, _expected=[5, 9, 1],
       feedback_hint="减去常数不改变X的方差；独立平均降低方差，不改变梯度均值。"),

    _q("q13", "open", "explanation", "每一步方差有限，为什么仍不能保证收敛？",
       r"教学反例：\(g(w)=w-1\)，\(\alpha_k=1/k\)。"
       r"\(Z_k\) 是独立标准正态且独立于更新前历史，令 \(\eta_k=kZ_k\)。"
       r"因此每一步 \(\mathbb E[\eta_k\mid\mathcal F_k]=0\)，"
       r"\(\mathbb E[\eta_k^2\mid\mathcal F_k]=k^2<\infty\)。"
       "我们采用噪声条件二阶矩统一有界的标准充分条件。"
       r"算出 \(\alpha_k\eta_k\)，说明缺了什么条件，为什么不能从“每步有限”"
       "直接宣称几乎必然收敛。最后说明有限实验轨迹为何不能替代证明。", 5,
       "识别随时间增大的噪声及有限实验的证据边界。", PROOF_SOURCE,
       rubric=[
           {"criterion": "指出无随k统一的二阶矩上界，k²虽逐项有限却不断增大。", "points": 2},
           {"criterion": "算出α_kη_k=Z_k，噪声更新项不因1/k而缩小。", "points": 2},
           {"criterion": "没有满足该充分条件；有限运行不是几乎必然收敛证明。", "points": 1}],
       model_answer=(
           "条件均值确实为0，但条件二阶矩是k²，不存在一个对所有k适用的有限常数C。"
           "\nα_kη_k=(1/k)kZ_k=Z_k，因此更新中的随机扰动仍是单位尺度。"
           "不能用这个标准定理保证收敛；步长满足求和条件并不能抵消任意增长的噪声。"
           "\n教材定理6.1(c)简写为二阶矩有限；此处明确采用§6.3中一致界的严格版本，"
           "不能把逐项有限当成一致有界。几乎必然收敛涉及无限序列，有限仿真不能证明。")),
    _q("q14", "open", "explanation", "把均值估计同时写成 RM 和 SGD",
       r"设 \(X_k\) 相互独立、同分布，均值 \(\mu\) 未知，方差有限。"
       r"更新前 \(w_k\) 由历史样本决定，当前 \(X_k\) 独立于该历史。"
       r"已有更新 \(w_{k+1}=w_k+\alpha_k(X_k-w_k)\)。"
       r"请分别给出RM的 \(g(w)\)、\(\widetilde g_k\)、噪声 \(\eta_k\)，"
       r"以及SGD的样本损失 \(f(w,X)\)、样本梯度和最优解。"
       r"说明为什么算法不需要事先知道 \(\mu\)。", 5,
       "建立增量均值、随机求根、平方损失SGD三者的等价关系。", MEAN_SOURCE,
       rubric=[
           {"criterion": "RM：g=w−μ，观测w−X，η=μ−X，条件均值0。", "points": 2},
           {"criterion": "SGD：f=(w−X)²/2，样本梯度w−X，最优解μ。", "points": 2},
           {"criterion": "实现只用w和当前样本；μ用于分析而不是算法输入。", "points": 1}],
       model_answer=(
           "RM取g(w)=w−μ，观测g̃_k=w_k−X_k，噪声η_k=μ−X_k。"
           "因当前样本独立于历史，E[η_k|F_k]=0。"
           "\nSGD取f(w,X)=(w−X)²/2，样本梯度w−X；期望梯度w−μ，"
           "所以最优解w*=μ。减去样本梯度得到w+α(X−w)。"
           "\nμ是未知目标，只出现在证明中。实现不需要知道μ，也不需要调用整个分布。")),
    _q("q15", "open", "explanation", "误差平方中的交叉项为什么消失？",
       r"给定误差递推 \(e_{k+1}=(1-a_k)e_k+b_k\eta_k\)。"
       r"\(e_k,a_k,b_k\) 都由更新前历史 \(\mathcal F_k\) 确定，"
       r"\(\mathbb E[\eta_k\mid\mathcal F_k]=0\)，"
       r"\(\mathbb E[\eta_k^2\mid\mathcal F_k]=\sigma^2\)。"
       r"展开 \(\mathbb E[e_{k+1}^2\mid\mathcal F_k]\)，解释交叉项为什么为0。"
       "结合Dvoretzky定理的思路，指出哪个项反映收缩、哪个项反映噪声注入。"
       "本题只考核心误差递推，不要求复写拟鞅证明。", 5,
       "理解Dvoretzky分析的能量递推和条件期望，而非背诵术语。", PROOF_SOURCE,
       rubric=[
           {"criterion": "写出(1−a_k)²e_k²+b_k²σ²。", "points": 2},
           {"criterion": "交叉项是已知系数乘条件均值0，不是每条路径η=0。", "points": 2},
           {"criterion": "辨认收缩项与噪声项，并联系b_k²可求和的作用。", "points": 1}],
       model_answer=(
           "展开平方有(1−a_k)²e_k²、2(1−a_k)e_kb_kη_k和b_k²η_k²。"
           "历史决定的系数可提出条件期望，交叉项因E[η_k|F_k]=0而消失。"
           "\n得到E[e_(k+1)²|F_k]=(1−a_k)²e_k²+b_k²σ²。"
           "前项在适当步长下表示误差收缩，后项是噪声注入。"
           "噪声贡献可求和与持续收缩共同支撑收敛分析，"
           "不是声称每次随机实现的误差都更小。")),
    _q("q16", "open", "explanation", "均值接近真解，就等于路径收敛吗？",
       r"教学模型：\(X_k\) 独立同分布，均值 \(\mu\)、方差1；"
       r"\(w_1\) 是确定初值，更新 \(w_{k+1}=0.8w_k+0.2X_k\)。"
       r"已知期望递推为 \(m_{k+1}=0.8m_k+0.2\mu\)。"
       "推导方差递推，求其稳态值，并说明为什么"
       r"\(\mathbb E[w_k]\to\mu\) 不能推出每条路径收敛到μ。"
       "联系SGD接近最优点时的随机波动；不要把有限图像当作理论保证。", 5,
       "区分期望收敛、非零稳态方差与几乎必然点收敛。", SGD_SOURCE,
       rubric=[
           {"criterion": "利用当前X与历史独立，得到v_(k+1)=0.64v_k+0.04。", "points": 2},
           {"criterion": "稳态v=0.04/(1−0.64)=1/9，不等于0。", "points": 2},
           {"criterion": "均值逼近不代表随机估计停止波动；固定步长不满足平方可求和条件。", "points": 1}],
       model_answer=(
           "X_k独立于由过去构成的w_k，因此协方差项为0，"
           "v_(k+1)=0.8²v_k+0.2²=0.64v_k+0.04。"
           "\n稳态v=0.04/0.36=1/9。期望虽然趋近μ，随机估计仍保持非零尺度的波动；"
           "这里不应仅用非零方差作任意过程的几乎必然收敛反证。"
           "对本递推而言，新样本持续注入扰动，"
           "若w_k几乎必然趋于μ，等式X_k=(w_(k+1)−0.8w_k)/0.2会要求X_k也趋于μ，"
           "这与非退化iid样本的持续随机性不符。"
           "\n固定步长的平方和发散。接近最优点时真梯度减小，噪声相对影响会更明显。")),

    _q("q17", "code", "coding", "一个循环看懂样本均值与固定增益",
       "实现 mean_path(samples, steps, initial=0.0)，返回"
       " [initial, 第一次更新结果, …, 最后结果]。"
       r"\n第k个样本和对应步长执行 \(w\leftarrow w+\alpha(x-w)\)，"
       "下一步使用更新后的w。所有输入保证合法，samples与steps长度一致；"
       "零次更新只返回[initial]。不要求类型检查或异常处理。"
       "\n公开测试分别给1/k步长与固定增益：同一个循环应产生不同的历史权重。"
       "只用标准库，不依赖其他题的函数。", 8,
       "用最短递推实现在线均值，直接观察步长改变权重。", MEAN_SOURCE,
       language="python", entry_point="mean_path",
       starter_code=(
           "def mean_path(samples, steps, initial=0.0):\n"
           "    w = initial\n    path = [w]\n"
           "    for x, alpha in zip(samples, steps):\n"
           "        # Update w using this sample, then append it.\n"
           "        path.append(w)\n    return path\n"),
       tests=[
           {"name": "prefix_means", "description": "1/k更新对应教材式6.2，结果是每个前缀的均值。",
            "code": "import math\nxs=[3,1,5,7]\np=mean_path(xs,[1,.5,1/3,.25],100)\nexpected=[100]+[sum(xs[:k])/k for k in range(1,5)]\nassert len(p)==5 and all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(p,expected))"},
           {"name": "fixed_gain", "description": "固定增益不应偷偷改成算术平均。",
            "code": "assert mean_path([6,0,6],[.5,.5,.5])==[0,3,1.5,3.75]"},
           {"name": "different_steps", "description": "逐步使用给定步长和刚更新的估计。",
            "code": "assert mean_path([2,4],[.25,.5],8)==[8,6.5,5.25]"},
           {"name": "gain_one", "description": "步长1让本次估计完全由当前样本决定。",
            "code": "assert mean_path([-1,2,10],[1,1,1],4)==[4,-1,2,10]"},
           {"name": "zero_updates", "description": "零次更新对应保留初值的数学语义。",
            "code": "assert mean_path([],[],9)==[9]"},
       ],
       rubric=[{"criterion": "当前样本、当前步长与最新w构成正确增量更新。", "points": 5},
               {"criterion": "路径包含初值与每一步结果，反映1/k和固定增益的不同。", "points": 3}],
       _solution=_MEAN_SOLUTION),
    _q("q18", "code", "coding", "只查询黑盒观测的 RM 求根",
       "实现 rm_path(oracle, steps, initial)，返回包含初值的整条路径。"
       "oracle是题目提供的可调用函数，oracle(w)返回本次带噪声观测；"
       "它可能有内部样本计数。每个步长只能在当前w上调用oracle一次。"
       r"\n核心更新为 \(w\leftarrow w-\alpha\,\mathrm{oracle}(w)\)。"
       "不查询真根、真实g或导数，不重新生成噪声，不把oracle当作需要求导的损失。"
       "所有输入合法且计算规模适当；只用标准库，不写校验，不依赖其他题。"
       "\n测试含稳定线性函数、固定噪声序列和小步长立方例子。"
       "通过有限测试不是任意初值下收敛的证明。", 8,
       "实现无解析表达式、无导数的观测反馈，并保持一步一次查询。", RM_SOURCE,
       language="python", entry_point="rm_path",
       starter_code=(
           "def rm_path(oracle, steps, initial):\n"
           "    w = initial\n    path = [w]\n"
           "    for alpha in steps:\n"
           "        observed = oracle(w)\n"
           "        # Use the observed output in the RM update.\n"
           "        path.append(w)\n    return path\n"),
       tests=[
           {"name": "linear_feedback", "description": "输出为负时增大w，输出为正时减小w。",
            "code": "assert rm_path(lambda w:2*w-8,[.25,.25],0)==[0,2,3]"},
           {"name": "one_observation_per_step", "description": "噪声oracle有内部状态，一步只能查一次当前w。",
            "code": "calls=[]\nnoise=iter([.5,-.25,.75])\ndef oracle(w):\n    calls.append(w)\n    return w-3+next(noise)\np=rm_path(oracle,[.5,.25,.125],1)\nassert p==[1,1.75,2.125,2.140625]\nassert calls==p[:-1]"},
           {"name": "stop_at_root", "description": "无噪声时到根便不动，不能继续把g降成负数。",
            "code": "assert rm_path(lambda w:w-3,[1,.5,.25],8)==[8,3,3,3]"},
           {"name": "mean_estimation_oracle", "description": "教材§6.2.2：观测w−当前样本，同样得到增量均值。",
            "code": "import math\nxs=iter([3,-1,2])\np=rm_path(lambda w:w-next(xs),[1,.5,1/3],7)\nassert len(p)==4 and all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(p,[7,3,1,4/3]))"},
           {"name": "small_cubic_steps", "description": "按教材函数做小步长教学演算，不套用全局保证。",
            "code": "import math\np=rm_path(lambda w:w**3-5,[.05,.05],1)\nassert len(p)==3 and all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(p,[1,1.2,1.3636]))"},
       ],
       rubric=[{"criterion": "在当前w上查询一次观测，并做减去步长乘观测的更新。", "points": 6},
               {"criterion": "路径包含初值，oracle内部噪声计数与更新一一对应。", "points": 2}],
       _solution=_RM_SOLUTION),
    _q("q19", "code", "coding", "同一个旧参数上计算批量回归梯度",
       "实现 regression_batch_step(w, points, alpha)，返回"
       " (更新后的w, 更新前的平均损失, 更新前的平均梯度)。"
       r"\n教学模型 \(\widehat y=wx\)，points是非空的[(x,y),…]；"
       r"样本损失 \(\ell(w;x,y)=\tfrac12(wx-y)^2\)，"
       r"样本梯度 \((wx-y)x\)。所有样本必须在同一个旧w上计算，"
       r"然后只更新一次 \(w_{\rm new}=w-\alpha\,\overline{\nabla\ell}\)。"
       "\n一个样本是SGD的一步；小批次是MBGD的一步；完整数据是BGD的一步。"
       "本题是对EasyRL样本回归训练的教学抽象，不是DQN实现。"
       "输入全部合法，不考校验；只用标准库，不依赖其他题。", 8,
       "实现样本梯度、批次平均和冻结旧参数，避免批内顺序SGD。", BATCH_SOURCE,
       language="python", entry_point="regression_batch_step",
       starter_code=(
           "def regression_batch_step(w, points, alpha):\n"
           "    # Compute all residuals at the same old w.\n"
           "    loss = 0.0\n    gradient = 0.0\n"
           "    return w - alpha * gradient, loss, gradient\n"),
       tests=[
           {"name": "mean_loss_and_gradient", "description": "平均平方损失含1/2，梯度含特征x。",
            "code": "import math\nout=regression_batch_step(1,[(1,2),(2,1)],.2)\nassert len(out)==3 and all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(out,[.9,.5,.5]))"},
           {"name": "feature_factor", "description": "梯度不是单独的预测残差，还需乘x。",
            "code": "import math\nout=regression_batch_step(0,[(3,2)],.1)\nassert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(out,[.6,2,-6]))"},
           {"name": "frozen_old_parameter", "description": "批次内不逐条更新w；反转数据不改变本次结果。",
            "code": "p=[(1,1),(2,2)]\na=regression_batch_step(0,p,.1)\nb=regression_batch_step(0,list(reversed(p)),.1)\nassert a==b==(.25,1.25,-2.5)"},
           {"name": "repeat_batch_and_exact_fit", "description": "重复整个批次不改变均值；拟合正确时梯度与损失为0。",
            "code": "p=[(1,2),(2,1)]\nassert regression_batch_step(1,p,.2)==regression_batch_step(1,p+p,.2)\nassert regression_batch_step(2,[(1,2),(-1,-2)],.3)==(2,0,0)"},
           {"name": "single_sample_sgd", "description": "大小为1的批次对应一次样本梯度更新。",
            "code": "import math\nout=regression_batch_step(1,[(2,3)],.1)\nassert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(out,[1.2,.5,-2]))"},
       ],
       rubric=[{"criterion": "半平方损失与含x的样本梯度正确，均按批次数平均。", "points": 5},
               {"criterion": "所有梯度用同一旧w；只更新一次，返回旧损失和旧梯度。", "points": 3}],
       _solution=_REGRESSION_SOLUTION),
    _q("q20", "code", "coding", "精确枚举 mini-batch 的梯度噪声",
       "实现 batch_gradients(w, values, batch_size)，返回所有可能批次的平均梯度列表。"
       r"\n样本损失 \(f(w,x)=\tfrac12(w-x)^2\)，样本梯度为w−x。"
       "values是非空的小数据集；每次均匀、有放回、独立抽batch_size次，"
       "因此共有 len(values)**batch_size 个等概率、有顺序的批次，重复值保留。"
       "\n用标准库 itertools.product(values, repeat=batch_size) 的顺序枚举；"
       "对每个批次，在同一个w上计算样本梯度并取平均。"
       "batch_size是1至3的整数，数据集最多4条，输入均合法。"
       "\n公开测试会对返回列表计算精确均值和总体方差："
       "观察批次变大是否改变均值，以及独立平均如何降低方差。"
       "本题不做随机抽样，避免偶然跑过；不写校验，不依赖其他题。", 8,
       "通过全部等概率批次验证无偏性、方差缩减与重复抽样。", BATCH_SOURCE,
       language="python", entry_point="batch_gradients",
       starter_code=(
           "def batch_gradients(w, values, batch_size):\n"
           "    from itertools import product\n"
           "    gradients = []\n"
           "    for batch in product(values, repeat=batch_size):\n"
           "        # Average the sample gradients at w for this batch.\n"
           "        gradients.append(0.0)\n    return gradients\n"),
       tests=[
           {"name": "single_sample_gradients", "description": "m=1保留全部可能的单样本梯度。",
            "code": "assert batch_gradients(.5,[-2,2],1)==[2.5,-1.5]"},
           {"name": "ordered_replacement_batches", "description": "两个独立抽样产生4个有序批次，重复也计入。",
            "code": "assert batch_gradients(.5,[-2,2],2)==[2.5,.5,.5,-1.5]"},
           {"name": "unbiased_and_variance", "description": "精确枚举的均值不变，总体方差为单样本方差/m。",
            "code": "import math\nfor m in [1,2,3]:\n    gs=batch_gradients(1.25,[-3,0,3],m)\n    assert len(gs)==3**m\n    mean=sum(gs)/len(gs)\n    var=sum((g-mean)**2 for g in gs)/len(gs)\n    assert math.isclose(mean,1.25,abs_tol=1e-9)\n    assert math.isclose(var,6/m,abs_tol=1e-9)"},
           {"name": "full_size_not_full_coverage", "description": "批次大小等于数据集仍可能重复，不能冒充每次BGD。",
            "code": "gs=batch_gradients(0,[1,3],2)\nassert gs==[-1,-2,-2,-3]\nassert len(set(gs))>1"},
           {"name": "parameter_shift", "description": "改变w平移梯度，不改变同一数据的梯度噪声。",
            "code": "import math\na=batch_gradients(.25,[-1,0,2],3)\nb=batch_gradients(2.25,[-1,0,2],3)\nassert len(a)==len(b)==27\nassert all(math.isclose(y-x,2,abs_tol=1e-9) for x,y in zip(a,b))"},
       ],
       rubric=[{"criterion": "按约定枚举全部有序、有放回批次，保留重复样本及批次。", "points": 4},
               {"criterion": "每个批次在同一w上计算并平均梯度，结果支撑精确矩分析。", "points": 4}],
       _solution=_BATCH_SOLUTION),
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
