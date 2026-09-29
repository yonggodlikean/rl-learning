"""EasyRL v1.0.6 §2.3: policy evaluation and model-based control.

Book: printed pp.29–43 = PDF pp.37–51. All decision tables are constructed
teaching data, not logged recommender or trading outcomes. Browser tests are
public self-study tests, not a secure Python execution service.
"""

import math

import bank

REFERENCE = "EasyRL v1.0.6 §2.3（印刷页 29–43；PDF 页 37–51）"
CONVENTIONS = (
    "状态顺序 [候选, 留存]，动作顺序 [普通, 激励]；P[s][a][s'] 是已知模型的"
    "条件转移概率，R[s][a] 是取动作后的即时期望净奖励。"
    "γ=0.5；本章示例仅作有限状态、模型已知时的规划练习。"
    "同步迭代必须使用同一轮旧 V；并列最大值取动作列表中靠前者。"
)
PRESENTATION = {
    "eyebrow": "让动作进入贝尔曼方程",
    "intro": "从固定策略的预测，走到比较动作的控制。先手算 Q 和 V，再在浏览器里写两段无依赖 Python。",
    "field_notes": [
        {"label": "01 / POLICY", "formula": "Pπ = ΣπP",
         "tex": r"P^\pi(s'\mid s)=\sum_a\pi(a\mid s)P(s'\mid s,a)",
         "description": "给定策略，按动作概率把 MDP 折成 MRP。"},
        {"label": "02 / EXPECTATION", "formula": "Vπ = ΣπQπ",
         "tex": r"V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)",
         "description": "先固定动作求 Q，再按当前策略的动作概率取平均。"},
        {"label": "03 / OPTIMALITY", "formula": "V*=max Q*",
         "tex": r"V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]",
         "description": "求最优策略时 max 放在动作项外，不能把即时项和未来项分别最大化。"},
    ],
    "note_footer": "印刷页 40 的中间推导排版容易误读：正确的最优备份是对整项取 max。",
}

# Constructed illustrative model, not production logs or a fitted causal model.
MODEL = {
    "states": ["候选", "留存"],
    "actions": ["普通", "激励"],
    "rewards": [[1.0, 1.0], [2.0, 1.0]],
    "transitions": [
        [[0.5, 0.5], [0.2, 0.8]],
        [[0.1, 0.9], [0.0, 1.0]],
    ],
    "policy": [[0.75, 0.25], [1.0, 0.0]],
    "values": [2.0, 4.0],
    "gamma": 0.5,
}

INDUSTRY_CASE = {
    "kind": "mdp-planning",
    "title": "从曝光汇总到策略评估：推荐案例与真实训练代码怎么区分",
    "scope": (
        "案例表是合成的用户状态/干预动作/下一状态汇总；把转移比例当作已知 P 来演示"
        "§2.3 的模型规划。真实业务需定义净奖励、曝光与反事实、时间切分及非平稳性；"
        "下表不是生产日志，也未实际训练推荐或量化交易模型。"
    ),
    "data_notice": (
        "合成汇总 4 组、每组 10 次曝光，n(下一状态) / n(曝光) 得到 P；"
        "R 为人为指定的即时净收益，不是这些曝光的真实观测均值。"
        "动作策略 π、旧 V 均为教学设定；没有用相同样本估计策略效果。"
    ),
    "model": MODEL,
    "rows": [
        {"state": 0, "action": 0, "exposures": 10, "next_counts": [5, 5]},
        {"state": 0, "action": 1, "exposures": 10, "next_counts": [2, 8]},
        {"state": 1, "action": 0, "exposures": 10, "next_counts": [1, 9]},
        {"state": 1, "action": 1, "exposures": 10, "next_counts": [0, 10]},
    ],
    "code_path": {
        "commit": "a2ad9f6",
        "project": "verl",
        "stages": [
            {"path": "verl/examples/data_preprocess/gsm8k.py",
             "lines": "61–77",
             "role": "GSM8K 预处理写出 data_source、prompt 和 reward_model.ground_truth；训练时生成 response，不从数据行直接读取模型输出。",
             "code": "data = {\n"
                     "    'data_source': data_source,\n"
                     "    'prompt': [{'role': 'user', 'content': question}],\n"
                     "    'reward_model': {'style': 'rule', 'ground_truth': solution},\n"
                     "    # extra_info 等其他字段见源码\n}"},
            {"path": "verl/verl/experimental/agent_loop/agent_loop.py",
             "lines": "997–1027, 1065–1097",
             "role": "仅启用异步规则奖励 worker 且本条尚无分数时，把生成响应和原始字段包装为 DataProto 调用 compute_score；仅所有样本有分数时，把标量写入末个有效 response 位置的 rm_scores。",
             "code": "rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)\n"
                     "rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)"},
            {"path": "verl/verl/experimental/reward_loop/reward_manager/naive.py",
             "lines": "32–43, 54–99",
             "role": "仅选中 experimental 的 naive manager、没有自定义 compute_score 时，解码有效 response，读取 data_source / ground_truth，并按数据源路由评分，返回 reward_score 标量。",
             "code": "data_source = data_item.non_tensor_batch['data_source']\n"
                     "ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']\n"
                     "response_str = await self.loop.run_in_executor(...)\n"
                     "return {'reward_score': reward, 'reward_extra_info': reward_extra_info}"},
            {"path": "verl/verl/utils/reward_score/gsm8k.py",
             "lines": "20–36, 52–72",
             "role": "当 data_source=openai/gsm8k 且为默认规则 scorer 时，strict 提取回答中的 #### 数字与 ground_truth 比较，默认给出 1 或 0。",
             "code": "answer = extract_solution(solution_str=solution_str, method=method)\n"
                     "if answer is None:\n    return 0\n"
                     "if answer == ground_truth:\n    return score\n"
                     "return format_score"},
            {"path": "verl/verl/trainer/ppo/v1/trainer_base.py",
             "lines": "1917–1950, 1987–2003, 2053–2062, 2384–2385",
             "role": "v1 _compute_advantage 把 rm_scores 写入 token_level_scores；无 KL-in-reward、无额外整形时构造 token_level_rewards，再调用 v1 的多轨迹优势估计入口；其他配置不满足此等式。",
             "code": "if self.config.algorithm.use_kl_in_reward:\n"
                     "    data, kl_metrics = apply_kl_penalty(...)\n"
                     "else:\n    data.batch['token_level_rewards'] = data.batch['token_level_scores']"},
            {"path": "verl/verl/trainer/ppo/v1/utils.py",
             "lines": "204–234",
             "role": "adv_estimator 不是 GRPO 时（包括 GAE），多轨迹适配器把 DataProto 原样交给 ray_trainer.compute_advantage。",
             "code": "if adv_estimator != core_algos.AdvantageEstimator.GRPO:\n"
                     "    return compute_advantage(data, adv_estimator=adv_estimator,\n"
                     "                             gamma=gamma, lam=lam, num_repeat=num_repeat,\n"
                     "                             norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
                     "                             config=config)"},
            {"path": "verl/verl/trainer/ppo/ray_trainer.py",
             "lines": "216–261",
             "role": "优势估计分派：adv_estimator=gae 且已提供 critic values 时，取 token_level_rewards / values / response_mask 调用下面的 GAE 函数。",
             "code": "advantages, returns = core_algos.compute_gae_advantage_return(\n"
                     "    token_level_rewards=data.batch['token_level_rewards'],\n"
                     "    values=data.batch['values'], response_mask=data.batch['response_mask'],\n"
                     "    gamma=gamma, lam=lam\n)"},
            {"path": "verl/verl/trainer/ppo/core_algos.py",
             "lines": "216–262",
             "role": "给定批次 token 奖励、critic values 与 mask，在 no_grad 内倒序算 TD 残差与 GAE，returns=advantages+values；不是枚举 P(s'|s,a)。",
             "code": "delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]\n"
                     "lastgaelam_ = delta + gamma * lam * lastgaelam\n"
                     "returns = advantages + values"},
        ],
        "notice": (
            "源码仅作『经典价值/回报概念怎样进入语言模型训练』的对照："
            "verl 的 GAE 用采样 token 与 critic，不读取本案例的 P[s][a][s']；"
            "此链要求 v1 流式异步规则奖励、默认 GSM8K scorer、GAE+critic、无 KL 奖励/额外整形。"
            "两条路径不是同一个算法。代码块是带条件的非连续简化片段，不能直接运行。"
        ),
        "batch": [
            {"id": "A", "mask": [1, 1, 1], "rm_scores": [0, 0, 1],
             "values": [0.2, 0.4, 0.5]},
            {"id": "B", "mask": [1, 1, 0], "rm_scores": [0, 1, 0],
             "values": [0.1, 0.3, 0.0]},
        ],
        "batch_note": (
            "以下 B=2,T=3 数据是独立构造的极小张量演算，不是 verl 运行产物。"
            "取 γ=0.5、λ=1，忽略白化后的 advantages；在有效位置计算原始 returns。"
        ),
    },
}


def _starter_eval():
    return '''def policy_backup(rewards, transitions, policy, values, gamma):
    """固定策略下做一次同步 Bellman 期望备份。
    rewards[s][a]: 即时期望奖励，形状 [S,A]
    transitions[s][a][next_s]: 已知转移概率，形状 [S,A,S]
    policy[s][a]: 动作概率，形状 [S,A]
    values[s]: 旧价值，长度 S
    返回: 新价值 list[float]，长度 S；不修改任何输入。
    空模型 rewards=[] 时返回 []。只用标准库。
    """
    # TODO: 对每个 s 先算各动作的 Q，再用 policy[s] 加权
    return list(values)'''


def _solution_eval():
    return '''def policy_backup(rewards, transitions, policy, values, gamma):
    new_values = []
    for s in range(len(rewards)):
        v = 0.0
        for a in range(len(rewards[s])):
            expected_next = sum(p * values[next_s]
                                for next_s, p in enumerate(transitions[s][a]))
            q = rewards[s][a] + gamma * expected_next
            v += policy[s][a] * q
        new_values.append(v)
    return new_values'''


def _starter_optimal():
    return '''def optimal_backup(rewards, transitions, values, gamma):
    """一次同步 Bellman 最优备份。
    输入形状同上一题；所有非空状态至少有一个动作。
    返回 (new_values, greedy_actions)，两者均为长度 S 的新列表。
    每个状态先求 Q(s,a)，取最大的动作；并列取下标最小者。
    空模型返回 ([], [])，不修改任何输入。只用标准库。
    """
    # TODO: max 作用在整个 R(s,a)+gamma*E[V(next)] 上
    return list(values), []'''


def _solution_optimal():
    return '''def optimal_backup(rewards, transitions, values, gamma):
    new_values, greedy_actions = [], []
    for s in range(len(rewards)):
        q_values = []
        for a in range(len(rewards[s])):
            expected_next = sum(p * values[next_s]
                                for next_s, p in enumerate(transitions[s][a]))
            q_values.append(rewards[s][a] + gamma * expected_next)
        best_action = max(range(len(q_values)), key=lambda a: q_values[a])
        new_values.append(q_values[best_action])
        greedy_actions.append(best_action)
    return new_values, greedy_actions'''


def _source(section, printed, pdf, eq=""):
    return {"text": f"EasyRL v1.0.6 §{section}，印刷页 {printed} / PDF 页 {pdf}"
            + (f"，式 {eq}" if eq else "")}


def _q(question_id, qtype, category, title, prompt, score, section, printed, pdf, **extra):
    return {
        "id": question_id, "type": qtype, "category": category,
        "title": title, "prompt": prompt, "max_score": score,
        "grading_mode": {"choice": "auto", "numeric": "auto",
                         "open": "self", "code": "browser"}[qtype],
        "source_note": _source(section, printed, pdf),
        "review_hint": f"回看 EasyRL §{section}：印刷页 {printed} / PDF 页 {pdf}。",
        **extra,
    }


QUESTIONS = [
    _q("q01", "choice", "concept", "MDP 比 MRP 多了什么？",
       "已知同一状态 s，以下哪项最准确地区分 MDP 和无动作的 MRP？", 5, "2.3.1–2.3.2", "29–30", "37–38",
       options=[
           "MDP 只需即时奖励，不再需要状态转移。",
           r"先依据 \(\pi(a\mid s)\) 选择动作；转移和奖励由 \((s,a)\) 决定。",
           "MDP 的下一个状态只由过去全部历史决定。",
           "MDP 规定动作必须随机且两个动作概率相同。"], _answer=1,
       feedback_hint="先由策略挑动作，再由动作条件转移；确定性策略也允许。"),
    _q("q02", "choice", "concept", "V 与 Q 的条件不同",
       r"固定策略 \(\pi\) 后，\(V^\pi(s)\) 与 \(Q^\pi(s,a)\) 的正确关系是哪一项？", 5, "2.3.3–2.3.4", "30–31", "38–39",
       options=[
           r"\(V^\pi(s)=\max_a Q^\pi(s,a)\)，任何固定策略都成立。",
           r"\(Q^\pi(s,a)=V^\pi(s)\)，因为未来动作按策略选。",
           r"\(V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)\)，Q 先固定首步动作。",
           r"\(V^\pi(s)=\sum_{s'}P(s'\mid s,a)Q^\pi(s,a)\)，不需要动作权重。"],
       _answer=2, feedback_hint="预测当前策略要对动作加权；max 用于控制。"),
    _q("q03", "choice", "concept", "预测还是控制",
       "同样知道环境模型，下列哪件事是控制（control），而不是策略预测（prediction）？", 5,
       "2.3.7、2.3.10–2.3.12", "34、38–41", "42、46–49",
       options=[
           "保持 π 不变，反复计算它在每个状态下的价值。",
           "评估固定的推荐曝光策略是否有长期价值。",
           "对每状态动作的 Q 值取 argmax 改进策略，再继续求最优价值。",
           "只把生成的一条轨迹的奖励相加。"],
       _answer=2, feedback_hint="给定 π 求 Vπ 是预测；优化 π 才是控制。"),
    _q("q04", "choice", "concept", "动态规划的前提",
       "想用 §2.3 的表格价值迭代处理历史推荐点击数据，哪一项是必要的前提/限定？", 5,
       "2.3.8、2.3.12", "35、41", "43、49",
       options=[
           "只要有一条用户轨迹，就知道所有动作的转移概率。",
           "动态规划需要给定或估计模型 P 和 R；历史点击受曝光策略影响，不能直接视作完整反事实模型。",
           "价值迭代和 PPO 是同一个实现，只是状态维度不同。",
           "不观察用户下一状态也能精确求出 P。"],
       _answer=1, feedback_hint="§2.3 的 DP 是已知模型下的规划；日志建模还要处理选择偏差。"),
    _q("q05", "numeric", "hand-calculation", "按策略折叠即时奖励",
       r"独立小例子（不同于下方案例表）：R(s,普通)=2、R(s,激励)=6，"
       r"\(\pi(普通\mid s)=0.75\)，\(\pi(激励\mid s)=0.25\)。求 \(r^\pi(s)\)。",
       8, "2.3.1", "29", "37", answer_type="number", tolerance=1e-6, _expected=3.0,
       feedback_hint="先按动作概率加权：0.75×2+0.25×6。"),
    _q("q06", "numeric", "hand-calculation", "对下一状态折叠转移",
       "按下方合成案例，候选状态采取普通动作时 [去候选,去留存]=[0.5,0.5]，"
       "激励动作时是 [0.2,0.8]，π=[0.75,0.25]。求固定策略下"
       r"\([P^\pi(候选\mid 候选),P^\pi(留存\mid 候选)]\)，按这个顺序输入两个数。",
       8, "2.3.1", "29", "37", answer_type="list", tolerance=1e-6,
       _expected=[0.425, 0.575], feedback_hint="每个下一状态分别对动作概率做一次加权。"),
    _q("q07", "numeric", "hand-calculation", "先求 Q，再求 Vπ",
       "按下方合成案例：旧 V=[2,4]、γ=0.5、候选状态 R=[1,1]，两动作转移分别为"
       "[0.5,0.5]、[0.2,0.8]，π=[0.75,0.25]。做一轮贝尔曼期望备份，"
       r"求候选状态的 \(V^\pi_{\mathrm{new}}\)。先写出两个 Q 再加权。",
       8, "2.3.3–2.3.4、2.3.9", "30–31、36–37", "38–39、44–45",
       answer_type="number", tolerance=1e-6, _expected=2.575,
       feedback_hint="Q普通=2.5、Q激励=2.8；不是两者的最大值。"),
    _q("q08", "numeric", "hand-calculation", "一次同步最优备份",
       "这是一个已知模型的两状态教学例子。状态顺序为 [候选, 留存]，"
       "动作有 [普通, 激励]；下列 P 向量的两项依次表示转移到 [候选, 留存] 的概率。"
       "旧价值 V=[2,4]，折扣因子 γ=0.5。\n"
       "候选 / 普通：R=1，P=[0.5, 0.5]\n"
       "候选 / 激励：R=1，P=[0.2, 0.8]\n"
       "留存 / 普通：R=2，P=[0.1, 0.9]\n"
       "留存 / 激励：R=1，P=[0, 1]\n"
       r"对每个状态的两个动作，分别计算 \(Q(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\)，"
       "再取较大的 Q。输入一次同步备份得到的新价值 [V_new(候选), V_new(留存)]，"
       "用逗号分隔；两行都使用给定的旧 V，不能把先算出的新值用于下一行。"
       "本题只求一轮更新，不是收敛后的最优价值。",
       8, "2.3.11–2.3.12", "39–41", "47–49", answer_type="list",
       tolerance=1e-6, _expected=[2.8, 3.9],
       feedback_hint="候选选激励、留存选普通；两行都读旧 V。"),
    _q("q09", "open", "application", "为什么不能把日志频率当成因果策略？",
       "合成表把转移计数归一化后做模型规划。若这些是实际推荐曝光日志，你需要"
       "额外检查什么才能说某个激励动作更好？把“已知模型的预测/控制”和"
       "“真实历史日志估计模型/评价策略”的差别写清楚。",
       9, "2.3.7–2.3.8", "34–35", "42–43",
       rubric=[
           {"criterion": "指出 §2.3 动态规划以前提为已知/可用的 P、R，只解规划不直接从日志学习。", "points": 3},
           {"criterion": "识别行为/曝光策略、混杂与未曝光动作的反事实缺失；日志频率不能当成随机干预效果。", "points": 3},
           {"criterion": "说明净奖励定义、时间切分与验证/敏感性等可操作检查。", "points": 3},
       ], model_answer="表里的 P、R 一旦给定，可以做策略评估或改进；但真实点击表只记录已展示动作的后果，可能受推荐器选择和用户状态混杂。先定义长期净收益、检查曝光倾向/支持集，按时间拆分训练和评估并做反事实或保守评估；不能由表格练习推出线上提升。"),
    _q("q10", "open", "application", "预测、策略迭代、价值迭代怎么选？",
       "给定一个小规模且已知的仓位状态转移模型：固定仓位策略要评估风险/收益时"
       "该选什么？想搜索最优策略时，策略迭代和价值迭代分别怎么推进？"
       "为什么历史回测盈利不足以证明新策略有效？",
       9, "2.3.7–2.3.14", "34–43", "42–51",
       rubric=[
           {"criterion": "固定策略是预测：以 π 加权的贝尔曼期望备份求 Vπ。", "points": 3},
           {"criterion": "策略迭代交替评估和贪心改进；价值迭代直接对整项 Bellman 最优备份取 max，收敛后提策略。", "points": 3},
           {"criterion": "真实量化需防前视/成本忽略和非平稳性，按时间验证并与非 RL 基线比较。", "points": 3},
       ], model_answer="如果仓位策略已固定，使用贝尔曼期望方程评估 Vπ；要寻找更优仓位，则可反复评估当前策略后对 Q 贪心改进，或直接价值迭代直至稳定再提取动作。历史数据估计的 P、R 可能变化，回测还会有前视、交易成本和反复试错的问题；需要样本外按时间验证及相同成本下的基线比较。"),
    _q("q11", "code", "coding", "编程：固定策略的同步评估",
       "实现 policy_backup(rewards, transitions, policy, values, gamma)。先对每个动作"
       r"计算 \(Q(s,a)\)，再按 \(\pi(a\mid s)\) 求每个状态的新价值。"
       "S×A×S 的列表模型已给定，不要从采样猜 P；空状态列表返回 []，"
       "不能改输入，标准库即可。代码只执行一次同步备份，不要求收敛。",
       15, "2.3.1、2.3.4、2.3.9", "29、31、36–37", "37、39、44–45",
       language="python", entry_point="policy_backup", starter_code=_starter_eval(),
       tests=[
           {"name": "case_model", "description": "合成案例 S=2,A=2，得到 [2.575,3.9]。",
            "code": "r=[[1,1],[2,1]]; p=[[[.5,.5],[.2,.8]],[[.1,.9],[0,1]]]; pi=[[.75,.25],[1,0]]; v=[2,4]\ngot=policy_backup(r,p,pi,v,.5)\nassert len(got)==2 and all(abs(a-b)<1e-9 for a,b in zip(got,[2.575,3.9]))"},
           {"name": "policy_weights", "description": "非均匀动作权重，不可取最大或简单平均。",
            "code": "got=policy_backup([[0,8]],[[[1],[1]]],[[.75,.25]],[0],.9)\nassert len(got)==1 and abs(got[0]-2)<1e-9"},
           {"name": "synchronous", "description": "两状态交换转移：必须同时读旧 V。",
            "code": "got=policy_backup([[0],[0]],[[[0,1]],[[1,0]]],[[1],[1]],[1,3],1)\nassert got==[3,1]"},
           {"name": "gamma_zero", "description": "γ=0 时结果只与即时奖励有关。",
            "code": "assert policy_backup([[1,5]],[[[1],[1]]],[[.25,.75]],[99],0)==[4.0]"},
           {"name": "empty", "description": "空状态模型返回 []。",
            "code": "assert policy_backup([],[],[],[],.5)==[]"},
           {"name": "no_mutation", "description": "P、R、π、旧 V 均不可原地改写。",
            "code": "import copy\nr=[[1]];p=[[[1]]];pi=[[1]];v=[2]; snapshot=copy.deepcopy((r,p,pi,v)); out=policy_backup(r,p,pi,v,.5)\nassert (r,p,pi,v)==snapshot and out is not v"},
       ],
       rubric=[{"criterion": "对每个 s,a 用正确的条件转移计算 Q(s,a)。", "points": 7},
               {"criterion": "按策略概率加权而不是取 max；所有状态读同一份旧 V。", "points": 5},
               {"criterion": "无副作用，空模型与 γ=0 正确。", "points": 3}],
       _solution=_solution_eval()),
    _q("q12", "code", "coding", "编程：一次价值迭代与贪心动作",
       "实现 optimal_backup(rewards, transitions, values, gamma)，返回"
       "(new_values, greedy_actions)。本题一次同步 Bellman 最优备份，"
       "每个状态对完整 Q 项取最大；并列动作取第一个。与上一题比较："
       "期望备份按 π 加权，而最优备份取 max。只用标准库，不做后续 PPO 训练。",
       15, "2.3.10–2.3.13", "38–42", "46–50",
       language="python", entry_point="optimal_backup", starter_code=_starter_optimal(),
       tests=[
           {"name": "case_model", "description": "同一合成模型得到新 V=[2.8,3.9]，动作 [1,0]。",
            "code": "r=[[1,1],[2,1]];p=[[[.5,.5],[.2,.8]],[[.1,.9],[0,1]]];v=[2,4]\ngot,acts=optimal_backup(r,p,v,.5)\nassert len(got)==2 and all(abs(a-b)<1e-9 for a,b in zip(got,[2.8,3.9])) and acts==[1,0]"},
           {"name": "joint_max", "description": "即时最大与后继最大属于不同动作，不能拆开 max。",
            "code": "got,acts=optimal_backup([[5,0],[0]],[[[1,0],[0,1]],[[0,1]]],[0,10],.5)\nassert got==[5,5] and acts==[0,0]"},
           {"name": "synchronous", "description": "更新第二状态不能读刚更新的第一状态。",
            "code": "got,acts=optimal_backup([[0],[0]],[[[0,1]],[[1,0]]],[1,3],1)\nassert got==[3,1] and acts==[0,0]"},
           {"name": "tie_gamma_zero", "description": "γ=0 时只比奖励；相等时返回动作 0。",
            "code": "got,acts=optimal_backup([[2,2]],[[[1],[1]]],[999],0)\nassert got==[2] and acts==[0]"},
           {"name": "empty", "description": "空模型应返回两个空列表。",
            "code": "assert optimal_backup([],[],[],.5)==([],[])"},
           {"name": "no_mutation", "description": "输入及旧价值不被修改。",
            "code": "import copy\nr=[[1]];p=[[[1]]];v=[2];before=copy.deepcopy((r,p,v));out=optimal_backup(r,p,v,.9)\nassert (r,p,v)==before and out[0] is not v"},
       ],
       rubric=[{"criterion": "完整 Q=R+γPV 计算与同步备份。", "points": 7},
               {"criterion": "逐状态 argmax，包含并列时第一个动作。", "points": 5},
               {"criterion": "空模型、无输入副作用、标准库。", "points": 3}],
       _solution=_solution_optimal()),
]

TOTAL_SCORE = sum(q["max_score"] for q in QUESTIONS)
assert len(QUESTIONS) == 12 and TOTAL_SCORE == 100
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
    except KeyError:
        raise bank.UnknownQuestion("unknown question id: %r" % (question_id,))


def public_payload():
    return {
        "reference": REFERENCE, "conventions": CONVENTIONS,
        "presentation": PRESENTATION, "industry_case": INDUSTRY_CASE,
        "industry_case_question_id": "q09", "count": len(QUESTIONS),
        "total_score": TOTAL_SCORE, "scoring": dict(SCORING),
        "questions": [bank.public_question(q) for q in QUESTIONS],
    }


def grade(question_id, answer):
    q = _get(question_id)
    if q["type"] == "choice":
        index = bank._normalize_choice(answer, q["options"])
        correct = index == q["_answer"]
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[q["_answer"]]
        return {
            "correct": correct, "score": q["max_score"] if correct else 0,
            "max_score": q["max_score"],
            "feedback": ("判断正确。" if correct else
                         f"选项 {letter} 才符合本节定义。{q['feedback_hint']}"),
            "expected": letter,
        }
    if q["type"] == "numeric":
        got = bank._as_float_list(answer)
        expected = q["_expected"]
        wanted = list(expected) if q["answer_type"] == "list" else [expected]
        if len(got) != len(wanted):
            raise bank.InvalidAnswer(f"答案个数不符，期望 {len(wanted)} 个数")
        correct = all(math.isclose(a, b, rel_tol=q["tolerance"], abs_tol=q["tolerance"])
                      for a, b in zip(got, wanted))
        return {
            "correct": correct, "score": q["max_score"] if correct else 0,
            "max_score": q["max_score"],
            "feedback": "数值正确。" if correct else f"数值不正确。{q['feedback_hint']}",
            "expected": expected,
        }
    raise bank.NotAutoGradable("解释题请自评；代码题只在浏览器运行公开测试。")


def reveal(question_id):
    q = _get(question_id)
    if q["type"] == "open":
        return {"id": q["id"], "type": "open", "max_score": q["max_score"],
                "self_assessment": True, "rubric": [dict(x) for x in q["rubric"]],
                "model_answer": q["model_answer"]}
    if q["type"] == "code":
        return {"id": q["id"], "type": "code", "max_score": q["max_score"],
                "self_assessment": False, "entry_point": q["entry_point"],
                "rubric": [dict(x) for x in q["rubric"]],
                "reference_solution": q["_solution"]}
    raise bank.NotRevealable("选择或计算题请先提交答案。")
