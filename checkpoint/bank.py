"""Question bank for the local RL checkpoint (browser-side, EasyRL v1.0.6 §2.1–§2.2.2).

Scope is intentionally frozen at concept/numeric/open-ended questions about
Markov processes, Markov reward processes, returns and the MRP Bellman equation,
plus two pure-Python coding exercises that run in a browser Pyodide worker.

This module is dependency-free (stdlib only) and performs NO server-side
execution of user code. Objective questions are auto-graded here; open-ended
questions are self-assessed from a rubric; coding exercises are graded entirely
in the browser.
"""

import json
import math


class BankError(Exception):
    """Base class for question-bank errors."""


class UnknownQuestion(BankError):
    """Raised when a question id does not exist."""


class NotAutoGradable(BankError):
    """Raised when an auto-grade request targets a non-objective question."""


class NotRevealable(BankError):
    """Raised when a reveal request targets an auto-gradable objective question."""


class InvalidAnswer(BankError):
    """Raised when a submitted answer cannot be parsed for its question type."""


REFERENCE = "EasyRL v1.0.6 §2.1–§2.2.2"

CONVENTIONS = (
    "约定（与 EasyRL v1.0.6 §2.1–§2.2.2 一致）：奖励序列写作 r_{t+1}, r_{t+2}, …，"
    "回报 G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + …，并满足递归 G_t = r_{t+1} + γ·G_{t+1}；"
    "本节不涉及时序差分与 MDP 控制。MRP 中 R(s) 是状态 s 的即时期望奖励，即 R(s) = E[r_{t+1} | S_t = s]，"
    "价值函数满足贝尔曼方程 V(s) = R(s) + γ·Σ_{s'} P(s'|s)·V(s')，其中转移概率按行归一（每行之和为 1）；"
    "“同步备份”指用同一份旧价值 V 一次性算出全部新价值 V' = R + γ·P·V（先用旧值算完再整体写回）。"
)

PRESENTATION = {
    "eyebrow": "从定义走到可计算",
    "intro": "不急着进入 PPO。先用概念判断、手算、解释和两段小代码，确认经典 RL 的第一块地基。",
    "field_notes": [
        {
            "label": "01 / RETURN",
            "formula": "Gₜ = rₜ₊₁ + γGₜ₊₁",
            "tex": r"G_t=r_{t+1}+\gamma G_{t+1}",
            "description": "从后往前算。第一个奖励不打折，后续奖励每晚一步乘一次 γ。",
        },
        {
            "label": "02 / VALUE",
            "formula": "V(s) = E[Gₜ | Sₜ = s]",
            "tex": r"V(s)=\mathbb E[G_t\mid S_t=s]",
            "description": "一次轨迹的回报是样本；状态价值是所有可能轨迹回报的期望。",
        },
        {
            "label": "03 / BELLMAN",
            "formula": "V = R + γPV",
            "tex": r"V=R+\gamma PV",
            "description": "P 的每一行对应一个当前状态，元素是去往各后继状态的概率。",
        },
    ],
    "note_footer": "仅覆盖当前阅读进度。遇到尚未学过的算法，可以先跳过。",
}


def _starter_compute_return():
    return "\n".join([
        "def compute_return(rewards, gamma):",
        '    """计算从最初时刻出发的折扣回报 G_0。',
        "",
        "    rewards: 按时间顺序排列的奖励序列 [r_1, r_2, r_3, ...]",
        "    gamma:   折扣因子, 0 <= gamma <= 1",
        "    返回:    G_0 = rewards[0] + gamma*rewards[1] + gamma**2*rewards[2] + ...",
        "    说明:    空序列返回 0.0；建议从最后一个奖励开始倒序递推。",
        '    """',
        "    # TODO: 仿照 G_t = r_{t+1} + gamma * G_{t+1} 倒序计算",
        "    return 0.0",
    ])


def _solution_compute_return():
    return "\n".join([
        "def compute_return(rewards, gamma):",
        "    running_return = 0.0",
        "    for reward in reversed(rewards):",
        "        running_return = reward + gamma * running_return",
        "    return running_return",
    ])


def _mc_gridworld_demo():
    """Interactive Monte Carlo evaluation demo for the 2x2 gridworld.

    All data here is a teaching construction: the 2x2 grid with a terminal
    +1 cell is invented for this page, while the exact values come from
    solving the uniform-policy Bellman equations at each gamma by the same
    value-iteration sweep shown in class. No external environment code runs.
    """
    transitions = {
        # 撞墙留在原地；只有走到 s4 才给 +1，然后回合结束。
        0: {0: [0, 0.0], 1: [2, 0.0], 2: [0, 0.0], 3: [1, 0.0]},
        1: {0: [1, 0.0], 1: [3, 1.0], 2: [0, 0.0], 3: [1, 0.0]},
        2: {0: [0, 0.0], 1: [2, 0.0], 2: [2, 0.0], 3: [3, 1.0]},
    }
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    exact = {}
    for gamma in gammas:
        values = [0.0, 0.0, 0.0, 0.0]
        for _ in range(2000):
            updated = values[:]
            for state, actions in transitions.items():
                total = 0.0
                for outcome in actions.values():
                    nxt, reward = outcome
                    total += reward + gamma * values[nxt]
                updated[state] = 0.25 * total
            values = updated
        exact[str(gamma)] = [round(values[i], 3) for i in range(3)]
    # 键统一为字符串，保证 HTTP JSON 与直接调用 public_payload() 一致。
    demo = {
        "kind": "mc-gridworld",
        "title": "蒙特卡洛实验场：2×2 网格上把 V(s) 跑出来",
        "data_notice": (
            "这个 2×2 网格是教学构造：s4 是终点，走到 s4 得 +1，撞墙原地不动且奖励 0，"
            "策略是上下左右各 25% 的均匀随机策略。下方“精确解”按同一组规则用值迭代 2000 轮"
            "算到收敛，供蒙特卡洛估计对照；全部计算只发生在你浏览器的本页脚本里。"
        ),
        "reading_guide": (
            "先看智能体一局怎么走：撞墙原地不动，只有进 s4 拿 +1。整局结束后才倒序算 G "
            "并更新价值——这正是你 compute_return 的用法。再观察奖励怎样一局一局从 s4 "
            "扩散到 s2/s3、再到 s1，最后点“快进”，看估计值慢慢逼近精确解。"
        ),
        "policy_note": "策略 π：上下左右各 25%，撞墙原地不动",
        "episode_cap": 80,
        "default_gamma": 0.9,
        "default_speed": 900,
        "speeds": [
            {"label": "0.5×", "ms": 1600},
            {"label": "1×", "ms": 900},
            {"label": "2×", "ms": 450},
            {"label": "4×", "ms": 220},
        ],
        "bulk_steps": [100, 1000, 10000],
        "states": ["s1", "s2", "s3", "s4"],
        "terminal": 3,
        "actions": ["↑", "↓", "←", "→"],
        "transitions": {
            str(state): {str(action): outcome for action, outcome in actions.items()}
            for state, actions in transitions.items()
        },
        "exact": exact,
    }
    # 往返一次，确保前端拿到的结构与 JSON 完全一致。
    return json.loads(json.dumps(demo))


INDUSTRY_CASE = {
    "title": "真实代码路径：四条回答怎样变成回报张量",
    "project": "verl",
    "commit": "a2ad9f6",
    "scope": (
        "追踪本地 verl 的 v1 流式规则奖励路径：GSM8K 风格的数据字段 → agent loop → "
        "experimental/reward_loop 的 NaiveRewardManager → rm_scores → v1 trainer → REINFORCE++ 的 returns。"
        "条件：reward_model.enable=False、未指定自定义奖励函数、use_kl_in_reward=False、"
        "没有额外奖励整形，并显式选择 adv_estimator=reinforce_plus_plus；不是说默认配置如此。"
    ),
    "data_notice": (
        "下方四条问答与位置切分为教学构造，不是下载的 GSM8K 样本、训练日志或真实 tokenizer ID；"
        "但 data_source / reward_model 字段结构、严格的“#### 数字”评分、末位奖励落点与回报循环"
        "均按所列源码逐项核对。只有末尾 padding，没有工具调用；小批量 B=4、回答宽度 T=4。"
    ),
    "reading_guide": (
        "先只跟踪 A：题目“2 + 2 = ?”、真值“4”、生成回答“先算，#### 4”。"
        "这三样不是同时出现：题目和真值来自数据行，回答稍后由模型生成。"
        "每一步先看“收到什么 → 做了什么 → 交出什么”，再逐行看摘录；"
        "最后用 B（短而正确）、C（数字错）、D（数字对但格式错）检查边界。"
    ),
    "stages": [
        {
            "number": "01",
            "title": "数据集预处理：把监督答案变成 ground_truth",
            "explanation": (
                "GSM8K 预处理器读取问题、从原答案提取最终数值，并写入 data_source、prompt 和"
                "reward_model.ground_truth。训练时模型生成 response；原数据行本身没有生成结果。"
            ),
            "path": "verl/examples/data_preprocess/gsm8k.py",
            "lines": "61–77",
            "code": (
                'solution = extract_solution(answer_raw)\n'
                'data = {\n'
                '    "data_source": data_source,\n'
                '    "prompt": [{"role": "user", "content": question}],\n'
                '    "ability": "math",\n'
                '    "reward_model": {"style": "rule", "ground_truth": solution},\n'
                '}'
            ),
            "code_note": "结构化摘录：prompt 在源码中分行书写，extra_info 字段为简明起见略去；data_source 在 47 行定义为 openai/gsm8k。",
            "input": "GSM8K 原始 question 和 answer_raw（后者末尾含“#### 4”）；还没有模型回答。",
            "output": "A 的记录含 prompt、data_source=openai/gsm8k、reward_model.ground_truth='4'；还没有 response。",
            "trace": "例如 answer_raw='计算过程 #### 4' → extract_solution 得到字符串 '4'。prompt 发给模型；ground_truth 留待评分。B/C/D 的真值均为 '2'。",
            "line_notes": [
                "预处理的 extract_solution 在原始答案里匹配“#### 数字”（若有多处则取首个匹配）；本例只有末尾一处，A 的 solution 是字符串 '4'，不是模型输出。",
                "建立一条新的数据记录；以下键值是在给这条记录贴标签。",
                "标识题目来源。后面评分路由正是凭 openai/gsm8k 选择 GSM8K 规则；这不是一个即时奖励。",
                "把问题封装成 user 消息交给生成模型；本页只写出问题，真实预处理代码还附有输出格式指令。",
                "把任务标为 math；本条案例的后续演算不依赖此字段。",
                "style='rule' 表示规则奖励的元数据；ground_truth 是前面取出的 '4'，供稍后与生成答案比较。",
                "这条记录打包完成。此刻没有 response，也没有 reward_score。",
            ],
        },
        {
            "number": "02",
            "title": "Agent loop：将数据字段和生成的 response 送到奖励 worker",
            "explanation": (
                "在无单独奖励模型的流式路径，rollout 按每条回答构建 DataProto：张量里有"
                " prompts / responses / attention_mask，非张量部分从 kwargs 带入 data_source、reward_model 等。"
                "异步 worker 返回每条标量 reward_score，而不是直接返回逐 token 价值函数。"
            ),
            "path": "verl/verl/experimental/agent_loop/agent_loop.py",
            "lines": "997–1027",
            "code": (
                'n = len(outputs)\n'
                'batch = TensorDict({\n'
                '    "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),\n'
                '    "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),\n'
                '    "attention_mask": torch.nn.utils.rnn.pad_sequence(all_attention_mask, batch_first=True, padding_value=0),\n'
                '}, batch_size=n)\n'
                'non_tensor_batch = {\n'
                '    **{k: np.array([v] * n) for k, v in kwargs.items()},\n'
                '    "__num_turns__": np.array([o.num_turns for o in outputs]),\n'
                '    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),\n'
                '    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),\n'
                '    "response_len": np.array([len(o.response_ids) for o in outputs]),\n'
                '}\n'
                'data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)\n'
                'result = await selected_reward_loop_worker_handle.compute_score.remote(data)\n'
                'final_output.reward_score = result["reward_score"]'
            ),
            "code_note": "结构化、非连续摘录：TensorDict 构造在真实源码中还包括 input_ids 与 position_ids，attention_mask 的 pad_sequence 原代码跨多行；DataProto 在原源码中跨多行；worker 选择行已省略。这不是可直接运行的一段代码。",
            "input": "阶段 01 的 prompt、data_source、ground_truth；模型随后生成 A='先算，#### 4' 等回答。",
            "output": "把响应 token、attention_mask 与题目元数据装入 DataProto 送给 reward worker；评分返回后存为每条回答的标量 reward_score。",
            "trace": "A 的回答在演示中占 4 个位置；B/C/D 各占 2 个并右补两个空位。kwargs 携带真值，避免评分器只拿到回答、却不知道应与什么比较。",
            "line_notes": [
                "n 表示这一次 agent loop 得到的输出数；下面 TensorDict 按它声明当前小批次大小，不必等同于整次训练的 B=4。",
                "开始建立一批样本的数值张量容器 TensorDict；本页只摘出理解评分所需的字段。",
                "把各条 prompt token 序列补齐为同宽，batch_first=True 表示第一维是样本行。",
                "把每条生成回答的 token ID 补齐到同宽；本页演示的回答宽度为 T=4。",
                "把哪些 token 真正存在也补齐成同形状的 attention_mask；有效=1，padding=0。",
                "标注此 TensorDict 当前输出批次数为 n；不应直接把这里的 n 误认为全局 B=4。",
                "开始建立随回答一起发送的非张量字段字典；文本来源、标准答案这类元数据不能直接当作浮点奖励矩阵。",
                "把 kwargs 里的每个字段复制给本次 outputs 中的 n 条结果；在此教学案例里关注 data_source 与 reward_model。此处的 n 是当前 agent-loop 输出数，不直接等于全局 B=4。",
                "记录每个输出经历了多少轮；本页单轮、无工具调用，不用它计算回报。",
                "收集工具产生的附加信息；本页没有工具调用。",
                "逐个记录问题 prompt 的 token 数，以便区分问题和回答。",
                "逐个记录回答的有效 token 数；长度不同，所以稍后要补齐。",
                "非张量字段字典结束。到此还没有分数。",
                "把张量 batch 和非张量元数据打包在一起，形成交给评分 worker 的 DataProto。",
                "异步调用所选 reward worker 的 compute_score，把 DataProto 交出去并等待其返回；不是在这里计算奖励。",
                "从返回字典中取 reward_score，挂在本条最终输出上。其内部如何算出来，要看接下来的阶段 03、04。",
            ],
        },
        {
            "number": "03",
            "title": "NaiveRewardManager：解码、读取真值、调用规则评分",
            "explanation": (
                "reward worker 通过配置加载 experimental 目录下注册为 naive 的 manager。"
                "run_single 只取有效 response token，解码后按 data_source / ground_truth 调用 compute_score。"
                "注意：这不是 workers/reward_manager/naive.py 中旧版 manager。"
            ),
            "path": "verl/verl/experimental/reward_loop/reward_manager/naive.py",
            "lines": "35–43, 54–83, 97–99",
            "code": (
                'response_ids = data_item.batch["responses"]\n'
                'response_length = response_ids.shape[-1]\n'
                'valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()\n'
                'valid_response_ids = response_ids[:valid_response_length]\n'
                'data_source = data_item.non_tensor_batch["data_source"]\n'
                'ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]\n'
                'response_str = await self.loop.run_in_executor(\n'
                '    None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)\n'
                ')\n'
                'result = await self.loop.run_in_executor(\n'
                '    None,\n'
                '    lambda: self.compute_score(\n'
                '        data_source=data_source,\n'
                '        solution_str=response_str,\n'
                '        ground_truth=ground_truth,\n'
                '        extra_info=extra_info,\n'
                '        **extra_reward_kwargs,\n'
                '    ),\n'
                ')\n'
                'return {"reward_score": reward, "reward_extra_info": reward_extra_info}'
            ),
            "code_note": "非连续节选：中间展示的是默认 compute_score 为同步函数时的 else 分支；省略了异步自定义函数分支、result→score→reward 的转换及额外信息处理。extra_reward_kwargs 在无 reward router 时为空字典；这不是可直接运行的完整函数。",
            "input": "阶段 02 发来的 DataProto：回答 token/attention_mask，以及非张量的 data_source 和 ground_truth。",
            "output": "把有效回答解码为 response_str，调用评分函数，再返回单个 reward_score；本页 A=1。",
            "trace": "若 B 的回答槽位是 [有效,有效,pad,pad]，attention_mask 对应的回答段求和得 2；只解码前两个有效 token，得到 '#### 2'，不会把 pad 误当作正文。",
            "line_notes": [
                "从当前数据项拿出回答 token ID 序列；这里还是数字 ID，不是给人看的文字。",
                "取得补齐后的回答槽位宽度，便于在 attention_mask 中定位回答部分。",
                "截取 attention_mask 的回答段并求和，得到真正有效的回答 token 数；B 是 2，不是补齐后的 4。",
                "只保留前面这么多个有效回答 ID，抛开右侧 padding。",
                "取出来源标签；评分器需要知道该套用哪一种题目的规则。",
                "从 reward_model 元数据中取出原题标准答案；A 为字符串 '4'。",
                "把解码工作安排在线程执行器上，不占用当前异步循环；返回的 response_str 是整段回答文字。",
                "tokenizer.decode 只解码有效 ID，并跳过特殊 token；本页 A 的示意文字是 '先算，#### 4'。",
                "解码调用结束；得到 response_str，下一段才会真正评分。",
                "在默认同步评分函数的分支，仍由线程执行器代为计算并等待结果；只截取了该分支。",
                "线程执行器的第一个位置参数，此处不另指定执行器。",
                "传给线程执行器一个稍后执行的 compute_score 调用。",
                "传入 data_source='openai/gsm8k'，让评分路由找到对应规则。",
                "传入刚解码的完整回答文字，而不是 token ID。",
                "传入题目自带的真值字符串，如 A 的 '4'。",
                "附加元数据；本页演算不使用它。",
                "传入其他可选参数；本页没有 reward router，因此这些参数为空。",
                "评分函数的参数列表结束。",
                "线程执行器调用结束，结果保存在 result；源码接着把 result 转成 score，再赋给 reward。",
                "把单个 reward 数值与附加信息交回 agent loop；这里还不是逐 token 张量。",
            ],
        },
        {
            "number": "04",
            "title": "评分路由：严格解析“#### 数字”并比较真值",
            "explanation": (
                "default_compute_score 看到 data_source=openai/gsm8k 后进入 gsm8k.compute_score。"
                "严格模式只识别回答末段中的“#### 4”形式；写“答案是 2”虽数字正确但未满足格式，仍得 0。"
                "本页四行的 1 / 1 / 0 / 0 由同一规则算出，不是凭空填分。"
            ),
            "path": "verl/verl/utils/reward_score/gsm8k.py",
            "lines": "20–36, 52–72",
            "source_label": (
                "verl/verl/utils/reward_score/__init__.py:44–47；"
                "verl/verl/utils/reward_score/gsm8k.py:20–36, 52–72"
            ),
            "code": (
                'if data_source == "openai/gsm8k":\n'
                '    from . import gsm8k\n'
                '    res = gsm8k.compute_score(solution_str, ground_truth)\n'
                '# ↓ 切换到 gsm8k.py / extract_solution（非连续摘录）\n'
                'solutions = re.findall("#### (\\\\-?[0-9\\\\.\\\\,]+)", solution_str)\n'
                '# ↓ 切换到 gsm8k.py / compute_score（非连续摘录）\n'
                'answer = extract_solution(solution_str=solution_str, method=method)\n'
                'if answer is None:\n'
                '    return 0\n'
                'else:\n'
                '    if answer == ground_truth:\n'
                '        return score\n'
                '    else:\n'
                '        return format_score'
            ),
            "code_note": "前三行来自 reward_score/__init__.py 的路由；后面是 gsm8k.py 的非连续摘录，两段不可直接拼接执行。严格提取还会只查看回答末尾 300 个字符，取最后一个匹配并去掉逗号；默认 score=1、format_score=0。",
            "input": "解码文字 response_str、来源 openai/gsm8k、标准答案 ground_truth。",
            "output": "整条回答一个分数：A/B/C/D = 1/1/0/0，不是四个 token 各有一个分数。",
            "trace": "A 提取 '4' = 真值 '4' → 1；B 提取 '2' = '2' → 1；C 提取 '3' ≠ '2' → 0；D 虽写了 2，却没有“#### 空格 数字”→ 无匹配 → 0。",
            "line_notes": [
                "默认评分器检查来源标签；只有 openai/gsm8k 才进入下面的 GSM8K 分支。",
                "延迟导入该题型的评分模块；这一步尚未比较答案。",
                "把回答文字和真值传给 gsm8k.compute_score；之后转入另一个文件。",
                "阅读分隔符，不是原代码：后面的正则来自 gsm8k.py 的答案提取函数。",
                "严格模式先用正则找 '#### ' 后的数字，可匹配多处；真正的 extract_solution 还限制在回答末尾 300 字符内，并取最后一处。",
                "阅读分隔符，不是原代码：之后展示的是 gsm8k.py 的评分函数，不与上面一行直接相连。",
                "调用 extract_solution 取得规范化后的答案字符串（如 '4'）；默认 method='strict'，不是见到任意数字就给分。",
                "如果找不到符合格式的答案（如 D），就进入无答案分支。",
                "无格式匹配直接返回 0，即便文字里另有正确数字也一样。",
                "找到了格式正确的候选答案，进入比较分支。",
                "比较字符串答案与 ground_truth；A/B 相同，C 不同。",
                "一致则返回默认 score=1.0；每条完整回答只产生一个数。",
                "不一致进入另一分支。",
                "返回默认 format_score=0.0；C 的 '#### 3' 虽格式符合，却与真值不符。",
            ],
        },
        {
            "number": "05",
            "title": "批次对齐：标量落在最后一个有效回答位置",
            "explanation": (
                "agent loop 把响应右侧补齐到相同宽度，构造 response_mask。"
                "批次后处理按 attention_mask 算有效长度，在每行末个有效位置写 reward_score，"
                "其余位置为零；这一步得到形状 [B,T] 的 rm_scores。"
            ),
            "path": "verl/verl/experimental/agent_loop/agent_loop.py",
            "lines": "785–798, 1091–1097",
            "code": (
                'scores = [input.reward_score for input in inputs]\n'
                'if all(score is not None for score in scores):\n'
                '    prompt_length = prompt_ids.size(1)\n'
                '    response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1\n'
                '    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)\n'
                '    rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)\n'
                '    batch["rm_scores"] = rm_scores'
            ),
            "code_note": "连续摘录 1091–1097 行；response_mask 的右侧补齐构造见 785–798 行。本例没有中间工具 token。",
            "input": "四条回答的标量分数 [1,1,0,0]，以及各自的有效长度 [4,2,2,2]。",
            "output": "同宽的 rm_scores [4,4]：A=[0,0,0,1]，B=[0,1,0,0]，C/D=[0,0,0,0]。",
            "trace": "B 的回答 mask=[1,1,0,0]，有效长之和 2，末位索引为 2−1=1；于是 B 的 1 被写在第 1 列，不是最后一个 padding 槽第 3 列。",
            "line_notes": [
                "从每条生成输出收集 reward_score；四个标量形成 [1,1,0,0]，不是逐 token 分数。",
                "只有所有输出都有分数才组装这个奖励张量；否则不能以缺失值冒充 0。",
                "取得 prompt 的补齐宽度；attention_mask 同时覆盖 prompt 和 response，需要先越过 prompt。",
                "截取回答段，逐行求有效位数，再减 1 得到零起始的最后有效索引；A=3、B/C/D=1。",
                "按 response_mask 造一个全零 float 张量；形状是 [B,T]=[4,4]。",
                "按行号与该行的末位索引配对写分数：A 的 1 写到 col3，B 的 1 写到 col1；零分行仍全零。",
                "把这个矩阵保存在 batch['rm_scores'] 中，后续 trainer 才能读取。",
            ],
        },
        {
            "number": "06",
            "title": "v1 trainer：分数变成逐 token 奖励",
            "explanation": (
                "流式规则路径在采样时已有 rm_scores，因此跳过 colocated RM 的再次计算。"
                "trainer 从队列取 response_mask / rm_scores；关闭 KL 奖励惩罚时，"
                "token_level_rewards 与 token_level_scores 相同。其他惩罚/修正不属于本例。"
            ),
            "path": "verl/verl/trainer/ppo/v1/trainer_base.py",
            "lines": "811–814, 1936–1940, 1987–1994",
            "code": (
                'response_mask = data["response_mask"]\n'
                'data = DataProto(batch=data.to_padded_tensor())\n'
                'data.batch["token_level_scores"] = data.batch["rm_scores"]\n'
                'if self.config.algorithm.use_kl_in_reward:\n'
                '    data, kl_metrics = apply_kl_penalty(\n'
                '        data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty\n'
                '    )\n'
                'else:\n'
                '    data.batch["token_level_rewards"] = data.batch["token_level_scores"]'
            ),
            "code_note": "两个位置的真实节选，非连续可运行代码；关闭 KL 的分支才使 rewards 与 scores 相等。",
            "input": "采样批次中的 response_mask 和 rm_scores；A=[0,0,0,1]、B=[0,1,0,0]。",
            "output": "在本例关闭 KL 惩罚与额外整形时，token_level_rewards 与 rm_scores 数值相同；这里只是把整条评分安置为逐位置奖励。",
            "trace": "名字分三层：reward_score 是一条回答的标量；rm_scores 是放到最后有效位的矩阵；token_level_rewards 是供回报算法读取的逐位置奖励。此例它们相关，但不是同一种数据结构。",
            "line_notes": [
                "取回答 mask，标清哪些位置可参与后续计算；与下行的 data 转换是相邻但不等于完整 trainer 函数。",
                "把队列取到的批次整理为带填充的 DataProto 张量形式，便于下游按矩阵运算。",
                "把已有 rm_scores 赋给 token_level_scores；数值没有改，字段表示未经 KL 调整的分数。",
                "如果 use_kl_in_reward 为真，会进入惩罚分支；本案例配置为 False，所以不会走它。",
                "调用 KL 惩罚函数的起始行；本例跳过。",
                "传入 KL 控制器和惩罚形式；仍是本例未执行的分支。",
                "KL 调用结束；本例不执行。",
                "本例执行 else 分支；还假定没有额外奖励整形等改写。",
                "把 token_level_scores 赋给 token_level_rewards；A/B 分别保持 [0,0,0,1]、[0,1,0,0]。",
            ],
        },
        {
            "number": "07",
            "title": "REINFORCE++：倒序得到每个有效位置的回报",
            "explanation": (
                "选择 adv_estimator=reinforce_plus_plus 后才进入此路径。"
                "循环用 gamma=config.gamma 从右向左递推 returns，response_mask 阻断 padding。"
                "此页只追踪 returns，不把随后的 masked_whiten 优势值误叫成回报。"
            ),
            "path": "verl/verl/trainer/ppo/core_algos.py",
            "lines": "715–751",
            "code": (
                'returns = torch.zeros_like(token_level_rewards)\n'
                'running_return = 0\n'
                'for t in reversed(range(token_level_rewards.shape[1])):\n'
                '    running_return = token_level_rewards[:, t] + gamma * running_return\n'
                '    returns[:, t] = running_return\n'
                '    running_return = running_return * response_mask[:, t]'
            ),
            "code_note": "连续摘录 739–746 行（略去 EOS 注释）；默认配置为 gamma=1.0、adv_estimator=gae，本例在滑块里演示改动 gamma。",
            "input": "token_level_rewards [4,4]、response_mask [4,4]、显式选择 REINFORCE++ 的配置；演示 γ=0.9。",
            "output": "returns [4,4]：A=[0.729,0.81,0.9,1]，B=[0.9,1,0,0]，C/D 全零。",
            "trace": "默认演算 γ=0.9：A 从最右侧奖励 1 开始，往左每走一步乘 0.9：1→0.9→0.81→0.729。B 右侧是 padding，先清零；在 col1 遇到 1，再到 col0 得 0.9。下方改动 γ 时，以动态表为准。四行各自独立，不会互相串分。",
            "line_notes": [
                "建立与逐位置奖励同形状的全零 returns 张量；每个有效回答位置最终有一个采样回报。",
                "初始化倒序累积量为 0；第一次处理一列后变为每行一个数的向量。",
                "从最右列走到最左列，因为当前位置的回报依赖右边下一位置已经算出的回报。",
                "一次处理 batch 全部行的第 t 列：当前奖励 + γ×右侧积累值；A 在 t=2 算出 0+0.9×1=0.9。",
                "把本轮每行算出的值写入 returns 的第 t 列；它是当前前缀以后实际生成轨迹的折扣和。",
                "用当前位置是否有效清掉 padding 行的累积值，以免从无效位置继续倒推；列维度批量并行、行之间本来也不会互相混合。源码先写 returns 再做这个乘法；本例只有右侧 padding 且 pad 奖励为 0。",
            ],
        },
    ],
    "samples": [
        {
            "id": "A",
            "data_source": "openai/gsm8k",
            "prompt": "2 + 2 = ?",
            "ground_truth": "4",
            "response": "先算，#### 4",
            "positions": ["先算", "，", "####", " 4"],
            "valid_length": 4,
            "score": 1,
        },
        {
            "id": "B",
            "data_source": "openai/gsm8k",
            "prompt": "1 + 1 = ?",
            "ground_truth": "2",
            "response": "#### 2",
            "positions": ["####", " 2", "<pad>", "<pad>"],
            "valid_length": 2,
            "score": 1,
        },
        {
            "id": "C",
            "data_source": "openai/gsm8k",
            "prompt": "1 + 1 = ?",
            "ground_truth": "2",
            "response": "#### 3",
            "positions": ["####", " 3", "<pad>", "<pad>"],
            "valid_length": 2,
            "score": 0,
        },
        {
            "id": "D",
            "data_source": "openai/gsm8k",
            "prompt": "1 + 1 = ?",
            "ground_truth": "2",
            "response": "答案是 2",
            "positions": ["答案", "是 2", "<pad>", "<pad>"],
            "valid_length": 2,
            "score": 0,
        },
    ],
}


def _starter_bellman_backup():
    return "\n".join([
        "def bellman_backup(rewards, transitions, values, gamma):",
        '    """执行一次同步贝尔曼备份 V\' = R + gamma * P V。',
        "",
        "    rewards:     长度 N 的即时期望奖励 R(s_i)",
        "    transitions: N x N 的转移矩阵, transitions[i][j] = P(s_j | s_i), 每行之和为 1",
        "    values:      长度 N 的当前价值估计 V(s_i)",
        "    gamma:       折扣因子",
        "    返回:        长度 N 的新价值向量 V'(s_i)",
        "    说明:        必须使用同一份旧 values 算出全部新值（同步/不可边算边写回），",
        "                 且不得原地修改传入的 values。",
        '    """',
        "    # TODO: 在这里实现（只用标准库）",
        "    return list(values)",
    ])


def _solution_bellman_backup():
    return "\n".join([
        "def bellman_backup(rewards, transitions, values, gamma):",
        "    n = len(values)",
        "    new_values = [0.0] * n",
        "    for i in range(n):",
        "        expected = 0.0",
        "        row = transitions[i]",
        "        for j in range(n):",
        "            expected += row[j] * values[j]",
        "        new_values[i] = rewards[i] + gamma * expected",
        "    return new_values",
    ])


QUESTIONS = [
    {
        "id": "q01",
        "type": "choice",
        "category": "concept",
        "title": "回报（Return）的定义",
        "prompt": (
            "在马尔可夫奖励过程（MRP）中，从时刻 t 出发的回报 G_t 的精确定义是下列哪一项？"
            "（r_{t+1} 表示 t 时刻之后获得的第一个奖励）"
        ),
        "max_score": 5,
        "grading_mode": "auto",
        "options": [
            "G_t = r_{t+1} + r_{t+2} + r_{t+3} + …，即所有未来奖励的等权求和",
            "G_t = γ·r_{t+1} + γ²·r_{t+2} + …，即从 γ 的一次幂开始加权",
            "G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + …，即未来奖励按 γ 的幂次折扣后求和",
            "G_t = R(s_t) + γ·R(s_{t+1}) + …，即对即时奖励函数逐状态求和",
        ],
        "_answer": 2,
    },
    {
        "id": "q02",
        "type": "choice",
        "category": "concept",
        "title": "折扣因子 γ 的作用",
        "prompt": "关于折扣因子 γ ∈ [0, 1]，下列说法正确的是：",
        "max_score": 5,
        "grading_mode": "auto",
        "options": [
            "γ 越大，智能体越只看重眼前奖励",
            "γ 越小，未来奖励的权重衰减越慢",
            "γ 越小，智能体越看重即时奖励；γ → 1 时更看重长期回报",
            "γ 必须严格等于 1，否则回报一定会发散",
        ],
        "_answer": 2,
    },
    {
        "id": "q03",
        "type": "choice",
        "category": "concept",
        "title": "马尔可夫性质",
        "prompt": "“马尔可夫性质”指的是：",
        "max_score": 5,
        "grading_mode": "auto",
        "options": [
            "下一状态只由当前状态（以及当前动作）决定，与更早的历史无关",
            "下一状态由全部历史轨迹共同决定",
            "奖励只取决于初始状态",
            "状态转移概率必须随时间保持不变——这就是马尔可夫性本身",
        ],
        "_answer": 0,
    },
    {
        "id": "q04",
        "type": "choice",
        "category": "concept",
        "title": "状态价值函数 V(s) 的含义",
        "prompt": "在 MRP 中，状态价值函数 V(s) 的含义是：",
        "max_score": 5,
        "grading_mode": "auto",
        "options": [
            "从状态 s 出发所能获得的即时奖励 R(s)",
            "从状态 s 出发、按该过程所得回报的期望，即 V(s) = E[G_t | S_t = s]",
            "状态 s 在整条轨迹中出现的频率",
            "状态 s 的最优动作价值",
        ],
        "_answer": 1,
    },
    {
        "id": "q05",
        "type": "numeric",
        "category": "hand-calculation",
        "title": "手算回报 G_0",
        "prompt": (
            "某轨迹的奖励序列为 r_1=1, r_2=2, r_3=3, r_4=4，折扣因子 γ=0.5。"
            "请计算回报 G_0 = r_1 + γ·r_2 + γ²·r_3 + γ³·r_4。"
        ),
        "max_score": 8,
        "grading_mode": "auto",
        "answer_type": "number",
        "tolerance": 1e-6,
        "_expected": 3.25,
    },
    {
        "id": "q06",
        "type": "numeric",
        "category": "hand-calculation",
        "title": "用递归关系求回报",
        "prompt": (
            "已知 G_{t+1} = 8，γ = 0.9，t+1 时刻拿到的奖励 r_{t+1} = 2。"
            "利用回报的递归关系 G_t = r_{t+1} + γ·G_{t+1} 求 G_t。"
        ),
        "max_score": 8,
        "grading_mode": "auto",
        "answer_type": "number",
        "tolerance": 1e-6,
        "_expected": 9.2,
    },
    {
        "id": "q07",
        "type": "numeric",
        "category": "hand-calculation",
        "title": "单状态贝尔曼备份",
        "prompt": (
            "状态 s 的即时期望奖励 R(s)=1，γ=0.9；从 s 出发以概率 0.5 转移到 s1（V=2），"
            "以概率 0.5 转移到 s2（V=4）。按 V(s) = R(s) + γ·Σ_{s'} P(s'|s)·V(s') 计算 V(s)。"
        ),
        "max_score": 8,
        "grading_mode": "auto",
        "answer_type": "number",
        "tolerance": 1e-6,
        "_expected": 3.7,
    },
    {
        "id": "q08",
        "type": "numeric",
        "category": "hand-calculation",
        "title": "两个状态的同步贝尔曼备份",
        "prompt": (
            "两状态 {A, B} 的 MRP：R(A)=1, R(B)=2, γ=0.5；"
            "P(A|A)=0.5, P(B|A)=0.5；P(A|B)=0.25, P(B|B)=0.75（每行之和为 1）。"
            "当前价值估计 V0(A)=2, V0(B)=4。做一次同步备份 V1 = R + γ·P·V0，"
            "求 V1。答案写成两个数的列表，如 [?, ?]（顺序为 [V1(A), V1(B)]）。"
        ),
        "max_score": 10,
        "grading_mode": "auto",
        "answer_type": "list",
        "tolerance": 1e-6,
        "_expected": [2.5, 3.75],
    },
    {
        "id": "q09",
        "type": "open",
        "category": "application",
        "title": "解释 MRP 的贝尔曼方程",
        "prompt": (
            "请说明 MRP 的贝尔曼方程 V(s) = R(s) + γ·Σ_{s'} P(s'|s)·V(s') 的含义："
            "R(s) 代表什么；为什么它可以写成关于 V 自身的自洽方程；"
            "以及它与回报递归 G_t = r_{t+1} + γ·G_{t+1} 之间的关系。"
        ),
        "max_score": 8,
        "grading_mode": "self",
        "rubric": [
            {"criterion": "指出 R(s) 是状态 s 的即时期望奖励 R(s)=E[r_{t+1}|S_t=s]，而非确定的单步奖励。", "points": 3},
            {"criterion": "利用（条件）期望的线性性，把 E[G_t|S_t=s] 拆成即时项与后续项，得到对后继状态价值的加权求和。", "points": 3},
            {"criterion": "说明贝尔曼方程是回报递归在“取期望”后的展开，把 V 表成 V 自身（以及 R、P）的线性方程。", "points": 2},
        ],
        "model_answer": (
            "R(s) 是在状态 s 处的即时期望奖励。对 G_t = r_{t+1} + γ·G_{t+1} 两边在 S_t=s 下取期望，"
            "左边是 V(s)；右边用期望线性性拆开：E[r_{t+1}|S_t=s]=R(s)，而 E[γ·G_{t+1}|S_t=s] = "
            "γ·Σ_{s'} P(s'|s)·E[G_{t+1}|S_{t+1}=s'] = γ·Σ_{s'} P(s'|s)·V(s')。于是 V(s) = "
            "R(s) + γ·Σ_{s'} P(s'|s)·V(s')，即贝尔曼方程就是回报递归在期望意义下的自洽形式。"
        ),
    },
    {
        "id": "q10",
        "type": "open",
        "category": "application",
        "title": "状态设计与马尔可夫性",
        "prompt": (
            "某电梯调度任务中，如果只用“当前楼层”作为状态，请分析它是否满足马尔可夫性，"
            "并说明你会如何补充状态使其成为马尔可夫状态。"
        ),
        "max_score": 8,
        "grading_mode": "self",
        "rubric": [
            {"criterion": "指出只用当前楼层一般不满足马尔可夫性：下一步转移还与到达方向、目标楼层等历史/上下文有关。", "points": 3},
            {"criterion": "说明应把对未来有影响、且当前可获得的信息纳入状态（如目标楼层、运行方向、门状态、待处理请求。", "points": 3},
            {"criterion": "给出修改后的状态定义，并解释为何此时未来只依赖当前状态（满足马尔可夫性）。", "points": 2},
        ],
        "model_answer": (
            "只看当前楼层通常不满足马尔可夫性，因为下一步“去哪一层”取决于乘客请求、目标楼层和运行方向等未编码信息。"
            "把这些信息补充进状态（例如 状态 =（当前楼层, 运行方向, 目标楼层集合, 门状态））后，"
            "下一步的转移只由这个扩充状态决定，与更早历史无关，从而满足马尔可夫性。"
        ),
    },
    {
        "id": "q11",
        "type": "code",
        "category": "coding",
        "title": "编程：折扣回报的递归计算",
        "prompt": (
            "实现 compute_return(rewards, gamma)：给定按时间顺序的奖励序列，返回从最初出发的折扣回报 G_0。"
            "要求只用 Python 标准库，使用与 EasyRL 一致的约定：rewards[k] 对应 r_{k+1}，G_0 = Σ_k γ^k·rewards[k]，"
            "空序列返回 0.0。尝试按照 G_t = r_{t+1} + γ·G_{t+1} 从后往前递推，"
            "再用浏览器内的公开测试核对。"
        ),
        "max_score": 15,
        "grading_mode": "browser",
        "language": "python",
        "entry_point": "compute_return",
        "starter_code": _starter_compute_return(),
        "tests": [
            {"name": "basic", "description": "γ=0.5 的四步折扣求和应等于 3.25。",
             "code": "assert abs(compute_return([1, 2, 3, 4], 0.5) - 3.25) < 1e-9"},
            {"name": "gamma_zero", "description": "γ=0 时只保留第一个奖励。",
             "code": "assert abs(compute_return([5, 100, 100], 0.0) - 5.0) < 1e-12"},
            {"name": "empty", "description": "空序列返回 0.0。",
             "code": "assert abs(compute_return([], 0.9) - 0.0) < 1e-12"},
            {"name": "long", "description": "20 个 1.0 的序列应与解析式 Σ_{k=0}^{19} γ^k 一致。",
             "code": "assert abs(compute_return([1.0] * 20, 0.9) - sum(0.9 ** k for k in range(20))) < 1e-9"},
            {"name": "delayed_reward", "description": "对应源码桥接：奖励 [0,0,4]、γ=0.5 时 G₀=1。",
             "code": "assert abs(compute_return([0, 0, 4], 0.5) - 1.0) < 1e-12"},
            {"name": "verl_batch", "description": "把工业案例 A/B 的奖励向量截到有效长度：G₀ 分别为 0.729、0.9。",
             "code": ("assert abs(compute_return([0, 0, 0, 1], 0.9) - 0.729) < 1e-12\n"
                      "mask_b = [1, 1, 0, 0]\n"
                      "rewards_b = [0, 1, 0, 0]\n"
                      "assert abs(compute_return(rewards_b[:sum(mask_b)], 0.9) - 0.9) < 1e-12")},
        ],
        "rubric": [
            {"criterion": "实现了 G_0 = Σ γ^k·rewards[k] 的正确求和（首项不加折扣）。", "points": 7},
            {"criterion": "正确处理边界：空序列返回 0.0、γ=0、γ=1。", "points": 4},
            {"criterion": "只用标准库、无副作用、可被公开测试直接调用。", "points": 4},
        ],
        "_solution": _solution_compute_return(),
        "demo": _mc_gridworld_demo(),
    },
    {
        "id": "q12",
        "type": "code",
        "category": "coding",
        "title": "编程：一次同步贝尔曼备份",
        "prompt": (
            "实现 bellman_backup(rewards, transitions, values, gamma)，执行一次同步贝尔曼备份 "
            "V' = R + γ·P·V。transitions[i][j] = P(s_j | s_i)，每行之和为 1。"
            "必须用同一份旧 values 一次性算出全部新值，且不得原地修改传入的 values。"
            "只使用标准库；公开测试可在浏览器 Pyodide 中运行。"
        ),
        "max_score": 15,
        "grading_mode": "browser",
        "language": "python",
        "entry_point": "bellman_backup",
        "starter_code": _starter_bellman_backup(),
        "tests": [
            {"name": "two_state", "description": "2x2 矩阵单次同步备份应得到 [2.5, 3.75]。",
             "code": ("r = [1.0, 2.0]\n"
                      "P = [[0.5, 0.5], [0.25, 0.75]]\n"
                      "v = [2.0, 4.0]\n"
                      "got = bellman_backup(r, P, v, 0.5)\n"
                      "assert abs(got[0] - 2.5) < 1e-9 and abs(got[1] - 3.75) < 1e-9")},
            {"name": "synchronous", "description": "交换矩阵下结果必须用旧值同步计算，顺序无关。",
             "code": ("got = bellman_backup([0.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], [1.0, 3.0], 1.0)\n"
                      "assert abs(got[0] - 3.0) < 1e-9 and abs(got[1] - 1.0) < 1e-9")},
            {"name": "identity", "description": "P=I 时 V' = R + γ·V。",
             "code": ("got = bellman_backup([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], [10.0, 20.0], 0.5)\n"
                      "assert abs(got[0] - 6.0) < 1e-9 and abs(got[1] - 12.0) < 1e-9")},
            {"name": "no_mutation", "description": "不得原地修改传入的 values，应返回新列表。",
             "code": ("v = [0.0]\n"
                      "out = bellman_backup([1.0], [[1.0]], v, 0.9)\n"
                      "assert v == [0.0] and abs(out[0] - 1.0) < 1e-9 and out is not v")},
        ],
        "rubric": [
            {"criterion": "正确实现 V'(s_i) = R(s_i) + γ·Σ_j P(s_j|s_i)·V(s_j)。", "points": 7},
            {"criterion": "保证同步：用同一份旧 values 算出全部新值，结果与遍历顺序无关。", "points": 5},
            {"criterion": "不修改传入 values，返回长度 N 的新列表；只用标准库。", "points": 3},
        ],
        "_solution": _solution_bellman_backup(),
    },
]

TOTAL_SCORE = sum(q["max_score"] for q in QUESTIONS)
assert TOTAL_SCORE == 100, "question bank must total 100 points"

SCORING = {
    "objective": sum(q["max_score"] for q in QUESTIONS if q["grading_mode"] == "auto"),
    "open_self_review": sum(q["max_score"] for q in QUESTIONS if q["grading_mode"] == "self"),
    "code_browser_tests": sum(q["max_score"] for q in QUESTIONS if q["grading_mode"] == "browser"),
}

_INDEX = {q["id"]: q for q in QUESTIONS}

_PUBLIC_COMMON = ("id", "type", "category", "title", "prompt", "max_score", "grading_mode")


def list_questions():
    """Return the full internal question list (includes hidden grading data)."""
    return list(QUESTIONS)


def get_question(question_id):
    """Return one internal question dict or raise UnknownQuestion."""
    try:
        return _INDEX[question_id]
    except KeyError:
        raise UnknownQuestion("unknown question id: %r" % (question_id,))


def get_public(question_id):
    """Return the public (answer-free) view of one question or raise UnknownQuestion."""
    return public_question(get_question(question_id))


def public_question(question):
    """Return only the fields safe to expose to the browser (no answers)."""
    out = {key: question[key] for key in _PUBLIC_COMMON}
    qtype = question["type"]
    if qtype == "choice":
        out["options"] = list(question["options"])
    elif qtype == "numeric":
        out["answer_type"] = question["answer_type"]
        out["tolerance"] = question["tolerance"]
        if "answer_count" in question:
            out["answer_count"] = question["answer_count"]
    elif qtype == "code":
        out["language"] = question["language"]
        out["entry_point"] = question["entry_point"]
        out["starter_code"] = question["starter_code"]
        out["tests"] = [
            {"name": t["name"], "description": t["description"], "code": t["code"]}
            for t in question["tests"]
        ]
    if "source_note" in question:
        out["source_note"] = dict(question["source_note"])
    if "review_hint" in question:
        out["review_hint"] = question["review_hint"]
    if "demo" in question:
        out["demo"] = question["demo"]
    return out


def public_payload():
    """Return the full public payload served by GET /api/questions."""
    return {
        "reference": REFERENCE,
        "conventions": CONVENTIONS,
        "presentation": PRESENTATION,
        "industry_case": INDUSTRY_CASE,
        "total_score": TOTAL_SCORE,
        "count": len(QUESTIONS),
        "scoring": dict(SCORING),
        "questions": [public_question(q) for q in QUESTIONS],
    }


def health():
    return {
        "status": "ok",
        "service": "rl-checkpoint",
        "reference": REFERENCE,
        "questions": len(QUESTIONS),
        "total_score": TOTAL_SCORE,
    }


def _normalize_choice(answer, options):
    if isinstance(answer, bool):
        raise InvalidAnswer("choice answer must be an index or an option letter")
    if isinstance(answer, int):
        index = answer
    elif isinstance(answer, float) and answer.is_integer():
        index = int(answer)
    elif isinstance(answer, str):
        text = answer.strip().upper()
        if len(text) == 1 and "A" <= text <= "Z":
            index = ord(text) - ord("A")
        elif text.isdigit():
            index = int(text)
        else:
            raise InvalidAnswer("choice answer must be an index or an option letter")
    else:
        raise InvalidAnswer("choice answer must be an index or an option letter")
    if index < 0 or index >= len(options):
        raise InvalidAnswer("choice index out of range")
    return index


def _as_float_list(answer):
    values = answer if isinstance(answer, (list, tuple)) else [answer]
    out = []
    for item in values:
        if isinstance(item, bool):
            raise InvalidAnswer("numeric answer must be a number or a list of numbers")
        if isinstance(item, (int, float)):
            val = float(item)
        elif isinstance(item, str):
            try:
                val = float(item.strip())
            except ValueError:
                raise InvalidAnswer("numeric answer must be a number or a list of numbers")
        else:
            raise InvalidAnswer("numeric answer must be a number or a list of numbers")
        if not math.isfinite(val):
            raise InvalidAnswer("numeric answer must be finite")
        out.append(val)
    if not out:
        raise InvalidAnswer("numeric answer must contain at least one number")
    return out


def grade(question_id, answer):
    """Auto-grade an objective question. Raises BankError subclasses otherwise."""
    question = get_question(question_id)
    qtype = question["type"]

    if qtype == "choice":
        index = _normalize_choice(answer, question["options"])
        correct = index == question["_answer"]
        letter = "ABCDEFG"[:len(question["options"])][question["_answer"]]
        return {
            "correct": correct,
            "score": question["max_score"] if correct else 0,
            "max_score": question["max_score"],
            "feedback": "回答正确。" if correct else "回答不正确，正确答案是 %s。" % letter,
            "expected": letter,
        }

    if qtype == "numeric":
        got = _as_float_list(answer)
        expected = question["_expected"]
        expected_list = list(expected) if question["answer_type"] == "list" else [expected]
        if len(got) != len(expected_list):
            raise InvalidAnswer("答案个数不符，期望 %d 个数" % len(expected_list))
        tolerance = question["tolerance"]
        correct = all(
            math.isclose(g, e, rel_tol=tolerance, abs_tol=tolerance)
            for g, e in zip(got, expected_list)
        )
        return {
            "correct": correct,
            "score": question["max_score"] if correct else 0,
            "max_score": question["max_score"],
            "feedback": "数值正确。" if correct else "数值不正确。",
            "expected": list(expected_list) if question["answer_type"] == "list" else expected_list[0],
        }

    if qtype == "code":
        raise NotAutoGradable("代码题在浏览器 Pyodide 中评测，请使用公开测试用例，服务器不做自动评分。")
    raise NotAutoGradable("开放题无自动评分，请调用 /api/reveal 获取评分要点后自评。")


def reveal(question_id):
    """Reveal rubric/model answer for self-assessment (open) or reference (code)."""
    question = get_question(question_id)
    qtype = question["type"]

    if qtype == "open":
        rubric = [dict(item) for item in question["rubric"]]
        return {
            "id": question["id"],
            "type": "open",
            "max_score": question["max_score"],
            "self_assessment": True,
            "rubric": rubric,
            "model_answer": question["model_answer"],
            "note": "开放题不自动评分；请对照评分要点自评。",
        }

    if qtype == "code":
        rubric = [dict(item) for item in question["rubric"]]
        return {
            "id": question["id"],
            "type": "code",
            "max_score": question["max_score"],
            "self_assessment": False,
            "entry_point": question["entry_point"],
            "reference_solution": question["_solution"],
            "rubric": rubric,
            "note": "代码题请在浏览器 Pyodide 中运行公开测试用例完成评测。",
        }

    raise NotRevealable("客观题请使用 /api/grade 进行自动评分。")
