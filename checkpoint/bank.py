"""Question bank for the local RL checkpoint (browser-side, EasyRL v1.0.6 §2.1–§2.2.2).

Scope is intentionally frozen at concept/numeric/open-ended questions about
Markov processes, Markov reward processes, returns and the MRP Bellman equation,
plus two pure-Python coding exercises that run in a browser Pyodide worker.

This module is dependency-free (stdlib only) and performs NO server-side
execution of user code. Objective questions are auto-graded here; open-ended
questions are self-assessed from a rubric; coding exercises are graded entirely
in the browser.
"""

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
                'non_tensor_batch = {\n'
                '    **{k: np.array([v] * n) for k, v in kwargs.items()},\n'
                '    "__num_turns__": np.array([o.num_turns for o in outputs]),\n'
                '    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),\n'
                '    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),\n'
                '    "response_len": np.array([len(o.response_ids) for o in outputs]),\n'
                '}\n'
                'result = await selected_reward_loop_worker_handle.compute_score.remote(data)\n'
                'final_output.reward_score = result["reward_score"]'
            ),
            "code_note": "两处真实源码节选；DataProto 构造与 worker 选择在原文件中间，未表示为可直接执行的一段代码。",
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
            "lines": "35–43, 54–55, 66–83, 97–99",
            "code": (
                'data_source = data_item.non_tensor_batch["data_source"]\n'
                'ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]\n'
                'response_str = await self.loop.run_in_executor(\n'
                '    None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)\n'
                ')\n'
                'return {"reward_score": reward, "reward_extra_info": reward_extra_info}'
            ),
            "code_note": "非连续节选；实际评分调用 compute_score(data_source=data_source, solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info) 见 66–83 行。",
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
            "code": (
                'solutions = re.findall("#### (\\\\-?[0-9\\\\.\\\\,]+)", solution_str)\n'
                'answer = extract_solution(solution_str=solution_str, method=method)\n'
                'if answer is None:\n'
                '    return 0\n'
                'else:\n'
                '    if answer == ground_truth:\n'
                '        return score\n'
                '    else:\n'
                '        return format_score'
            ),
            "code_note": "摘取评分规则与函数主体；源码中默认 score=1、format_score=0，入口路由见 reward_score/__init__.py:44–47。",
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
    return out


def public_payload():
    """Return the full public payload served by GET /api/questions."""
    return {
        "reference": REFERENCE,
        "conventions": CONVENTIONS,
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
