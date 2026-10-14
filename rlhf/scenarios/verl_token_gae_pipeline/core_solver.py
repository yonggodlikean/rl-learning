# core_solver.py - 【核心考核文件】大模型推理的规则奖励路由与 GAE 优势计算
import re
from schema import VerlTrajectory

class VerlMiniEngine:
    """
    极简还原字节跳动开源大模型强化学习框架 verl (commit a2ad9f6)
    核心数学与算法实现：
      - 规则打分与稀疏奖励映射 (verl/utils/reward_score/gsm8k.py)
      - 广义优势估计与回报倒序累积 (verl/trainer/ppo/core_algos.py)
    """

    @staticmethod
    def extract_and_assign_rewards(traj: VerlTrajectory) -> list[float]:
        """
        【考核任务 1：规则奖励分发】
        对应源码：verl/utils/reward_score/gsm8k.py & agent_loop.py
        
        逻辑要求:
        1. 检查生成的 token 序列，拼接有效文本；
        2. 查找 '####' 之后的提取结果并 strip()；
        3. 如果提取结果 == traj.ground_truth，则得分为 +1.0，否则为 0.0；
        4. 【关键】：在大模型生成中，即时奖励是极度稀疏的！
           仅在【最后一个有效 token】（即 response_mask 为 1 的最靠右位置）赋予该得分；
           其余所有位置的奖励均为 0.0。
           
        返回:
            rewards: 长度与 traj.tokens 相同的浮点数列表
        """
        # TODO: 【请你在此补全奖励提取与末位赋值逻辑】
        # 提示:
        # 1. 初始化 rewards = [0.0] * len(traj.tokens)
        # 2. 找到最后一个 mask == 1 的索引 last_valid_idx
        # 3. 拼接 tokens 字符串，用正则 r"####\s*([^\s]+)" 提取答案
        # 4. 若匹配成功且 answer == traj.ground_truth，rewards[last_valid_idx] = 1.0
        # 5. 返回 rewards
        raise NotImplementedError("请补全任务 1：规则奖励分发 (extract_and_assign_rewards)")

    @staticmethod
    def compute_gae_and_returns(token_rewards: list[float],
                                critic_values: list[float],
                                response_mask: list[int],
                                gamma: float = 0.5,
                                lam: float = 0.95) -> tuple[list[float], list[float], list[float]]:
        """
        【考核任务 2：GAE 广义优势估计与回报倒序计算】
        对应源码：verl/trainer/ppo/core_algos.py (compute_gae_advantage_return)
        与 EasyRL §2.3 的内在联系：
          经典单步 TD 残差: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
          GAE 指数衰减累加: A_t = delta_t + gamma * lam * A_{t+1}
          目标回报自举: G_t = A_t + V(s_t)
          
        算法要求:
        1. 从最右侧最后一个 token 开始逆序（倒序）循环：t = T-1, T-2, ..., 0
        2. 注意后继价值 next_value:
           - 若 t == T-1，后继状态视为终止状态，next_value = 0.0
           - 否则，next_value = critic_values[t+1]
        3. 计算当前步的时序差分残差:
           delta_t = token_rewards[t] + gamma * next_value - critic_values[t]
        4. 计算累积 GAE 优势:
           adv_t = delta_t + gamma * lam * last_gae
        5. 【核心掩码保护】：若当前位置无效 (response_mask[t] == 0)，
           不仅 adv_t 必须归零，倒序传递的 last_gae 也必须清零！
        6. 目标回报: returns[t] = adv_t + critic_values[t] (有效位)
        
        返回:
            (deltas, advantages, returns) 三者均为长度为 T 的浮点数列表
        """
        # TODO: 【请你在此补全 GAE 倒序递推算法】
        # 提示:
        # T = len(token_rewards)
        # deltas = [0.0] * T
        # advantages = [0.0] * T
        # returns = [0.0] * T
        # last_gae = 0.0
        # for t in reversed(range(T)):
        #     if response_mask[t] == 0:
        #         last_gae = 0.0
        #         continue
        #     next_val = 0.0 if t == T - 1 or response_mask[t+1] == 0 else critic_values[t+1]
        #     delta = token_rewards[t] + gamma * next_val - critic_values[t]
        #     adv = delta + gamma * lam * last_gae
        #     deltas[t] = delta
        #     advantages[t] = adv
        #     returns[t] = adv + critic_values[t]
        #     last_gae = adv
        # return deltas, advantages, returns
        raise NotImplementedError("请补全任务 2：GAE 倒序递推算法 (compute_gae_and_returns)")
