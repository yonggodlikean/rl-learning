# schema.py - 用户留存营销决策 MDP 数据契约与环境配置
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class UserRetentionMDP:
    """
    用户生命周期与营销补贴的已知马尔可夫决策过程 (MDP)
    """
    states: List[str]            # 状态名列表，长度 S
    actions: List[str]           # 动作名列表，长度 A
    rewards: List[List[float]]   # 即时净收益 R[s][a]，形状 [S, A]
    transitions: List[List[List[float]]] # 状态转移概率 P[s][a][s']，形状 [S, A, S]
    gamma: float = 0.5           # 折扣因子

def load_default_retention_mdp() -> UserRetentionMDP:
    """
    基础 2 状态玩具案例 (对应教材原版):
      - 状态: [候选用户, 留存用户]
      - 动作: [普通运营, 强力激励]
    """
    return UserRetentionMDP(
        states=["候选用户", "留存用户"],
        actions=["普通运营", "强力激励"],
        rewards=[
            [1.0, 1.0],  # 候选用户
            [2.0, 1.0],  # 留存用户
        ],
        transitions=[
            [[0.5, 0.5], [0.2, 0.8]],  # 候选用户
            [[0.1, 0.9], [0.0, 1.0]],  # 留存用户
        ],
        gamma=0.5
    )

def load_complex_retention_mdp() -> UserRetentionMDP:
    """
    工业级复杂化算例：
      - 5 个细分生命周期状态:
          0: '新潜客 (New)'
          1: '初阶活跃 (Active)'
          2: '高价值忠诚 (VIP)'
          3: '沉睡预警 (Dormant)'
          4: '已流失 (Churned, 吸收态/自旋)'
      - 3 种干预动作:
          0: '不干预/日常推送 (No Action)' (成本 0)
          1: '轻量权益/签到奖励 (Light Promo)' (成本 0.5)
          2: '大额满减券/VIP专线 (Heavy Promo)' (成本 2.0)
      - 折扣因子设为 0.95 (更符合长生命周期终身价值 LTV 计算，考验多步收敛能力)
    """
    states = ["新潜客", "初阶活跃", "高价值忠诚", "沉睡预警", "已流失"]
    actions = ["不干预", "轻量权益", "大额优惠券"]
    
    # 状态转移概率 P[s][a][s_prime]
    P_0 = [
        [0.4, 0.2, 0.0, 0.4, 0.0],
        [0.3, 0.5, 0.1, 0.1, 0.0],
        [0.1, 0.6, 0.3, 0.0, 0.0]
    ]
    P_1 = [
        [0.0, 0.5, 0.2, 0.3, 0.0],
        [0.0, 0.5, 0.4, 0.1, 0.0],
        [0.0, 0.3, 0.7, 0.0, 0.0]
    ]
    P_2 = [
        [0.0, 0.1, 0.8, 0.1, 0.0],
        [0.0, 0.05, 0.9, 0.05, 0.0],
        [0.0, 0.0, 0.95, 0.05, 0.0]
    ]
    P_3 = [
        [0.0, 0.0, 0.0, 0.4, 0.6],
        [0.0, 0.3, 0.0, 0.4, 0.3],
        [0.0, 0.7, 0.1, 0.1, 0.1]
    ]
    P_4 = [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.0, 0.0, 0.0, 0.9],
        [0.3, 0.0, 0.0, 0.0, 0.7]
    ]
    
    transitions = [P_0, P_1, P_2, P_3, P_4]
    
    rewards = [
        [1.0, 0.8, -0.5],   # 新客
        [4.0, 3.8, 2.5],    # 活跃
        [10.0, 9.5, 7.5],   # VIP (过度补贴惩罚)
        [0.2, -0.2, -1.5],  # 沉睡 (必须大额投入才能挽回)
        [0.0, -0.5, -2.0]   # 流失
    ]
    
    return UserRetentionMDP(
        states=states,
        actions=actions,
        rewards=rewards,
        transitions=transitions,
        gamma=0.95
    )
