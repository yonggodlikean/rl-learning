# schema.py - verl 风格的极简数据协议与样本定义
from dataclasses import dataclass
from typing import List

@dataclass
class VerlTrajectory:
    """
    模拟 verl 框架中 DataProto 在单个大模型推理轨迹上的核心切片
    
    字段解析:
      - prompt: 提问，如数学题 "2 + 2 = ?"
      - tokens: 生成的序列（含思考步骤与最终答案标记）
      - response_mask: 有效 token 的布尔掩码 (1 代表有效，0 代表 padding 填充)
      - critic_values: Critic 模型给出的每步状态价值估计 V(s_t)，形状与 tokens 一致
      - ground_truth: 标准答案，用于规则打分
    """
    id: str
    prompt: str
    tokens: List[str]
    response_mask: List[int]
    critic_values: List[float]
    ground_truth: str

def load_verl_gsm8k_samples() -> List[VerlTrajectory]:
    """
    构造两个典型的 verl 大模型推理批次样本：
      样本 A: 解题过程正确，最终给出 #### 4，长度 4，无 padding
      样本 B: 解题过程正确，最终给出 #### 2，长度 2，尾部含 2 个 padding
    """
    return [
        VerlTrajectory(
            id="Sample_A",
            prompt="2 + 2 = ?",
            tokens=["先算", "，", "####", " 4"],
            response_mask=[1, 1, 1, 1],
            critic_values=[0.20, 0.35, 0.50, 0.65],
            ground_truth="4"
        ),
        VerlTrajectory(
            id="Sample_B",
            prompt="1 + 1 = ?",
            tokens=["####", " 2", "<pad>", "<pad>"],
            response_mask=[1, 1, 0, 0],
            critic_values=[0.30, 0.60, 0.0, 0.0],
            ground_truth="2"
        ),
    ]
