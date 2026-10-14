# 4x4 网格世界：基于 Monte Carlo ε-Greedy 的控制与价值演化

本项目完整整合了强化学习检查点（Checkpoint）中的核心算法实现：
- **题 16**：`discounted_returns`（有限轨迹全部折扣回报逆向递推）
- **题 17**：`mc_action_values`（基于 First-visit / Every-visit 的多回合动作价值估计）
- **题 18**：`epsilon_greedy_probs`（探索与利用权衡的 ε-Greedy 动作概率生成）

## 目录结构
- `core.py`: 蒙特卡洛纯算法实现（对应题 16、17、18）
- `gridworld.py`: 4x4 悬崖迷宫网格世界环境
- `run.py`: 800 回合策略迭代主脚本，跟踪 s0 处动作价值与状态价值的动态演变，并生成图表
- `mc_learning_curves.png`: 训练收敛曲线与策略热力图

## 快速运行
```bash
python3 run.py
```
