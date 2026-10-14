"""
网格世界蒙特卡洛算法核心实现
对应检查点：
- 题 16: discounted_returns
- 题 17: mc_action_values
- 题 18: epsilon_greedy_probs
"""
import math

def discounted_returns(rewards, gamma):
    """题 16：有限轨迹的全部折扣回报（从后往前反向递推）"""
    T = len(rewards)
    returns = [0.0] * T
    total = 0.0
    for t in range(T - 1, -1, -1):
        total = rewards[t] + gamma * total
        returns[t] = total
    return returns


def epsilon_greedy_probs(q_values, epsilon):
    """题 18：根据 Q 值计算 epsilon-greedy 概率分布"""
    n = len(q_values)
    best_idx = max(range(n), key=lambda i: q_values[i])
    probs = [epsilon / n] * n
    probs[best_idx] += (1.0 - epsilon)
    return probs


def mc_action_values(episodes, gamma, total_returns, N, visit="first"):
    """
    题 17：按 first/every visit 累计动作价值函数 Q(s, a)
    支持增量维护全局 total_returns 和 N
    """
    for ep in episodes:
        pairs = ep["pairs"]
        rewards = ep["rewards"]
        T = len(pairs)
        if T == 0:
            continue

        returns = discounted_returns(rewards, gamma)
        seen = set()

        for t in range(T):
            sa = tuple(pairs[t])
            if visit == "first":
                if sa in seen:
                    continue
                seen.add(sa)

            total_returns[sa] = total_returns.get(sa, 0.0) + returns[t]
            N[sa] = N.get(sa, 0) + 1

    Q = {sa: total_returns[sa] / N[sa] for sa in N}
    return Q, N
