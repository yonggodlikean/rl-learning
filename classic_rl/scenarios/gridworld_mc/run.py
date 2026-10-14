"""
完整训练与监控脚本：
跟踪并分析随着 Episode 更新，起点 s0 的各个动作价值 Q(s0, a) 和状态价值 V(s0) 的演变。
"""
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np

from core import epsilon_greedy_probs, mc_action_values
from gridworld import GridWorld


def run_episode(env, Q, epsilon, max_steps=50):
    """根据当前 Q 和 epsilon 采样一条完整回合轨迹"""
    s_str = env.reset()
    pairs = []
    rewards = []

    for _ in range(max_steps):
        q_vals = [Q.get((s_str, a), 0.0) for a in GridWorld.ACTIONS]
        probs = epsilon_greedy_probs(q_vals, epsilon)
        action = random.choices(GridWorld.ACTIONS, weights=probs)[0]

        next_s_str, r, done = env.step(action)
        pairs.append((s_str, action))
        rewards.append(r)
        s_str = next_s_str
        if done:
            break

    return {"pairs": pairs, "rewards": rewards}


def main():
    random.seed(42)
    np.random.seed(42)

    env = GridWorld(size=4)
    gamma = 0.9
    num_episodes = 2000

    total_returns = {}
    N = {}
    Q = {}

    s0 = "0_0"  # 起点状态字符串

    # 监控指标记录列表
    history_q_s0 = {a: [] for a in GridWorld.ACTIONS}
    history_v_s0 = []
    history_v_greedy = []
    history_episode_rewards = []
    history_episode_lengths = []

    print(f"=== 开始 Monte Carlo ε-greedy 训练 (共 {num_episodes} 回合) ===")

    for ep_idx in range(1, num_episodes + 1):
        # 探索率 epsilon 从 0.6 线性衰减到 0.05
        epsilon = max(0.05, 0.6 * (1.0 - ep_idx / 1500))

        # 1. 采样一回合
        ep = run_episode(env, Q, epsilon)
        history_episode_rewards.append(sum(ep["rewards"]))
        history_episode_lengths.append(len(ep["pairs"]))

        # 2. 蒙特卡洛更新 Q(s, a)
        Q, N = mc_action_values([ep], gamma, total_returns, N, visit="first")

        # 3. 记录当前时刻 s0 处的各动作价值与状态价值
        q_s0_vals = [Q.get((s0, a), 0.0) for a in GridWorld.ACTIONS]
        probs_s0 = epsilon_greedy_probs(q_s0_vals, epsilon)

        # 期望状态价值 V(s0) = sum(pi(a|s0) * Q(s0, a))
        v_exp = sum(p * q for p, q in zip(probs_s0, q_s0_vals))
        v_greedy = max(q_s0_vals)

        for a, q_val in zip(GridWorld.ACTIONS, q_s0_vals):
            history_q_s0[a].append(q_val)
        history_v_s0.append(v_exp)
        history_v_greedy.append(v_greedy)

    print("=== 训练完成，正在输出统计数据与生成可视化图表 ===")

    # 打印特定阶段的指标表格
    checkpoints = [1, 50, 200, 500, 1000, 1500, 2000]
    print("\n------------------------- 起点 s0 价值演变关键里程碑 -------------------------")
    print(f"{'回合(Ep)':<8}{'Q(s0, U)':<12}{'Q(s0, D)':<12}{'Q(s0, L)':<12}{'Q(s0, R)':<12}{'V(s0)期望':<12}{'V(s0)贪心':<10}")
    print("-" * 78)
    for ep in checkpoints:
        idx = ep - 1
        q_u = history_q_s0["U"][idx]
        q_d = history_q_s0["D"][idx]
        q_l = history_q_s0["L"][idx]
        q_r = history_q_s0["R"][idx]
        v_exp = history_v_s0[idx]
        v_g = history_v_greedy[idx]
        print(f"{ep:<8}{q_u:<12.2f}{q_d:<12.2f}{q_l:<12.2f}{q_r:<12.2f}{v_exp:<12.2f}{v_g:<10.2f}")
    print("-" * 78)

    # 绘制高清对比图表
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图 1: 起点 s0 的 4 个动作价值 Q(s0, a) 随 Episode 演变
    ax1 = axes[0, 0]
    ep_range = range(1, num_episodes + 1)
    colors = {"U": "#4A90E2", "D": "#E74C3C", "L": "#95A5A6", "R": "#2ECC71"}
    labels = {"U": "Q(s0, 上/U) 撞墙", "D": "Q(s0, 下/D) 靠近陷阱", "L": "Q(s0, 左/L) 撞墙", "R": "Q(s0, 右/R) 通向终点"}
    for a in GridWorld.ACTIONS:
        ax1.plot(ep_range, history_q_s0[a], label=labels[a], color=colors[a], linewidth=2.0)
    ax1.set_title("起点 s0 的动作价值 Q(s0, a) 演化过程", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode (回合数)", fontsize=11)
    ax1.set_ylabel("动作价值 Q", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # 子图 2: 起点状态价值 V(s0) 演变（期望值 vs 贪婪最大值）
    ax2 = axes[0, 1]
    ax2.plot(ep_range, history_v_greedy, label="贪婪状态价值 max_a Q(s0,a)", color="#27AE60", linewidth=2.2)
    ax2.plot(ep_range, history_v_s0, label="ε-greedy 策略期望价值 V(s0)", color="#F39C12", linestyle="--", linewidth=2.0)
    ax2.set_title("起点状态价值 V(s0) 演化过程", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode (回合数)", fontsize=11)
    ax2.set_ylabel("状态价值 V", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    # 子图 3: 每回合累计奖励与步数平滑曲线
    ax3 = axes[1, 0]
    # 滑动平均
    window = 30
    smooth_rewards = np.convolve(history_episode_rewards, np.ones(window)/window, mode="valid")
    smooth_lengths = np.convolve(history_episode_lengths, np.ones(window)/window, mode="valid")
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(range(window, num_episodes + 1), smooth_rewards, color="#8E44AD", label="回合总回报(30步滑动平均)", linewidth=2.0)
    l2 = ax3_twin.plot(range(window, num_episodes + 1), smooth_lengths, color="#D35400", linestyle=":", label="回合步数(30步滑动平均)", linewidth=2.0)
    ax3.set_title("训练过程：回报收敛与步数缩短", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Episode (回合数)", fontsize=11)
    ax3.set_ylabel("累计回报 (Return)", color="#8E44AD", fontsize=11)
    ax3_twin.set_ylabel("回合完成步数", color="#D35400", fontsize=11)
    # 合并图例
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="lower right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 子图 4: 最终网格世界的最优策略与最大状态价值热力图
    ax4 = axes[1, 1]
    v_grid = np.zeros((4, 4))
    arrow_grid = [["" for _ in range(4)] for _ in range(4)]
    arrow_symbols = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

    for r in range(4):
        for c in range(4):
            pos = (r, c)
            if pos == env.goal:
                v_grid[r, c] = 10.0
                arrow_grid[r][c] = "GOAL"
            elif pos in env.traps:
                v_grid[r, c] = -10.0
                arrow_grid[r][c] = "TRAP"
            else:
                s_str = f"{r}_{c}"
                q_vals = [Q.get((s_str, a), -99.0) for a in GridWorld.ACTIONS]
                best_idx = max(range(4), key=lambda i: q_vals[i])
                v_grid[r, c] = q_vals[best_idx]
                arrow_grid[r][c] = arrow_symbols[GridWorld.ACTIONS[best_idx]]

    im = ax4.imshow(v_grid, cmap="YlGnBu", origin="upper")
    fig.colorbar(im, ax=ax4, label="最大状态价值 V*(s)")
    ax4.set_title("最终各状态的最优策略箭头与价值", fontsize=13, fontweight="bold")
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))

    for r in range(4):
        for c in range(4):
            ax4.text(c, r, f"{arrow_grid[r][c]}\n({v_grid[r, c]:.1f})",
                     ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    plt.tight_layout()
    chart_path = Path(__file__).resolve().with_name("mc_learning_curves.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n[OK] 可视化图表已保存至: {chart_path}")


if __name__ == "__main__":
    main()
