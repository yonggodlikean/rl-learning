# run_benchmark.py - 复杂 5 状态全生命周期 MDP 测试与两种迭代算法收敛速度可视化
import os
import sys
import math
import matplotlib.pyplot as plt

# 导入基础组件
from schema import UserRetentionMDP, load_default_retention_mdp
from core_solver import RetentionDPSolver

def load_complex_retention_mdp() -> UserRetentionMDP:
    """
    更加复杂且真实的 5 状态全生命周期用户运营 MDP：
      - 状态空间 (5 个状态):
          0: '新注册用户' (初始流量，对促活敏感)
          1: '高活跃核心' (平台主力，带来高自然广告与消费分成)
          2: '普通活跃客' (日常使用，可能逐渐沉睡)
          3: '轻度沉睡客' (活跃度显著下滑，急需召回)
          4: '彻底流失客' (极难唤醒，接近吸收状态)
      - 动作空间 (3 种营销干预):
          0: '静默运营' (0 成本，仅自然浏览)
          1: '轻量推送' (低成本，如站内信+5元首充券)
          2: '深度大促' (高成本，如专属客服+30元大额直减)
      - 折扣因子: gamma = 0.9 (放眼长期留存价值)
    """
    states = ["新注册用户", "高活跃核心", "普通活跃客", "轻度沉睡客", "彻底流失客"]
    actions = ["静默运营", "轻量推送", "深度大促"]
    
    # 即时收益矩阵 R[s][a] (业务收益扣除运营/补贴成本后的净收益)
    rewards = [
        # 新注册用户: 深度大促成本虽高，但能锁定次留；静默运营收益低
        [0.5, 1.2, 1.0],
        # 高活跃核心: 自然产生高贡献 5.0；发小券 4.5；发大额补贴被薅羊毛，净利降为 3.0
        [5.0, 4.5, 3.0],
        # 普通活跃客: 自然产生 2.0；发小券净利提升至 2.5；大额券成本过高 1.5
        [2.0, 2.5, 1.5],
        # 轻度沉睡客: 自然流失几乎无收益 0.2；轻量推送能激活一部分 1.0；深度大促能强力召回 1.8
        [0.2, 1.0, 1.8],
        # 彻底流失客: 自然收益 0.0；发小券基本无响应 -0.2；深度大促有极小概率召回 0.5
        [0.0, -0.2, 0.5],
    ]
    
    # 状态转移概率矩阵 P[s][a][s'] (5 x 3 x 5)
    # 转移到: [新注册, 核心活跃, 普通活跃, 沉睡, 彻底流失]
    transitions = [
        # 0: 新注册用户
        [
            [0.1, 0.1, 0.4, 0.2, 0.2],  # 静默: 大概率沦为普通或沉睡
            [0.05, 0.3, 0.45, 0.15, 0.05], # 轻量推送: 核心比例提升
            [0.0, 0.6, 0.3, 0.1, 0.0],  # 深度大促: 60%概率直接转化为核心
        ],
        # 1: 高活跃核心
        [
            [0.0, 0.8, 0.15, 0.05, 0.0],  # 静默: 80%继续核心，15%降为普通
            [0.0, 0.85, 0.12, 0.03, 0.0], # 轻量推送: 维持 85%
            [0.0, 0.9, 0.1, 0.0, 0.0],   # 深度大促: 90%维持
        ],
        # 2: 普通活跃客
        [
            [0.0, 0.1, 0.5, 0.3, 0.1],   # 静默: 仅10%升核心，30%沉睡
            [0.0, 0.3, 0.5, 0.15, 0.05], # 轻量推送: 30%升核心
            [0.0, 0.5, 0.4, 0.1, 0.0],   # 深度大促: 50%升核心
        ],
        # 3: 轻度沉睡客
        [
            [0.0, 0.0, 0.1, 0.4, 0.5],   # 静默: 50%彻底流失
            [0.0, 0.05, 0.35, 0.4, 0.2], # 轻量推送: 35%召回为普通活跃
            [0.0, 0.25, 0.5, 0.2, 0.05], # 深度大促: 75%成功唤醒
        ],
        # 4: 彻底流失客
        [
            [0.0, 0.0, 0.0, 0.05, 0.95], # 静默: 95%停留在流失槽
            [0.0, 0.0, 0.05, 0.15, 0.8], # 轻量推送: 20%微弱召回
            [0.0, 0.05, 0.2, 0.25, 0.5], # 深度大促: 50%被激活回到各阶段
        ]
    ]
    
    return UserRetentionMDP(
        states=states,
        actions=actions,
        rewards=rewards,
        transitions=transitions,
        gamma=0.9
    )

def track_policy_iteration(solver, theta=1e-6):
    """带轨迹记录的策略迭代"""
    policy = [0] * solver.num_states
    iteration_steps = 0
    pi_history = []
    total_eval_sweeps = 0
    
    while True:
        iteration_steps += 1
        values = [0.0] * solver.num_states
        eval_sweeps = 0
        while True:
            eval_sweeps += 1
            delta = 0.0
            new_values = [0.0] * solver.num_states
            for s in range(solver.num_states):
                a = policy[s]
                expected = sum(solver.mdp.transitions[s][a][s_p] * values[s_p] for s_p in range(solver.num_states))
                new_values[s] = solver.mdp.rewards[s][a] + solver.mdp.gamma * expected
                delta = max(delta, abs(new_values[s] - values[s]))
            values = new_values
            if delta < theta:
                break
        total_eval_sweeps += eval_sweeps
        pi_history.append({
            "iter": iteration_steps,
            "values": list(values),
            "policy": list(policy),
            "total_sweeps": total_eval_sweeps
        })
        
        policy_stable = True
        for s in range(solver.num_states):
            q_values = [
                solver.mdp.rewards[s][a] + solver.mdp.gamma * sum(solver.mdp.transitions[s][a][s_p] * values[s_p] for s_p in range(solver.num_states))
                for a in range(solver.num_actions)
            ]
            best_a = q_values.index(max(q_values))
            if best_a != policy[s]:
                policy_stable = False
                policy[s] = best_a
        if policy_stable:
            break
            
    return values, policy, iteration_steps, pi_history

def track_value_iteration(solver, theta=1e-6):
    """带轨迹记录的值迭代"""
    values = [0.0] * solver.num_states
    steps = 0
    vi_history = []
    
    while True:
        steps += 1
        delta = 0.0
        new_values = [0.0] * solver.num_states
        current_greedy_policy = [0] * solver.num_states
        for s in range(solver.num_states):
            q_values = [
                solver.mdp.rewards[s][a] + solver.mdp.gamma * sum(solver.mdp.transitions[s][a][s_p] * values[s_p] for s_p in range(solver.num_states))
                for a in range(solver.num_actions)
            ]
            new_values[s] = max(q_values)
            current_greedy_policy[s] = q_values.index(new_values[s])
            delta = max(delta, abs(new_values[s] - values[s]))
            
        values = new_values
        vi_history.append({
            "step": steps,
            "values": list(values),
            "delta": delta,
            "policy": current_greedy_policy
        })
        if delta < theta:
            break
            
    policy = [0] * solver.num_states
    for s in range(solver.num_states):
        q_values = [
            solver.mdp.rewards[s][a] + solver.mdp.gamma * sum(solver.mdp.transitions[s][a][s_p] * values[s_p] for s_p in range(solver.num_states))
            for a in range(solver.num_actions)
        ]
        policy[s] = q_values.index(max(q_values))
        
    return values, policy, steps, vi_history

def plot_comparison(mdp, pi_history, vi_history, save_path):
    """绘制收敛速度与价值演进对比图"""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    # 1. 左图：值迭代每步的最大误差衰减曲线 (log scale)
    deltas = [h["delta"] for h in vi_history]
    axes[0].plot(range(1, len(deltas) + 1), deltas, marker="o", markersize=3, color="#d9534f", label="值迭代误差 (Max |V_k+1 - V_k|)")
    axes[0].set_yscale("log")
    axes[0].axhline(y=1e-6, color="#5bc0de", linestyle="--", label="收敛阈值 theta = 1e-6")
    axes[0].set_title("值迭代 (Value Iteration) 误差指数收敛曲线", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("更新轮数 (Iterations)", fontsize=11)
    axes[0].set_ylabel("最大价值变动 (Log Scale)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # 2. 右图：两种算法对各状态终身价值 V(s) 的逼近轨迹
    colors = ["#2b5c8f", "#d9534f", "#2ca02c", "#ff7f0e", "#9467bd"]
    for s_idx, s_name in enumerate(mdp.states):
        vi_v_s = [h["values"][s_idx] for h in vi_history]
        axes[1].plot(range(1, len(vi_v_s) + 1), vi_v_s, color=colors[s_idx], label=f"值迭代: {s_name}", alpha=0.8, linewidth=1.8)
        
        pi_sweeps = [h["total_sweeps"] for h in pi_history]
        pi_v_s = [h["values"][s_idx] for h in pi_history]
        axes[1].scatter(pi_sweeps, pi_v_s, color=colors[s_idx], s=70, edgecolors="black", zorder=5, label=f"策略迭代: {s_name}" if s_idx==0 else None)
        
    axes[1].set_title(f"各状态价值 V*(s) 演进轨迹 (gamma={mdp.gamma})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("等价 Bellman 遍历次数 (Total Sweeps)", fontsize=11)
    axes[1].set_ylabel("状态终身价值 V(s)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n📊 对比图表已成功生成并保存至: {save_path}")

def main():
    print("=" * 75)
    print("🌟 升级测试：5 状态用户全生命周期运营 MDP (策略迭代 vs 值迭代)")
    print("=" * 75)
    
    complex_mdp = load_complex_retention_mdp()
    # 适配 num_states 属性
    complex_mdp.num_states = len(complex_mdp.states)
    complex_mdp.num_actions = len(complex_mdp.actions)
    
    solver = RetentionDPSolver(complex_mdp)
    
    print(f"\n[环境配置]")
    print(f"  • 状态空间 (5): {complex_mdp.states}")
    print(f"  • 动作空间 (3): {complex_mdp.actions}")
    print(f"  • 长期折扣因子 gamma: {complex_mdp.gamma}")
    
    # 1. 运行带轨迹的策略迭代
    print("\n[算法 1] 运行策略迭代 (Policy Iteration)...")
    pi_v, pi_policy, pi_steps, pi_history = track_policy_iteration(solver)
    total_sweeps = pi_history[-1]["total_sweeps"]
    print(f"  ✅ 策略迭代完成！")
    print(f"  • 策略升级轮数: {pi_steps} 轮")
    print(f"  • 累计底层贝尔曼扫描: {total_sweeps} 次")
    
    # 2. 运行带轨迹的值迭代
    print("\n[算法 2] 运行值迭代 (Value Iteration)...")
    vi_v, vi_policy, vi_steps, vi_history = track_value_iteration(solver)
    print(f"  ✅ 值迭代完成！")
    print(f"  • 贝尔曼最优算子迭代次数: {vi_steps} 步")
    
    # 3. 策略与价值对比验证
    print("\n" + "=" * 75)
    print(f"{'用户生命周期状态':<10} | {'策略迭代推荐决策':<12} | {'值迭代推荐决策':<12} | {'最终最优期望价值 V*':<15}")
    print("-" * 75)
    for s in range(complex_mdp.num_states):
        st_name = complex_mdp.states[s]
        pi_act = complex_mdp.actions[pi_policy[s]]
        vi_act = complex_mdp.actions[vi_policy[s]]
        v_star = pi_v[s]
        print(f"{st_name:<10} | {pi_act:<12} | {vi_act:<12} | {v_star:<.4f}")
    print("-" * 75)
    
    assert pi_policy == vi_policy, "双算法推导出的最优策略应当一致！"
    for v1, v2 in zip(pi_v, vi_v):
        assert abs(v1 - v2) < 1e-4, "双算法计算出的最优状态价值应当一致！"
    print("🎯 双算法结论完全一致！")
    
    # 4. 业务洞察解读
    print("\n💡 业务策略洞见 (为什么算法会学出这样的策略？):")
    print(f"  1. 【{complex_mdp.states[0]}】-> 选【{complex_mdp.actions[pi_policy[0]]}】: 新用户首充大促能高效锁住向高价值核心转化！")
    print(f"  2. 【{complex_mdp.states[1]}】-> 选【{complex_mdp.actions[pi_policy[1]]}】: 核心用户自然留存极高，静默运营利润最大化，杜绝被薅羊毛！")
    print(f"  3. 【{complex_mdp.states[2]}】-> 选【{complex_mdp.actions[pi_policy[2]]}】: 普通活跃客给予小额优惠即可激发活跃！")
    print(f"  4. 【{complex_mdp.states[3]}】-> 选【{complex_mdp.actions[pi_policy[3]]}】: 沉睡流失边缘，轻量推送唤不醒，必须下重注大额权益强力召回！")
    print(f"  5. 【{complex_mdp.states[4]}】-> 选【{complex_mdp.actions[pi_policy[4]]}】: 彻底流失客唤醒难度极大，算法发现唯有深度大促才有一线生机！")

    # 5. 生成可视化图表
    chart_path = os.path.join(os.path.dirname(__file__), "convergence_comparison.png")
    plot_comparison(complex_mdp, pi_history, vi_history, chart_path)

if __name__ == "__main__":
    main()
