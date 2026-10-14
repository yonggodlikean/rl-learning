# run.py - 支持复杂 5 状态 3 动作算例求解与收敛速度曲线绘图
import os
import sys
import math
from schema import load_default_retention_mdp, load_complex_retention_mdp
from core_solver import RetentionDPSolver

def plot_convergence(pi_history, vi_history, v_opt, states, save_path="convergence_comparison.png"):
    """
    绘制策略迭代与值迭代的价值误差收敛曲线对比图
    """
    import matplotlib.pyplot as plt
    # 设置学术级排版风格与字体
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Songti SC", "Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 计算每轮与最终 V* 的无穷范数误差: max_s |V_k(s) - V*(s)|
    def compute_errors(history):
        errors = []
        for v in history:
            err = max(abs(v[s] - v_opt[s]) for s in range(len(v_opt)))
            errors.append(max(err, 1e-12)) # 避免 log(0)
        return errors

    pi_errors = compute_errors(pi_history)
    vi_errors = compute_errors(vi_history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 图 1：对数尺度下的误差收敛速度对比 (Log Error vs Iteration Steps)
    ax1.plot(range(len(pi_errors)), pi_errors, 'o-', color='#C0392B', linewidth=2.2, markersize=7, label='策略迭代 (Policy Iteration)')
    ax1.plot(range(len(vi_errors)), vi_errors, 's--', color='#2980B9', linewidth=2.0, markersize=4, label='值迭代 (Value Iteration)')
    ax1.set_yscale('log')
    ax1.set_xlabel('迭代轮数 (Iteration Steps)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'与最优价值的距离 $\max_s |V_k(s) - V^*(s)|$ (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('收敛速度对比：策略迭代 vs 值迭代', fontsize=14, pad=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11, frameon=True)

    # 图 2：5 个状态在值迭代过程中的价值演化曲线 (State Value Evolution in VI)
    colors = ['#8E44AD', '#27AE60', '#D35400', '#2C3E50', '#7F8C8D']
    for s_idx in range(len(states)):
        val_traj = [v[s_idx] for v in vi_history]
        ax2.plot(range(len(vi_history)), val_traj, label=f'状态: {states[s_idx]}', color=colors[s_idx % len(colors)], linewidth=2)
    ax2.set_xlabel('值迭代步数 (VI Steps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('状态价值估计 V(s)', fontsize=12, fontweight='bold')
    ax2.set_title('值迭代各状态价值函数演化过程', fontsize=14, pad=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n📊 [图表生成成功] 收敛速度对比高清曲线图已保存至: {save_path}")

def run_experiment(mdp, case_name="基础 2 状态算例"):
    print("\n" + "=" * 75)
    print(f"🚀 EasyRL §2.3 动态规划实战：【{case_name}】策略迭代与值迭代求解")
    print(f"   状态数: {len(mdp.states)} | 动作数: {len(mdp.actions)} | 折扣因子 γ: {mdp.gamma}")
    print("=" * 75)

    solver = RetentionDPSolver(mdp)

    # 1. 运行策略迭代
    pi_v_opt, pi_policy, pi_history = solver.policy_iteration_with_history()
    print(f"\n1️⃣ 【策略迭代 (Policy Iteration)】")
    print(f"   - 收敛轮数: {len(pi_history)} 轮")
    print("   - 最优策略与终身价值:")
    for s_idx, state_name in enumerate(mdp.states):
        act_name = mdp.actions[pi_policy[s_idx]]
        print(f"     * [{state_name}]: 推荐动作 => 【{act_name}】 (V* = {pi_v_opt[s_idx]:.4f})")

    # 2. 运行值迭代
    vi_v_opt, vi_policy, vi_history = solver.value_iteration_with_history(theta=1e-6)
    print(f"\n2️⃣ 【值迭代 (Value Iteration)】")
    print(f"   - 贝尔曼最优算子迭代步数: {len(vi_history) - 1} 步")
    
    # 验证一致性
    assert pi_policy == vi_policy, "策略迭代与值迭代得出的最优决策必须一致！"
    for v1, v2 in zip(pi_v_opt, vi_v_opt):
        assert abs(v1 - v2) < 1e-4, "策略迭代与值迭代价值必须一致！"
    print("   ✅ 校验一致：两种迭代算法输出的最优决策和终身价值完全吻合！")

    return pi_history, vi_history, pi_v_opt

def main():
    # 场景 1：基础教材案例
    mdp_base = load_default_retention_mdp()
    run_experiment(mdp_base, "教材基础 2 状态算例")

    # 场景 2：工业级复杂多状态营销决策案例
    mdp_complex = load_complex_retention_mdp()
    pi_hist, vi_hist, v_opt = run_experiment(mdp_complex, "工业级 5 状态 3 动作生命周期营销算例")

    # 绘制对比曲线
    plot_file = os.path.join(os.path.dirname(__file__), "convergence_comparison.png")
    plot_convergence(pi_hist, vi_hist, v_opt, mdp_complex.states, save_path=plot_file)

    print("\n" + "=" * 75)
    print("🏆 恭喜！复杂化算例与收敛对比全流程运行圆满完成！")
    print("=" * 75)

if __name__ == "__main__":
    main()
