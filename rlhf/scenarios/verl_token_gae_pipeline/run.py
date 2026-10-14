# run.py - 一键运行检验与 verl 风格控制台报表输出
import sys
from schema import load_verl_gsm8k_samples
from core_solver import VerlMiniEngine

def main():
    print("=" * 75)
    print("🚀 verl 框架实战：大模型推理链路的规则打分与 GAE 优势计算")
    print("=" * 75)

    samples = load_verl_gsm8k_samples()
    gamma = 0.5
    lam = 0.95

    for sample in samples:
        print(f"\n📋 正在处理样本: {sample.id} | Prompt: \"{sample.prompt}\"")
        print(f"   生成 Token 序列: {sample.tokens}")
        print(f"   有效掩码 Mask  : {sample.response_mask}")

        # 1. 验证任务 1：规则奖励分发
        try:
            rewards = VerlMiniEngine.extract_and_assign_rewards(sample)
            print(f"   ✅ [Task 1 规则打分通过] 稀疏奖励分布 R: {rewards}")
            # 基础验证：奖励应仅在有效位置末尾出现
            valid_indices = [i for i, m in enumerate(sample.response_mask) if m == 1]
            last_idx = valid_indices[-1]
            assert rewards[last_idx] == 1.0, f"样本 {sample.id} 最终回答正确，末个有效位必须给 1.0 奖励！"
            assert sum(rewards) == 1.0, "除末尾有效 token 外，中间 token 不应获得即时奖励！"
        except NotImplementedError:
            print("   ⚠️ 待完成：请在 core_solver.py 中实现 extract_and_assign_rewards！")
            sys.exit(0)

        # 2. 验证任务 2：GAE 与 Returns
        try:
            deltas, advantages, returns = VerlMiniEngine.compute_gae_and_returns(
                token_rewards=rewards,
                critic_values=sample.critic_values,
                response_mask=sample.response_mask,
                gamma=gamma,
                lam=lam
            )
            print(f"   ✅ [Task 2 GAE 优势计算通过]")
            print("\n   📊 Token 级多维价值流转对照表:")
            print("   " + "-" * 65)
            print(f"   {'Token':<8} | {'Mask':<4} | {'Critic V':<8} | {'Reward':<6} | {'TD δ':<8} | {'GAE Adv':<8} | {'Return G'}")
            print("   " + "-" * 65)
            for i, tok in enumerate(sample.tokens):
                m = sample.response_mask[i]
                v = sample.critic_values[i]
                r = rewards[i]
                d = deltas[i]
                a = advantages[i]
                g = returns[i]
                print(f"   {tok:<8} | {m:<4} | {v:<8.3f} | {r:<6.1f} | {d:<8.3f} | {a:<8.3f} | {g:<.3f}")
            print("   " + "-" * 65)

            # 严格数值断言与 Padding 保护测试
            for i, m in enumerate(sample.response_mask):
                if m == 0:
                    assert advantages[i] == 0.0, "Padding 位置的优势值必须严格为 0！"
                    assert returns[i] == 0.0, "Padding 位置的目标回报必须严格为 0！"
                else:
                    assert abs(returns[i] - (advantages[i] + sample.critic_values[i])) < 1e-6, "G_t 必须严格满足 G_t = A_t + V_t！"
        except NotImplementedError:
            print("   ⚠️ 待完成：请在 core_solver.py 中实现 compute_gae_and_returns！")
            sys.exit(0)

    print("\n" + "=" * 75)
    print("🏆 恭喜！你已亲手复现了 verl 核心算法链路，彻底打通经典 RL 价值论与大模型对齐实战！")
    print("=" * 75)

if __name__ == "__main__":
    main()
