# core_solver.py - 【核心算法求解器】含收敛轨迹追踪 (History Tracking)
from schema import UserRetentionMDP

class RetentionDPSolver:
    """
    针对用户留存 MDP 的动态规划求解器
    对应核心理论：EasyRL §2.3（策略评估、策略迭代、值迭代）
    """
    def __init__(self, mdp: UserRetentionMDP):
        self.mdp = mdp
        self.num_states = len(mdp.states)
        self.num_actions = len(mdp.actions)

    def policy_evaluation(self, policy: list[int], theta: float = 1e-6) -> list[float]:
        """
        策略评估 (Policy Evaluation)
        """
        values = [0.0] * self.num_states
        while True:
            delta = 0.0
            new_values = [0.0] * self.num_states
            for s in range(self.num_states):
                a = policy[s]
                expected_future = 0.0
                for s_prime in range(self.num_states):
                    expected_future += self.mdp.transitions[s][a][s_prime] * values[s_prime]
                new_values[s] = self.mdp.rewards[s][a] + self.mdp.gamma * expected_future
                delta = max(delta, abs(new_values[s] - values[s]))
            values = new_values
            if delta < theta:
                break
        return values

    def policy_iteration_with_history(self) -> tuple[list[float], list[int], list[list[float]]]:
        """
        策略迭代（记录每轮策略改进后的状态价值向量 V_history）
        """
        policy = [0] * self.num_states
        history = []
        while True:
            values = self.policy_evaluation(policy)
            history.append(list(values))
            policy_stable = True
            for s in range(self.num_states):
                q_values = [
                    self.mdp.rewards[s][a] + self.mdp.gamma * sum(
                        self.mdp.transitions[s][a][s_prime] * values[s_prime]
                        for s_prime in range(self.num_states)
                    )
                    for a in range(self.num_actions)
                ]
                best_a = q_values.index(max(q_values))
                if best_a != policy[s]:
                    policy_stable = False
                    policy[s] = best_a
            if policy_stable:
                return values, policy, history

    def policy_iteration(self) -> tuple[list[float], list[int], int]:
        values, policy, history = self.policy_iteration_with_history()
        return values, policy, len(history)

    def value_iteration_with_history(self, theta: float = 1e-6) -> tuple[list[float], list[int], list[list[float]]]:
        """
        值迭代（记录每轮贝尔曼最优算子更新后的状态价值向量 V_history）
        """
        values = [0.0] * self.num_states
        history = [list(values)]
        while True:
            delta = 0.0
            new_values = [0.0] * self.num_states
            for s in range(self.num_states):
                q_values = [
                    self.mdp.rewards[s][a] + self.mdp.gamma * sum(
                        self.mdp.transitions[s][a][s_prime] * values[s_prime]
                        for s_prime in range(self.num_states)
                    )
                    for a in range(self.num_actions)
                ]
                new_values[s] = max(q_values)
                delta = max(delta, abs(new_values[s] - values[s]))
            values = new_values
            history.append(list(values))
            if delta < theta:
                break
        policy = [0] * self.num_states
        for s in range(self.num_states):
            q_values = [
                self.mdp.rewards[s][a] + self.mdp.gamma * sum(
                    self.mdp.transitions[s][a][s_prime] * values[s_prime]
                    for s_prime in range(self.num_states)
                )
                for a in range(self.num_actions)
            ]
            policy[s] = q_values.index(max(q_values))
        return values, policy, history

    def value_iteration(self, theta: float = 1e-6) -> tuple[list[float], list[int], int]:
        values, policy, history = self.value_iteration_with_history(theta)
        return values, policy, len(history) - 1
