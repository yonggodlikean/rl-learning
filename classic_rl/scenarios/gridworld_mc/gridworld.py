"""
4x4 网格世界迷宫环境
- 起点: (0, 0)
- 终点: (3, 3), 奖励 +10, 回合结束
- 陷阱: (1, 1), (2, 2), 奖励 -10, 回合结束
- 普通步: 奖励 -1
- 撞墙: 留在原地
"""

class GridWorld:
    ACTIONS = ["U", "D", "L", "R"]  # 上、下、左、右
    ACTION_DELTA = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1),
    }

    def __init__(self, size=4, goal=(3, 3), traps=None):
        self.size = size
        self.goal = goal
        self.traps = set(traps) if traps else {(1, 1), (2, 2)}
        self.reset()

    def reset(self):
        self.current_state = (0, 0)
        return self.get_state_str()

    def get_state_str(self, state=None):
        if state is None:
            state = self.current_state
        return f"{state[0]}_{state[1]}"

    def step(self, action):
        dr, dc = self.ACTION_DELTA[action]
        nr = max(0, min(self.size - 1, self.current_state[0] + dr))
        nc = max(0, min(self.size - 1, self.current_state[1] + dc))
        next_state = (nr, nc)
        self.current_state = next_state

        if next_state == self.goal:
            return self.get_state_str(), 10.0, True
        elif next_state in self.traps:
            return self.get_state_str(), -10.0, True
        else:
            return self.get_state_str(), -1.0, False
