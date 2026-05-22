# 竞赛简介

> **所属页面**：[Home](Home.md) > 竞赛简介

---

## 1. 比赛背景

**Orbit Wars** 是 Kaggle 平台上的一场实时策略 AI 竞赛。参赛者需要编写一个 Python Agent，在一张以太阳为中心的 100×100 星球地图上与对手博弈。比赛的核心挑战在于：

- **动态物理环境**：部分星球以 0.025–0.05 rad/turn 的角速度绕太阳公转，舰队飞行路径需要考虑太阳碰撞与行星运动的预测。
- **多对手对抗**：支持 2 人或 4 人对局，需要同时处理多方威胁与联合战术。
- **实时决策**：每回合计算时间受到严格限制（`remainingOverageTime` 预算），超时导致超额时间缩减直至禁用。
- **隐藏信息缺失**：所有信息均为全局可见（全观测博弈），胜负完全取决于策略质量。

---

## 2. 竞赛规格

| 属性 | 值 |
|------|----|
| 竞赛类型 | 代理对抗（Simulation / Multi-agent） |
| 支持玩家数 | 2 人或 4 人 |
| 单局回合数 | 500 回合 |
| Agent 接口 | `def agent(obs) -> list[move]` |
| 提交格式 | Python 脚本（`main.py`） |
| 评分算法 | Kaggle 内部 TrueSkill（初始分约 1000） |
| 获胜条件 | 终局时己方总船数（星球驻守 + 在途舰队）最多 |

---

## 3. Agent 接口规范

每个 Agent 是一个实现了以下签名的 Python 函数：

```python
def agent(obs) -> list[move]
```

### 输入：观测对象 `obs`

| 字段 | 类型 | 说明 |
|------|------|------|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` | 全图所有星球（含彗星） |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` | 全图所有在途舰队 |
| `player` | `int` | 己方玩家 ID（0–3） |
| `angular_velocity` | `float` | 轨道星球旋转角速度（rad/turn） |
| `initial_planets` | 同 `planets` | 本局游戏开始时的星球初始状态 |
| `comets` | `[{planet_ids, paths, path_index}, ...]` | 当前活跃彗星组数据 |
| `comet_planet_ids` | `[int, ...]` | 属于彗星的星球 ID 列表 |
| `remainingOverageTime` | `float` | 剩余超额时间预算（秒） |

### 输出：移动指令列表

每条指令格式：`[from_planet_id, direction_angle, num_ships]`

| 字段 | 说明 |
|------|------|
| `from_planet_id` | 发射来源星球 ID（必须是己方星球） |
| `direction_angle` | 发射方向（弧度，相对 x 轴正方向） |
| `num_ships` | 发射船数（不超过星球当前驻守数） |

---

## 4. 分数体系

竞赛 LB（排行榜）分数使用内部 TrueSkill 算法计算，1000 分为初始值。

本地平台使用调整参数的 TrueSkill，初始平均分设为 600（`mu₀ = 600, σ₀ = 200`），以保守分（`mu - 3σ`）排序，避免场次少的 Agent 虚高。

### 已知竞赛高分参考

| Agent | 最高 LB 分数 |
|-------|------------|
| tamrazov-starwars | **1224** |
| ykhnkf-distance-prioritized | ~1100 |
| OW-Proto (自研) | **1080**（峰值），稳定 1020–1050 |
| pilkwang-structured | ~1000 |
| sigmaborov-reinforce | 958.1 |

---

## 5. 竞赛策略路线图

```
基础理解阶段
  └─ 阅读游戏规则 → 跑通 Starter Kit → 理解观测格式

规则型 Agent 阶段
  └─ 实现贪心/狙击策略 → 加入物理预测 → 加入世界模型 → 多战术评分

强化学习阶段
  └─ 特征工程设计 → PPO 训练框架 → 自对弈 → 与规则型对抗验证
```

---

## 相关页面

- [游戏规则详解](Game-Rules.md)
- [Agent 全览](Agent-Zoo.md)
- [RL Agent 设计](RL-Agent.md)
